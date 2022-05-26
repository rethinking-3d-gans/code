from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.torch_utils import misc
from src.torch_utils import persistence
from src.torch_utils.ops import conv2d_resample
from src.torch_utils.ops import upfirdn2d
from src.torch_utils.ops import bias_act

#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x: torch.Tensor, dim=1, eps=1e-8) -> torch.Tensor:
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        activation         = 'linear', # Activation function: 'relu', 'lrelu', etc.
        bias               = True,     # Apply additive bias before the activation function?
        lr_multiplier      = 1,        # Learning rate multiplier.
        weight_init        = 1,        # Initial standard deviation of the weight tensor.
        bias_init          = 0,        # Initial value of the additive bias.
        num_hyper_classes  = 1,      # For hyper-networks with class-wise conditioning
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([num_hyper_classes, out_features, in_features]) * (weight_init / lr_multiplier))
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [num_hyper_classes, out_features])
        self.bias = torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x, hyper_class_idx: torch.Tensor=None):
        # Selecting the weights
        if hyper_class_idx is None:
            w = self.weight[0] # [c_out, c_in]
            b = self.bias[0] if not self.bias is None else None # [c_out]
        else:
            assert hyper_class_idx.max() < self.weight.shape[1], f'poor: {hyper_class_idx} {self.weight.shape}'
            w = self.weight[hyper_class_idx] # [batch_size, c_out, c_in]
            b = self.bias[hyper_class_idx] if not self.bias is None else None # [batch_size, c_out]

        # Adjusting the scales
        w = w.to(x.dtype) * self.weight_gain # [c_out, c_in] or [batch_size, c_out, c_in]
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        # Applying the weights
        if hyper_class_idx is None:
            if self.activation == 'linear' and b is not None:
                x = torch.addmm(b.unsqueeze(0), x, w.t())
            else:
                x = x.matmul(w.t())
                x = bias_act.bias_act(x, b, act=self.activation)
        else:
            x = (w @ x.unsqueeze(2)).squeeze(2) # [batch_size, c_out]
            x = (x + b) if b is None else x # [batch_size, c_out]
            if self.activation != 'linear':
                dummy_b = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype) # [c_out]
                x = bias_act.bias_act(x, dummy_b, act=self.activation)

        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                         # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                         # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                         # Intermediate latent (W) dimensionality.
        num_ws,                        # Number of intermediate latents to output, None = do not broadcast.
        num_layers         = 2,        # Number of mapping layers.
        embed_features     = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features     = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation         = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier      = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta         = 0.998,    # Decay for tracking the moving average of W during training, None = do not track.
        num_instances      = 0,        # If we use instance_id for reconstruction losses
        camera_cond        = False,    # Camera conditioning
        camera_cond_drop_p = 0.0,      # Camera conditioning dropout
        mean_camera_pose   = None,     # Average camera pose for use at test time.
        **layer_kwargs,                # Arguments for FC layers
    ):
        super().__init__()
        if camera_cond:
            self.camera_scalar_enc = ScalarEncoder1d(coord_dim=3, x_multiplier=16.0, const_emb_dim=0)
            c_dim = c_dim + self.camera_scalar_enc.get_dim()
            assert self.camera_scalar_enc.get_dim() > 0
        else:
            self.camera_scalar_enc = None

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta
        self.camera_cond_drop_p = camera_cond_drop_p

        if embed_features is None:
            embed_features = w_dim
        if self.c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if self.c_dim > 0:
            self.embed = FullyConnectedLayer(self.c_dim, embed_features, **layer_kwargs)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier, **layer_kwargs)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

        if num_instances > 0:
            self.inst_embs = nn.Embedding(num_instances, self.z_dim)
        else:
            self.inst_embs = None

        if not mean_camera_pose is None:
            self.register_buffer('mean_camera_pose', mean_camera_pose)

    def forward(self, z, c, camera_angles: torch.Tensor=None, instance_id: torch.Tensor=None, truncation_psi=1,
                      truncation_cutoff=None, update_emas=False, **layer_kwargs):
        """
        instance_id: [batch_size] --- one hot labels
        """
        if (not self.camera_scalar_enc is None) and (not self.training) and (camera_angles is None):
            camera_angles = self.mean_camera_pose.unsqueeze(0).repeat(len(z), 1) # [batch_size, 3]

        if not self.camera_scalar_enc is None:
            camera_angles = camera_angles % (2.0 * np.pi) / (2.0 * np.pi) # [batch_size, 3]
            camera_angles[camera_angles < 0.0] = 2 * np.pi - camera_angles[camera_angles < 0.0] # [...any...]
            camera_angles_embs = self.camera_scalar_enc(camera_angles) # [batch_size, fourier_dim]
            camera_angles_embs = F.dropout(camera_angles_embs, p=self.camera_cond_drop_p, training=self.training) # [batch_size, fourier_dim]
            c = torch.zeros(len(camera_angles_embs), 0, device=camera_angles_embs.device) if c is None else c # [batch_size, c_dim]
            c = torch.cat([c, camera_angles_embs], dim=1) # [batch_size, c_dim + angle_emb_dim]

        if not instance_id is None:
            misc.assert_shape(instance_id, [len(z)])
            z = self.inst_embs(instance_id) # [batch_size, z_dim]
        elif not self.inst_embs is None:
            z = z + 0.0 * self.inst_embs(torch.tensor([0], device=z.device).long()) # [batch_size, z_dim]
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32), **layer_kwargs))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x, **layer_kwargs)

        # Update moving average of W.
        if update_emas and self.w_avg_beta is not None:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                      # Number of input channels.
        out_channels,                     # Number of output channels.
        kernel_size,                      # Width and height of the convolution kernel.
        bias              = True,         # Apply additive bias before the activation function?
        activation        = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up                = 1,            # Integer upsampling factor.
        down              = 1,            # Integer downsampling factor.
        resample_filter   = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp        = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last     = False,        # Expect the input to have memory_format=channels_last?
        trainable         = True,         # Update the weights of this layer during training?
        num_hyper_classes = 1,            # If it's hyperconditioned, then
        c_dim             = 0,            # Passing c via re-normalization?
        renorm     = False,        # Should we use instance norm?
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([num_hyper_classes, out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([num_hyper_classes, out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

        if renorm:
            assert c_dim > 0, f"Why using instance norm if c_dim == {c_dim}?"
            assert num_hyper_classes == 1, "Not supported"
            self.affine = FullyConnectedLayer(c_dim, in_channels, bias_init=0)
        else:
            self.affine = None

    def forward(self, x, c: torch.Tensor=None, gain=1, hyper_class_idx=None):
        batch_size, c_out, c_in = x.shape[0], self.weight.shape[1], self.weight.shape[2]
        if hyper_class_idx is None:
            w = self.weight[0] * self.weight_gain # [c_out, c_in, k, k]
            groups = 1
        else:
            misc.assert_shape(hyper_class_idx, [x.shape[0]])
            w = self.weight[hyper_class_idx] * self.weight_gain # [batch_size, c_out, c_in, k, k]
            w = w.reshape(-1, c_in, w.shape[3], w.shape[4]) # [batch_size * c_out, c_in, k, k]
            x = x.reshape(1, batch_size * c_in, *x.shape[2:]) # [1, batch_size * c_in, h, w]
            groups = batch_size
        flip_weight = (self.up == 1) # slightly faster
        if not self.affine is None:
            weights = 1.0 + self.affine(c).tanh().unsqueeze(2).unsqueeze(3) # [batch_size, c_in, 1, 1]
            x = (x * weights).to(x.dtype) # [batch_size, c_out, h, w]
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight, groups=groups)
        if not hyper_class_idx is None:
            # We applied the convolutional layer in a group-wise fashion and need to reshape activations back
            x = x.reshape(batch_size, c_out, *x.shape[2:]) # [batch_size, c_out, h, w]
        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        if hyper_class_idx is None:
            b = self.bias[0].to(x.dtype) if self.bias is not None else None
            x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        elif not self.bias is None:
            # Apply the bias
            b = self.bias[hyper_class_idx].to(x.dtype) # [batch_size, c_out]
            x = x + b.view(batch_size, c_out, 1, 1) # [batch_size, c_out, h, w]
            dummy_b = torch.zeros(c_out, device=x.device, dtype=x.dtype) # [c_out]
            # Apply the activation
            x = bias_act.bias_act(x, dummy_b, act=self.activation, gain=act_gain, clamp=act_clamp) # [batch_size, c_out]
        return x

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, activation={self.activation:s},',
            f'up={self.up}, down={self.down}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class ScalarEncoder1d(nn.Module):
    """
    1-dimensional Fourier Features encoder (i.e. encodes raw scalars)
    Assumes that scalars are in [0, 1]
    """
    def __init__(self, coord_dim: int, x_multiplier: float, const_emb_dim: int, **fourier_enc_kwargs):
        super().__init__()
        self.coord_dim = coord_dim
        self.const_emb_dim = const_emb_dim
        self.x_multiplier = x_multiplier
        if self.const_emb_dim > 0:
            self.const_embed = nn.Embedding(int(np.ceil(x_multiplier)) + 1, self.const_emb_dim)
        else:
            self.const_embed = None
        self.fourier_encoder = FourierEncoder1d(coord_dim, max_x_value=x_multiplier, **fourier_enc_kwargs)
        self.fourier_dim = self.fourier_encoder.get_dim()

    def get_dim(self) -> int:
        return self.coord_dim * (self.const_emb_dim + self.fourier_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Assumes that x is in [0, 1] range
        """
        misc.assert_shape(x, [None, self.coord_dim])

        batch_size, coord_dim = x.shape
        # Convert from [0, 1] to the [0, `x_multiplier`] range
        x = x.float() * self.x_multiplier # [batch_size, coord_dim]
        fourier_embs = self.fourier_encoder(x) # [batch_size, coord_dim, fourier_dim]
        if self.const_embed is None:
            out = fourier_embs # [batch_size, coord_dim, fourier_dim]
        else:
            const_embs = self.const_embed(x.round().long()) # [batch_size, coord_dim, const_emb_dim]
            out = torch.cat([const_embs, fourier_embs], dim=2) # [batch_size, coord_dim, const_emb_dim + fourier_dim]
        out = out.view(batch_size, coord_dim * (self.const_emb_dim + self.fourier_dim)) # [batch_size, coord_dim * (const_emb_dim + fourier_dim)]

        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class FourierEncoder1d(nn.Module):
    def __init__(self,
            coord_dim: int,               # Number of scalars to encode for each sample
            max_x_value: float=100.0,       # Maximum scalar value (influences the amount of fourier features we use)
            transformer_pe: bool=False,     # Whether we should use positional embeddings from Transformer
            use_cos: bool=True,
            **construct_freqs_kwargs,
        ):
        super().__init__()
        assert coord_dim >= 1, f"Wrong coord_dim: {coord_dim}"
        self.coord_dim = coord_dim
        self.use_cos = use_cos
        if transformer_pe:
            d_model = 512
            fourier_coefs = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)) # [d_model]
        else:
            fourier_coefs = construct_log_spaced_freqs(max_x_value, **construct_freqs_kwargs)
        self.register_buffer('fourier_coefs', fourier_coefs) # [num_fourier_feats]
        self.fourier_dim = self.fourier_coefs.shape[0]

    def get_dim(self) -> int:
        return self.fourier_dim * (2 if self.use_cos else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, f"Wrong shape: {x.shape}"
        assert x.shape[1] == self.coord_dim
        fourier_raw_embs = self.fourier_coefs.view(1, 1, self.fourier_dim) * x.float().unsqueeze(2) # [batch_size, coord_dim, fourier_dim]
        if self.use_cos:
            fourier_embs = torch.cat([fourier_raw_embs.sin(), fourier_raw_embs.cos()], dim=2) # [batch_size, coord_dim, 2 * fourier_dim]
        else:
            fourier_embs = fourier_raw_embs.sin() # [batch_size, coord_dim, fourier_dim]
        return fourier_embs

#----------------------------------------------------------------------------

def construct_log_spaced_freqs(max_t: int, skip_small_t_freqs: int=0, skip_large_t_freqs: int=0) -> Tuple[int, torch.Tensor]:
    time_resolution = 2 ** np.ceil(np.log2(max_t))
    num_fourier_feats = np.ceil(np.log2(time_resolution)).astype(int)
    powers = torch.tensor([2]).repeat(num_fourier_feats).pow(torch.arange(num_fourier_feats)) # [num_fourier_feats]
    powers = powers[skip_large_t_freqs:len(powers) - skip_small_t_freqs] # [num_fourier_feats]
    fourier_coefs = powers.float() * np.pi # [1, num_fourier_feats]

    return fourier_coefs / time_resolution

#----------------------------------------------------------------------------
