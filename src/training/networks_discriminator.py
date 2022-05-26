from typing import Dict, List
import numpy as np
import torch
from omegaconf import DictConfig
from src.torch_utils import misc
from src.torch_utils import persistence
from src.torch_utils.ops import upfirdn2d

from src.training.networks_stylegan2 import SynthesisLayer
from src.training.layers import (
    FullyConnectedLayer,
    MappingNetwork,
    Conv2dLayer,
    ScalarEncoder1d,
)
from src.training.utils import compute_patch_coords


#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        cfg: DictConfig,                    # Main config.
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
        down                = 2,            # Downsampling factor
        c_dim               = 0,            # Hyper-conditioning dimension
        renorm              = False,        # Should we use instance norm in Conv2dLayer?
        **layer_kwargs,
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation, c_dim=c_dim, renorm=False,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)

        if self.cfg.hyper_type in ['hyper', 'dummy_hyper']:
            assert next(trainable_iter)
            assert c_dim > 0, f"Non need to use hyper-modulation here"
            self.conv0 = SynthesisLayer(
                tmp_channels, tmp_channels, w_dim=c_dim, resolution=self.resolution, kernel_size=3, activation=activation,
                conv_clamp=conv_clamp, channels_last=self.channels_last, use_noise=False)
        elif self.cfg.hyper_type == 'no_hyper':
            self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation, c_dim=c_dim, renorm=False,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        else:
            raise NotImplementedError("Unknown hyper type:", self.cfg.hyper_type)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=down, c_dim=c_dim, renorm=renorm,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=down, c_dim=c_dim, renorm=False,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last, **layer_kwargs)

    def forward(self, x, img, c: torch.Tensor=None, force_fp32=False, **layer_kwargs):
        if (x if x is not None else img).device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img, c=c, **layer_kwargs)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Defining hyper-modulation parameters.
        if self.cfg.hyper_type == 'hyper':
            conv0_kwargs = {'w': c}
            assert len(layer_kwargs) == 1 and 'hyper_class_idx' in layer_kwargs, f"Unsupported kwargs: {layer_kwargs}"
        elif self.cfg.hyper_type == 'dummy_hyper':
            conv0_kwargs = {'w': c * 0.0}
            assert len(layer_kwargs) == 1 and 'hyper_class_idx' in layer_kwargs, f"Unsupported kwargs: {layer_kwargs}"
        elif self.cfg.hyper_type == 'no_hyper':
            conv0_kwargs = dict(**layer_kwargs)
            conv0_kwargs['c'] = c # [batch_size, c_dim]
        else:
            raise NotImplementedError("Unknwon hyper type", self.cfg.hyper_type)

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, c=c, gain=np.sqrt(0.5), **layer_kwargs)
            x = self.conv0(x, **conv0_kwargs)
            x = self.conv1(x, c=c, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, **conv0_kwargs)
            x = self.conv1(x, c=c, **layer_kwargs)

        assert x.dtype == dtype
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        batch_size, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(batch_size)).item() if self.group_size is not None else batch_size
        F = self.num_channels
        c = C // F
        num_groups = batch_size // G

        y = x.reshape(G, num_groups, F, c, H, W)    # [group_size, num_groups, F, c, H, W]  Split minibatch batch_size into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)                       # [group_size, num_groups, F, c, H, W]  Subtract mean over group.
        y = y.square().mean(dim=0)                  # [num_groups, F, c, H, W]              Calc variance over group.
        y = (y + 1e-8).sqrt()                       # [num_groups, F, c, H, W]              Calc stddev over group.
        y = y.mean(dim=[2,3,4])                     # [num_groups, F]                       Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)                  # [num_groups, F, 1, 1]                 Add missing dimensions.
        y = y.repeat(G, 1, H, W)                    # [batch_size, F, H, W]                 Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)                # [N, C + 1, H, W]                      Append to input as new channels.
        return x

    def extra_repr(self):
        return f'group_size={self.group_size}, num_channels={self.num_channels:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        pose_pred: bool     = False,    # Should we predict poses?
        **layer_kwargs,                 # Arguments for Conv2dLayer
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation, **layer_kwargs)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp, **layer_kwargs)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), out_features=in_channels, activation=activation, **layer_kwargs)
        self.out = FullyConnectedLayer(in_channels, out_features=(1 if cmap_dim == 0 else cmap_dim), **layer_kwargs)

        if pose_pred:
            self.pose_pred_net = FullyConnectedLayer(in_channels * (resolution ** 2), out_features=2, **layer_kwargs)
        else:
            self.pose_pred_net = None

    def forward(self, x, img, cmap, force_fp32=False, **layer_kwargs):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img, **layer_kwargs)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x, **layer_kwargs)
        x = x.flatten(1) # [batch_size, in_channels * (resolution ** 2)]
        pose_preds = None if self.pose_pred_net is None else self.pose_pred_net(x) # [batch_size, 2]
        x = self.fc(x, **layer_kwargs) # [batch_size, in_channels]
        x = self.out(x, **layer_kwargs) # [batch_size, 1 or cmap_dim]

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x, pose_preds

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
        cfg: DictConfig,                # Main config.
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()

        self.cfg = cfg
        self.c_dim = c_dim
        self.img_resolution = img_resolution * (2 ** self.cfg.num_additional_start_blocks)
        self.img_resolution_log2 = int(np.log2(self.img_resolution))
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]

        # Concatenating coordinates to the input
        assert self.cfg.patch.discr_concat_coords_strategy in [None, 'raw', 'fourier'], f"Unknown concat coords strategy: {self.cfg.patch.discr_concat_coords_strategy}"
        if self.cfg.patch.discr_concat_coords_strategy == 'fourier':
            self.coords_encoder = ScalarEncoder1d(coord_dim=2, x_multiplier=100.0, const_emb_dim=16)
            self.img_channels = img_channels + self.coords_encoder.get_dim()
        elif self.cfg.patch.discr_concat_coords_strategy == 'raw':
            self.img_channels = img_channels + 2
        else:
            self.img_channels = img_channels

        if self.cfg.patch.patch_params_cond > 0:
            self.scalar_enc = ScalarEncoder1d(coord_dim=3, x_multiplier=1000.0, const_emb_dim=256)
            assert self.scalar_enc.get_dim() > 0
        else:
            self.scalar_enc = None

        if (self.c_dim == 0) and (self.scalar_enc is None) and (not self.cfg.camera_cond):
            cmap_dim = 0

        if self.cfg.patch.discr_scales_hyper_cond:
            self.patch_to_hyper_idx = {f'{s:.02f}': i for i, s in enumerate(self.cfg.patch.discrete_support)}
            num_hyper_classes = len(self.patch_to_hyper_idx)
        else:
            num_hyper_classes = 1

        if self.cfg.renorm:
            renorm_dim = 512
            self.renorm_mapping = MappingNetwork(
                z_dim=0, c_dim=self.scalar_enc.get_dim(), camera_cond=False, camera_cond_drop_p=0.0,
                w_dim=renorm_dim, num_ws=None, w_avg_beta=None, num_hyper_classes=num_hyper_classes, **mapping_kwargs)
        else:
            self.renorm_mapping = None
            renorm_dim = 0

        common_kwargs = dict(img_channels=self.img_channels, architecture=architecture, conv_clamp=conv_clamp)
        total_conditioning_dim = c_dim + (0 if self.scalar_enc is None else self.scalar_enc.get_dim())
        cur_layer_idx = 0

        for i, res in enumerate(self.block_resolutions):
            in_channels = channels_dict[res] if res < self.img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            down = 1 if i < self.cfg.num_additional_start_blocks else 2
            block = DiscriminatorBlock(
                cfg, in_channels, tmp_channels, out_channels, resolution=res, first_layer_idx=cur_layer_idx, use_fp16=use_fp16,
                num_hyper_classes=num_hyper_classes, down=down, c_dim=renorm_dim, renorm=self.cfg.renorm, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

        if self.c_dim > 0 or not self.scalar_enc is None:
            self.mapping = MappingNetwork(
                z_dim=0, c_dim=total_conditioning_dim, camera_cond=self.cfg.camera_cond, camera_cond_drop_p=self.cfg.camera_cond_drop_p,
                w_dim=cmap_dim, num_ws=None, w_avg_beta=None, num_hyper_classes=num_hyper_classes, **mapping_kwargs)
        else:
            self.mapping = None
        self.b4 = DiscriminatorEpilogue(
            channels_dict[4], cmap_dim=cmap_dim, resolution=4, num_hyper_classes=num_hyper_classes,
            pose_pred=self.cfg.predict_pose_loss_weight > 0, **epilogue_kwargs, **common_kwargs)

    def get_hyper_class_idx(self, patch_params: Dict[str, torch.Tensor], device: str='cpu'):
        """
        Takes patch parameters and decided which hyperclasses should they be
        """
        hyper_class_idx = [self.patch_to_hyper_idx[k] for k in self.get_patch_hyper_keys(patch_params)] # [batch_size]
        hyper_class_idx = torch.tensor(hyper_class_idx).to(device) # [batch_size]
        return hyper_class_idx

    def get_patch_hyper_keys(self, patch_params: Dict[str, torch.Tensor]) -> List[str]:
        return [f'{s:.02f}' for s in patch_params['scales'][:, 0].cpu().tolist()]

    def compute_patch_keys_to_idx(self, patch_params: Dict[str, torch.Tensor]) -> Dict[str, int]:
        patch_keys = self.get_patch_hyper_keys(patch_params) # (batch_size)
        patch_keys_to_idx = {s: [] for s in set(patch_keys)}
        for i, s in enumerate(patch_keys):
            patch_keys_to_idx[s].append(i)
        return patch_keys_to_idx

    def forward(self, img, c, patch_params: torch.Tensor=None, camera_angles: torch.Tensor=None, update_emas=False, **block_kwargs):
        _ = update_emas # unused
        batch_size, _, h, w = img.shape

        if not self.scalar_enc is None:
            patch_scales = patch_params['scales'] # [batch_size, 2]
            patch_offsets = patch_params['offsets'] # [batch_size, 2]
            patch_params_cond = torch.cat([patch_scales[:, [0]], patch_offsets], dim=1) # [batch_size, 3]
            misc.assert_shape(patch_params_cond, [batch_size, 3])
            patch_scale_embs = self.scalar_enc(patch_params_cond) # [batch_size, fourier_dim]
            c = torch.cat([c, patch_scale_embs], dim=1) # [batch_size, c_dim + fourier_dim]

        if self.cfg.patch.discr_scales_hyper_cond and self.training:
            assert not patch_params is None
            hyper_class_idx = self.get_hyper_class_idx(patch_params, device=img.device)
        else:
            hyper_class_idx = None

        if not self.cfg.patch.discr_concat_coords_strategy is None:
            coords = compute_patch_coords(patch_params, img.shape[2]) # [batch_size, h, w, 2]
            if self.cfg.patch.discr_concat_coords_strategy == 'fourier':
                coords = coords.reshape(batch_size * h * w, 2) * 0.5 + 0.5 # [batch_size * h * w, 2]
                coord_embs = self.coords_encoder(coords).reshape(batch_size, h, w, self.coords_encoder.get_dim()) # [batch_size, h, w, emb_dim]
                coord_embs = coord_embs.permute(0, 3, 1, 2) # [batch_size, emb_dim, h, w]
            else:
                coord_embs = coords.permute(0, 3, 1, 2) # [batch_size, 2, h, w]
            img = torch.cat([img, coord_embs], dim=1) # [batch_size, num_channels + 2, h, w]

        if not self.renorm_mapping is None:
            renorm_c = self.renorm_mapping(z=None, c=patch_scale_embs, hyper_class_idx=hyper_class_idx) # [batch_size, 512]
        else:
            renorm_c = None

        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, c=renorm_c, hyper_class_idx=hyper_class_idx, **block_kwargs)

        if self.c_dim > 0 or not self.scalar_enc is None:
            assert c.shape[1] > 0
        if not self.mapping is None:
            cmap = self.mapping(z=None, c=c, camera_angles=camera_angles, hyper_class_idx=hyper_class_idx) # [TODO]
        else:
            cmap = None
        x, pose_preds = self.b4(x, img, cmap, hyper_class_idx=hyper_class_idx)
        x = x.squeeze(1) # [batch_size]
        misc.assert_shape(x, [batch_size])

        if not pose_preds is None:
            # For DDP consistency
            x = x + pose_preds.max() * 0.0 # [batch_size]

        return x, pose_preds

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'

#----------------------------------------------------------------------------
