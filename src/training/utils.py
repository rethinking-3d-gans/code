from typing import Tuple, Dict, Callable, Any, List
import torch
import torch.nn.functional as F
from scipy.stats import truncnorm
import numpy as np
from src.torch_utils import misc

#----------------------------------------------------------------------------

def linear_schedule(step: int, val_start: float, val_end: float, period: int) -> float:
    """
    Returns the current value from `val_from` to `val_to` if the linear growth over `period` steps is used
    If out of range, then returns the boundary value
    """
    if step >= period:
        return val_end
    elif step <= 0:
        return val_start
    else:
        return val_start + (val_end - val_start) * step / period

#----------------------------------------------------------------------------

def extract_patches(x: torch.Tensor, patch_params: Dict, resolution: int, strategy: str='normal') -> torch.Tensor:
    """
    Extracts patches from images and interpolates them to a desired resolution
    Assumes, that scales/offests in patch_params are given for the [0, 1] image range (i.e. not [-1, 1])
    """
    _, _, h, w = x.shape
    assert h == w, "Can only work on square images (for now)"
    if strategy == 'paste':
        coords = compute_patch_coords(patch_params, resolution, align_corners=False) # [batch_size, resolution, resolution, 2]
        patches = F.grid_sample(x, coords, mode='nearest', align_corners=True) # [batch_size, c, resolution, resolution]
        out = paste_patches(downupsample(x, resolution), patches, patch_params) # [batch_size, c, h, w]
    elif strategy == 'concat':
        coords = compute_patch_coords(patch_params, resolution) # [batch_size, resolution, resolution, 2]
        patches = F.grid_sample(x, coords, mode='bilinear', align_corners=True) # [batch_size, c, resolution, resolution]
        img_lowres = F.interpolate(x, size=(resolution, resolution), mode='bilinear', align_corners=True) # [batch_size, c, resolution, resolution]
        out = torch.cat([img_lowres, patches], dim=1) # [batch_size, 2 * c, resolution, resolution]
    elif strategy == 'normal':
        coords = compute_patch_coords(patch_params, resolution) # [batch_size, resolution, resolution, 2]
        out = F.grid_sample(x, coords, mode='bilinear', align_corners=True) # [batch_size, c, resolution, resolution]
    else:
        raise NotImplementedError(f'Uknown patch-wise training strategy: {strategy}')
    return out

#----------------------------------------------------------------------------

def paste_patches(x: torch.Tensor, patches: torch.Tensor, patch_params: Dict) -> torch.Tensor:
    """
    Pastes patches onto an image. Assumes that all patch scales are the same.
    """
    batch_size, c, h, w = x.shape
    _, _, patch_h, patch_w = patches.shape
    misc.assert_shape(patches, [batch_size, c, None, None])
    assert patch_h == patch_w, f"Unsupported patch format: {patch_h} != {patch_w}"
    assert torch.all(patch_params['scales'] == patch_h / h), f"Bad scales: {patch_params}"

    h_start = (patch_params['offsets'][:, 0] * h).long() # [batch_size]
    w_start = (patch_params['offsets'][:, 1] * w).long() # [batch_size]
    h_idx = torch.arange(patch_h, device=x.device).unsqueeze(1).repeat(1, patch_w).unsqueeze(0) + h_start.unsqueeze(1).unsqueeze(2) # [batch_size, patch_h, patch_w]
    w_idx = torch.arange(patch_w, device=x.device).unsqueeze(0).repeat(patch_h, 1).unsqueeze(0) + w_start.unsqueeze(1).unsqueeze(2) # [batch_size, patch_h, patch_w]
    batch_idx = torch.arange(batch_size).unsqueeze(1).unsqueeze(2).repeat(1, patch_h, patch_w) # [batch_size, patch_h, patch_w]
    x[batch_idx, :, h_idx, w_idx] = patches.permute(0, 2, 3, 1) # [batch_size, patch_h, patch_w, 3]
    masks = torch.zeros_like(x[:, [0]]) # [batch_size, 1, h, w]
    masks[batch_idx, :, h_idx, w_idx] = torch.ones_like(patches[:, [0]].permute(0, 2, 3, 1)) # [batch_size, patch_h, patch_w, 1]
    x = torch.cat([x, masks], dim=1) # [batch_size, c + 1, h, w]

    return x

#----------------------------------------------------------------------------

def discretize_patch_params(patch_params: Dict, patch_size: int) -> Dict:
    patch_params['offsets'] = (patch_params['offsets'] * patch_size).floor() / patch_size
    patch_params['scales'] = (patch_params['scales'] * patch_size).floor() / patch_size

    return patch_params

#----------------------------------------------------------------------------

def compute_patch_coords(patch_params: Dict, resolution: int, align_corners: bool=True, for_grid_sample: bool=True) -> torch.Tensor:
    """
    Given patch parameters and the target resolution, it extracts
    """
    patch_scales, patch_offsets = patch_params['scales'], patch_params['offsets'] # [batch_size, 2], [batch_size, 2]
    batch_size, _ = patch_scales.shape
    coords = generate_coords(batch_size=batch_size, img_size=resolution, device=patch_scales.device, align_corners=align_corners) # [batch_size, out_h, out_w, 2]

    # First, shift the coordinates from the [-1, 1] range into [0, 2]
    # Then, multiply by the patch scales
    # After that, shift back to [-1, 1]
    # Finally, apply the offset converted from [0, 1] to [0, 2]
    coords = (coords + 1.0) * patch_scales.view(batch_size, 1, 1, 2) - 1.0 + patch_offsets.view(batch_size, 1, 1, 2) * 2.0 # [batch_size, out_h, out_w, 2]

    if for_grid_sample:
        # Transforming the coords to the layout of `F.grid_sample`
        coords[:, :, :, 1] = -coords[:, :, :, 1] # [batch_size, out_h, out_w]

    return coords

#----------------------------------------------------------------------------

def sample_patch_params(batch_size: int, patch_cfg: Dict, device: str='cpu', structs: torch.Tensor=None) -> Dict:
    """
    Samples patch parameters: {scales: [x, y], offsets: [x, y]}
    It assumes to follow image memory layout
    """
    if patch_cfg['distribution'] == 'uniform':
        return sample_patch_params_uniform(
            batch_size=batch_size,
            min_scale=patch_cfg['min_scale'],
            max_scale=patch_cfg['max_scale'],
            group_size=patch_cfg['mbstd_group_size'],
            device=device,
        )
    elif patch_cfg['distribution'] == 'discrete_uniform':
        return sample_patch_params_uniform(
            batch_size=batch_size,
            min_scale=patch_cfg['min_scale'],
            max_scale=patch_cfg['max_scale'],
            discrete_support=patch_cfg['discrete_support'],
            group_size=patch_cfg['mbstd_group_size'],
            device=device,
        )
    elif patch_cfg['distribution'] == 'beta':
        return sample_patch_params_beta(
            batch_size=batch_size,
            min_scale=patch_cfg['min_scale'],
            max_scale=patch_cfg['max_scale'],
            alpha=patch_cfg['alpha'],
            beta=patch_cfg['beta'],
            group_size=patch_cfg['mbstd_group_size'],
            device=device,
        )
    elif patch_cfg['distribution'] == 'categorical':
        return sample_patch_params_categorical(
            batch_size=batch_size,
            support=patch_cfg['support'],
            probs=patch_cfg['probs'],
            group_size=patch_cfg['mbstd_group_size'],
            device=device,
        )
    else:
        raise NotImplementedError(f'Unkown patch sampling distrubtion: {patch_cfg["distribution"]}')

#----------------------------------------------------------------------------

def sample_patch_params_structured(min_scale: float, max_scale: float, structs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates patch scales and patch offsets
    Returns patch offsets for [0, 1] range (i.e. image_size = 1 unit)

    Params:
    - min_scale --- minimum patch scale
    - min_scale --- maximum patch scale
    - structs: [batch_size, h, w] --- list of structures
    """
    patch_scales = sample_patch_params_uniform(
        batch_size=structs.shape[0],
        min_scale=min_scale,
        max_scale=max_scale,
        device=structs.device,
    )['scales'] # [batch_size, 2]
    patch_offsets = sample_offsets_from_structs(patch_scales, structs) # [batch_size, 2]

    return {'scales': patch_scales, 'offsets': patch_offsets}

#----------------------------------------------------------------------------

def sample_offsets_from_structs(patch_scales: torch.Tensor, structs: torch.Tensor) -> torch.Tensor:
    """
    - patch_scales: [batch_size, 2] --- list of patch scales
    - structs: [batch_size, h, w] --- list of structures
    """
    batch_size, h, w = structs.shape
    assert len(patch_scales) == len(structs), f"Wrong sizes: {patch_scales.shape} or {structs.shape}"
    assert h == w, f"Cant handle non-square structures: h={h} != w={w}"
    assert h > 16, f"Structure is too small."
    patch_sizes = (patch_scales[:, 0] * h).ceil().long() # [batch_size]

    # TODO: we are doing this in a for-loop since it's much easier this way
    # but it might be the case that doing this in parallel would be faster
    # (though not necessarily since we would use the largest patch size to batchify)
    heatmaps = [compute_sampling_heatmap(p, s) for p, s in zip(patch_sizes, structs)] # (batch_size, [h, w])
    heatmaps = torch.stack(heatmaps, dim=0) # [batch_size, h, w]
    misc.assert_shape(heatmaps, list(structs.shape))

    # Simulating categorical distribution
    probs = heatmaps.view(batch_size, h * w) # [batch_size, h * w]
    cumulatives = probs.cumsum(dim=1) # [batch_size, h * w]
    pivots = torch.rand(batch_size, 1, device=cumulatives.device) # [batch_size, h * w]
    coords = (cumulatives > pivots).long().argmax(dim=1) # [batch_size]
    coords_i = (coords / w).floor().long() # [batch_size]
    coords_j = coords % w # [batch_size]
    # assert torch.all(coords_i + patch_sizes - 1 < h), f"Bad coords_i: {coords_i}"
    # assert torch.all(coords_j + patch_sizes - 1 < w), f"Bad coords_j: {coords_j}"
    offsets = torch.stack([coords_i / h, coords_j / w], dim=1) # [batch_size, 2]
    # assert torch.all(offsets + patch_scales < 1.0), f"Bad offsets: {offsets} for scales {offsets}"

    return offsets

#----------------------------------------------------------------------------

def compute_sampling_heatmap(patch_size: int, struct: torch.Tensor) -> torch.Tensor:
    """
    Computes the heatmap in a *non-batchwise* fashion (for simplicity)
    """
    weight = torch.ones(1, 1, patch_size, patch_size, device=struct.device) # [1, 1, patch_size, patch_size]
    weight = weight / (patch_size * patch_size) # [1, 1, patch_size, patch_size]
    heatmap = F.conv2d(struct.unsqueeze(0).unsqueeze(1), weight=weight, padding=0) # [1, 1, h - patch_size + 1, w - patch_size + 1]
    heatmap = F.pad(heatmap, pad=(0, patch_size - 1, 0, patch_size - 1)) # [1, 1, h, w]
    heatmap = heatmap.squeeze(1).squeeze(0) # [h, w]

    return heatmap / heatmap.sum()

#----------------------------------------------------------------------------

def sample_patch_params_uniform(batch_size: int, min_scale: float, max_scale: float, discrete_support: List[float]=None, group_size: int=1, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates patch scales and patch offsets
    Returns patch offsets for [0, 1] range (i.e. image_size = 1 unit)
    """
    assert max_scale <= 1.0, f"Too large max_scale: {max_scale}"
    assert min_scale <= max_scale, f"Incorrect params: min_scale = {min_scale}, max_scale = {max_scale}"

    num_groups = batch_size // group_size

    if discrete_support is None:
        patch_scales_x = np.random.rand(num_groups) * (max_scale - min_scale) + min_scale # [num_groups]
    else:
        # Sampling from the discrete distribution
        curr_support = [s for s in discrete_support if min_scale <= s <= max_scale]
        patch_scales_x = np.random.choice(curr_support, size=num_groups, replace=True).astype(np.float32) # [num_groups]

    return create_patch_params_from_x_scales(patch_scales_x, group_size, **kwargs)

#----------------------------------------------------------------------------

def sample_patch_params_gaussian(batch_size: int, min_scale: float, max_scale: float, std: float=1.0, device: str='cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates patch scales from uniform distribution and patch offsets from truncated normal
    Returns patch offsets for [0, 1] range (i.e. image_size = 1 unit)
    """
    assert max_scale <= 1.0, f"Too large max_scale: {max_scale}"
    assert min_scale <= max_scale, f"Incorrect params: min_scale = {min_scale}, max_scale = {max_scale}"

    if min_scale == max_scale:
        return sample_patch_params_uniform(
            batch_size=batch_size,
            min_scale=min_scale,
            max_scale=max_scale,
            device=device,
        )

    # Sample scales
    patch_scales_x = np.random.rand(batch_size) * (max_scale - min_scale) + min_scale # [batch_size]
    patch_scales = np.stack([patch_scales_x, patch_scales_x], axis=1) # [batch_size, 2]

    # Sample an offset from [0, 1 - patch_size]
    mean = 0.5 - patch_scales * 0.5 # [batch_size, 2]
    clip_left = np.zeros(patch_scales.shape) # [batch_size, 2]
    clip_right = 1.0 - patch_scales # [batch_size, 2]
    alpha = (clip_left - mean) / std
    beta = (clip_right - mean) / std
    patch_offsets = truncnorm.rvs(a=alpha, b=beta, scale=std, loc=mean, size=(batch_size, 2)) # [batch_size, 2]

    patch_scales = torch.from_numpy(patch_scales.astype(np.float32)).to(device) # [batch_size, 2]
    patch_offsets = torch.from_numpy(patch_offsets.astype(np.float32)).to(device) # [batch_size, 2]

    return {'scales': patch_scales, 'offsets': patch_offsets}

#----------------------------------------------------------------------------

def sample_patch_params_beta(batch_size: int, min_scale: float, max_scale: float, alpha: float, beta: float, group_size: int=1, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates patch scales and patch offsets
    Returns patch offsets for [0, 1] range (i.e. image_size = 1 unit)
    """
    assert max_scale <= 1.0, f"Too large max_scale: {max_scale}"
    assert min_scale <= max_scale, f"Incorrect params: min_scale = {min_scale}, max_scale = {max_scale}"
    num_groups = batch_size // group_size
    patch_scales_x = np.random.beta(a=alpha, b=beta, size=num_groups) * (max_scale - min_scale) + min_scale
    return create_patch_params_from_x_scales(patch_scales_x, group_size, **kwargs)

#----------------------------------------------------------------------------

def sample_patch_params_categorical(batch_size: int, support: List[float], probs: List[float], group_size: int=1, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates patch scales and patch offsets
    Returns patch offsets for [0, 1] range (i.e. image_size = 1 unit)
    """
    assert len(support) == len(probs), f"Wrong sizes: {support}, {probs}"
    assert sum(probs) == 1.0, f"Wrong probs normalization: {probs}"
    num_groups = batch_size // group_size
    patch_scales_x = np.random.choice(a=support, p=probs, size=num_groups) # [num_groups]
    return create_patch_params_from_x_scales(patch_scales_x, group_size, **kwargs)

#----------------------------------------------------------------------------

def create_patch_params_from_x_scales(patch_scales_x: np.ndarray, group_size: int=1, device: str='cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Since we assume that patches are square and we sample assets uniformly,
    we can share a lot of code parts.
    """
    patch_scales_x = torch.from_numpy(patch_scales_x).float().to(device)
    patch_scales = torch.stack([patch_scales_x, patch_scales_x], dim=1) # [num_groups, 2]

    # Sample an offset from [0, 1 - patch_size]
    patch_offsets = torch.rand(patch_scales.shape, device=device) * (1.0 - patch_scales) # [num_groups, 2]

    # Replicate the groups (needed for the MiniBatchStdLayer)
    patch_scales = patch_scales.repeat_interleave(group_size, dim=0) # [batch_size, 2]
    patch_offsets = patch_offsets.repeat_interleave(group_size, dim=0) # [batch_size, 2]

    return {'scales': patch_scales, 'offsets': patch_offsets}

#----------------------------------------------------------------------------

def generate_coords(batch_size: int, img_size: int, device='cpu', align_corners: bool=False) -> torch.Tensor:
    """
    Generates the coordinates in [-1, 1] range for a square image
    if size (img_size x img_size) in such a way that
    - upper left corner: coords[idx, 0, 0] = (-1, 1)
    - lower right corner: coords[idx, -1, -1] = (1, -1)
    In this way, the `y` axis is flipped to follow image memory layout
    """
    if align_corners:
        row = torch.linspace(-1, 1, img_size, device=device).float() # [img_size]
    else:
        row = (torch.arange(0, img_size, device=device).float() / img_size) * 2 - 1 # [img_size]
    x_coords = row.view(1, -1).repeat(img_size, 1) # [img_size, img_size]
    y_coords = -x_coords.t() # [img_size, img_size]

    coords = torch.stack([x_coords, y_coords], dim=2) # [img_size, img_size, 2]
    coords = coords.view(-1, 2) # [img_size ** 2, 2]
    coords = coords.t().view(1, 2, img_size, img_size).repeat(batch_size, 1, 1, 1) # [batch_size, 2, img_size, img_size]
    coords = coords.permute(0, 2, 3, 1) # [batch_size, 2, img_size, img_size]

    return coords

#----------------------------------------------------------------------------

def run_batchwise(fn: Callable, data: Dict[str, torch.Tensor], batch_size: int, dim: int=0, **kwargs) -> Any:
    """
    Runs a function in a batchwise fashion along the `dim` dimension to prevent OOM
    Params:
        - fn: the function to run
        - data: a dict of tensors which should be split batchwise
    """
    # Filter out None data types
    keys, values = zip(*data.items())
    assert batch_size >= 1, f"Wrong batch_size: {batch_size}"
    assert len(set([v.shape[dim] for v in values])) == 1, \
        f"Tensors must be of the same size along dimension {dim}. Got {[v.shape[dim] for v in values]}"

    # Early exit
    if values[0].shape[dim] <= batch_size:
        return fn(**data, **kwargs)

    results = []
    num_runs = (values[0].shape[dim] + batch_size - 1) // batch_size

    for i in range(num_runs):
        assert dim == 1, f"Sorry, works only for dim=1, while provided dim={dim}"
        curr_data = {k: d[:, i * batch_size: (i+1) * batch_size] for k, d in data.items()}
        results.append(fn(**curr_data, **kwargs))

    if isinstance(results[0], torch.Tensor):
        return torch.cat(results, dim=dim)
    elif isinstance(results[0], list) or isinstance(results[0], tuple):
        return [torch.cat([r[i] for r in results], dim=dim) for i in range(len(results[0]))]
    elif isinstance(results[0], dict):
        return {k: torch.cat([r[k] for r in results], dim=dim) for k in results[0].keys()}
    else:
        raise NotImplementedError(f"Cannot handle {type(results[0])} result types.")

#----------------------------------------------------------------------------

def downupsample(x: torch.Tensor, lowres_size: int):
    """Downsamples and then upsamples the image"""
    _, _, h, w = x.shape
    x = F.interpolate(x, size=lowres_size, mode='bilinear', align_corners=False) # [batch_size, c, lowres_h, lowres_w]
    x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False) # [batch_size, c, h, w]

    return x

#----------------------------------------------------------------------------
