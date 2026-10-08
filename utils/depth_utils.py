import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from contextlib import contextmanager
from utils.runtime_compat import (
    cuda_model_session,
    detached_depth_inference,
    offload_cuda_model,
)
# Ensure ml-depth-pro is on sys.path before importing depth_pro
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "submodules", "ml-depth-pro", "src"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import depth_pro


def _adapt_depth_pro_transform_for_tensor_input(transform):
    """Drop only DepthPro's leading PIL/NumPy-only ToTensor transform."""
    transforms = getattr(transform, "transforms", None)
    if not transforms or transforms[0].__class__.__name__ != "ToTensor":
        return transform
    return type(transform)(transforms[1:])


depth_pro_model, depth_pro_transform = depth_pro.create_model_and_transforms(device=torch.device("cuda"))
depth_pro_transform = _adapt_depth_pro_transform_for_tensor_input(depth_pro_transform)
depth_pro_model.eval()
for param in depth_pro_model.parameters():
    param.requires_grad = False
    
def estimate_depth_pro(tensor, mode='test'):
    # ``mode`` is retained for compatibility with the original TWINGS callers.
    # Both paths have always produced a target: the original pseudo path was
    # detached immediately by torch.tensor(...), so no gradients are required.
    return detached_depth_inference(depth_pro_model, depth_pro_transform, tensor)


def offload_depth_pro():
    offload_cuda_model(depth_pro_model)


@contextmanager
def depth_pro_on_cuda():
    """Keep the shared DepthPro instance on CUDA only for the enclosed inference."""
    with cuda_model_session(depth_pro_model):
        yield

def apply_colormap(depth_map, cmap_name='jet'):
    # Check input type and shape
    if isinstance(depth_map, torch.Tensor):
        if depth_map.dim() > 2:
            depth_map = depth_map.squeeze(0)  # Remove the batch dimension, resulting in (H, W)
        depth_np = depth_map.cpu().numpy()
    elif isinstance(depth_map, np.ndarray):
        if depth_map.ndim > 2:
            depth_map = np.squeeze(depth_map)
        depth_np = depth_map
    else:
        raise TypeError("Input must be either a torch.Tensor or a numpy.ndarray")

    # Apply colormap
    cmap = plt.get_cmap(cmap_name)
    colored_depth = cmap(depth_np)
    
    # Convert back to torch tensor (H, W, 4) -> (3, H, W)
    colored_depth_tensor = torch.from_numpy(colored_depth).permute(2, 0, 1)[:3]
    
    return colored_depth_tensor
