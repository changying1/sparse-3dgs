"""Small compatibility helpers for optional CUDA extension APIs."""

from contextlib import contextmanager
import inspect

import torch


def unwrap_first_result(result):
    """Accept extension functions returning either a tensor or a tuple/list."""
    if isinstance(result, (tuple, list)):
        return result[0]
    return result


def accepts_keyword(callable_obj, keyword):
    """Return whether a callable exposes a named keyword argument."""
    fields = getattr(callable_obj, "_fields", ())
    if keyword in fields:
        return True

    try:
        parameters = inspect.signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return False

    return any(
        parameter.name == keyword
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def make_rasterization_settings(settings_type, confidence, **kwargs):
    """Construct settings with ``confidence`` only when the backend supports it."""
    if accepts_keyword(settings_type, "confidence"):
        kwargs["confidence"] = confidence
    return settings_type(**kwargs)


def should_sample_pseudo(iteration, interval, start, end):
    """Mirror TWINGS' original pseudo-view sampling condition."""
    return iteration % interval == 0 and iteration > start and iteration < end


def offload_cuda_model(model, empty_cache=torch.cuda.empty_cache):
    """Move a model to CPU and release now-unused CUDA allocator blocks."""
    model.cpu()
    empty_cache()


@contextmanager
def cuda_model_session(model, empty_cache=torch.cuda.empty_cache):
    """Temporarily place a model on CUDA, always offloading it afterwards."""
    model.cuda()
    try:
        yield model
    finally:
        offload_cuda_model(model, empty_cache)


def detached_depth_inference(model, transform, image):
    """Run DepthPro as a target generator, outside the autograd graph."""
    with torch.no_grad():
        prediction = model.infer(transform(image))
        return prediction["depth"].detach()


def detached_tensor_to(value, device):
    """Move an existing target without triggering ``torch.tensor(tensor)`` warnings."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device)
    return torch.as_tensor(value, device=device)
