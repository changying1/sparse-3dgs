"""Minimal FSGS observation-evidence support used by Innovation 1.

The formulas in this module are kept aligned with FSGS's
``utils/evidence_support.py``.  Diagnostic-only I2 helpers are intentionally
not imported into this baseline migration.
"""

import math

import torch


def _as_depth_map(depth, detach=True):
    tensor = torch.as_tensor(depth)
    if detach:
        tensor = tensor.detach()
    tensor = tensor.float()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    if tensor.ndim != 2:
        raise ValueError("depth inputs must be 2D maps or single-channel 3D tensors.")
    return tensor


def _safe_float(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("Only scalar tensors can be converted to float.")
        value = value.detach().cpu().item()
    value = float(value)
    return value if math.isfinite(value) else None


def _pearson_corr(a, b, eps=1e-8):
    valid = torch.isfinite(a) & torch.isfinite(b)
    if valid.sum().item() < 2:
        return None
    x = a[valid].float()
    y = b[valid].float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.norm(x) * torch.linalg.norm(y)
    if not torch.isfinite(denom) or float(denom.item()) <= eps:
        return None
    return _safe_float((x * y).sum() / denom.clamp_min(eps))


def _reference_candidates(reference_depth):
    return {
        "negative_midas": -reference_depth,
        "inverse_midas_shift_200": 1.0 / (reference_depth + 200.0),
    }


def select_fsgs_reference_depth(rendered_depth, reference_depth, valid_mask=None, eps=1e-8):
    rendered = _as_depth_map(rendered_depth)
    reference = _as_depth_map(reference_depth).to(device=rendered.device, dtype=rendered.dtype)
    if rendered.shape != reference.shape:
        raise ValueError("rendered_depth and reference_depth must have matching shapes.")

    valid = torch.isfinite(rendered) & torch.isfinite(reference)
    if valid_mask is not None:
        mask = torch.as_tensor(valid_mask, device=rendered.device).detach().bool()
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.shape != rendered.shape:
            raise ValueError("valid_mask must match depth map shape.")
        valid = valid & mask

    candidates = _reference_candidates(reference)
    best_name = None
    best_reference = None
    best_corr = None
    candidate_corr = {}
    flat_rendered = rendered.reshape(-1)
    flat_valid = valid.reshape(-1)
    for name, candidate in candidates.items():
        corr = _pearson_corr(flat_rendered[flat_valid], candidate.reshape(-1)[flat_valid], eps=eps)
        candidate_corr[name] = corr
        if corr is not None and (best_corr is None or corr > best_corr):
            best_name = name
            best_reference = candidate
            best_corr = corr

    if best_reference is None:
        best_name = "negative_midas"
        best_reference = candidates[best_name]

    return best_reference, best_name, best_corr, candidate_corr, valid


def _centered_rms_normalize(values, eps=1e-8):
    if values.numel() < 2:
        return None, None, None
    mean = values.float().mean()
    centered = values.float() - mean
    scale = torch.sqrt((centered * centered).mean())
    if not torch.isfinite(scale) or float(scale.item()) <= eps:
        return None, _safe_float(mean), _safe_float(scale)
    return centered / scale.clamp_min(eps), _safe_float(mean), _safe_float(scale)


def compute_real_depth_evidence_maps(
    rendered_depth,
    reference_depth,
    valid_mask=None,
    eps=1e-8,
    detach_rendered=False,
):
    rendered = _as_depth_map(rendered_depth, detach=detach_rendered)
    reference = _as_depth_map(reference_depth).to(device=rendered.device, dtype=rendered.dtype)
    selected_reference, selected_transform, selected_corr, candidate_corr, valid = select_fsgs_reference_depth(
        rendered.detach(),
        reference,
        valid_mask=valid_mask,
        eps=eps,
    )
    selected_reference = selected_reference.detach().to(device=rendered.device, dtype=rendered.dtype)
    valid = valid.to(device=rendered.device).detach() & torch.isfinite(rendered) & torch.isfinite(selected_reference)

    z_render = torch.full_like(rendered, float("nan"))
    z_reference = torch.full_like(rendered, float("nan"))
    normalized_signed_residual = torch.full_like(rendered, float("nan"))
    normalized_abs_residual = torch.full_like(rendered, float("nan"))
    evidence = torch.full_like(rendered, float("nan"))
    rendered_mean = None
    rendered_scale = None
    reference_mean = None
    reference_scale = None
    normalized_valid = False

    if int(valid.sum().item()) >= 2:
        rendered_values = rendered[valid]
        reference_values = selected_reference[valid]

        rendered_mean_tensor = rendered_values.float().mean()
        rendered_centered = rendered_values.float() - rendered_mean_tensor
        rendered_scale_tensor = torch.sqrt((rendered_centered * rendered_centered).mean())
        reference_mean_tensor = reference_values.float().mean()
        reference_centered = reference_values.float() - reference_mean_tensor
        reference_scale_tensor = torch.sqrt((reference_centered * reference_centered).mean())

        rendered_mean = _safe_float(rendered_mean_tensor.detach())
        rendered_scale = _safe_float(rendered_scale_tensor.detach())
        reference_mean = _safe_float(reference_mean_tensor.detach())
        reference_scale = _safe_float(reference_scale_tensor.detach())
        rendered_scale_ok = (
            torch.isfinite(rendered_scale_tensor.detach())
            and float(rendered_scale_tensor.detach().item()) > eps
        )
        reference_scale_ok = (
            torch.isfinite(reference_scale_tensor.detach())
            and float(reference_scale_tensor.detach().item()) > eps
        )

        if rendered_scale_ok and reference_scale_ok:
            normalized_valid = True
            z_render_values = rendered_centered / rendered_scale_tensor.clamp_min(eps)
            z_reference_values = (reference_centered / reference_scale_tensor.clamp_min(eps)).detach()
            residual = z_render_values - z_reference_values
            abs_residual = residual.abs()
            z_render[valid] = z_render_values.to(dtype=rendered.dtype)
            z_reference[valid] = z_reference_values.to(dtype=rendered.dtype)
            normalized_signed_residual[valid] = residual.to(dtype=rendered.dtype)
            normalized_abs_residual[valid] = abs_residual.to(dtype=rendered.dtype)
            evidence[valid] = (1.0 / (1.0 + abs_residual.detach())).to(dtype=rendered.dtype)

    return {
        "valid_mask": valid.detach(),
        "selected_reference_depth": selected_reference,
        "reference_transform": selected_transform,
        "reference_correlation": selected_corr,
        "candidate_correlations": candidate_corr,
        "normalization": "centered_rms",
        "normalized_valid": normalized_valid,
        "rendered_mean": rendered_mean,
        "rendered_centered_rms": rendered_scale,
        "reference_mean": reference_mean,
        "reference_centered_rms": reference_scale,
        "z_render": z_render,
        "z_reference": z_reference.detach(),
        "normalized_signed_residual": normalized_signed_residual,
        "normalized_absolute_residual": normalized_abs_residual.detach(),
        "evidence": evidence.detach(),
    }
