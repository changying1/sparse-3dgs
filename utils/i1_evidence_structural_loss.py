"""Frozen I1 observation-evidence structural loss for TWINGS.

This module intentionally contains only the dependency closure needed by I1.
In particular, it does not include the FSGS MiDaS transforms or any cross-view
diagnostic machinery.
"""

import math

import torch


I1_GATE_MODES = ("stable_only", "stable_evidence")


def resolve_i1_gate_mode(
    iteration,
    enabled=True,
    phase1_start=1600,
    phase2_start=2000,
    end_iter=5000,
):
    """Resolve the frozen two-phase I1 schedule (all bounds are inclusive)."""
    if not enabled:
        return None
    iteration = int(iteration)
    phase1_start = int(phase1_start)
    phase2_start = int(phase2_start)
    end_iter = int(end_iter)
    if not (phase1_start <= phase2_start <= end_iter):
        raise ValueError("I1 schedule must satisfy phase1_start <= phase2_start <= end_iter")
    if iteration < phase1_start or iteration > end_iter:
        return None
    if iteration < phase2_start:
        return "stable_only"
    return "stable_evidence"


def _as_depth_map(depth, detach=False):
    tensor = torch.as_tensor(depth)
    if detach:
        tensor = tensor.detach()
    tensor = tensor.float()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    if tensor.ndim != 2:
        raise ValueError("depth inputs must be 2D maps or single-channel 3D tensors")
    return tensor


def adapt_depthpro_reference(reference_depth, rendered_depth):
    """Adapt cached TWINGS DepthPro depth without changing its direction."""
    rendered = _as_depth_map(rendered_depth)
    reference = _as_depth_map(reference_depth, detach=True).to(
        device=rendered.device, dtype=rendered.dtype
    )
    if reference.shape != rendered.shape:
        raise ValueError("rendered_depth and reference_depth must have matching shapes")
    return reference.detach()


def compute_stable_mask_from_gt_rgb(gt_rgb, stable_quantile=0.80):
    """Return the low-local-gradient region used by the original frozen I1."""
    if not 0.0 <= float(stable_quantile) <= 1.0:
        raise ValueError("stable_quantile must satisfy 0 <= q <= 1")
    with torch.no_grad():
        rgb = torch.as_tensor(gt_rgb).detach().float()
        if rgb.ndim != 3:
            raise ValueError("gt_rgb must be a 3D tensor")
        if rgb.shape[0] not in (1, 3) and rgb.shape[-1] in (1, 3):
            rgb = rgb.permute(2, 0, 1)
        if rgb.shape[0] == 1:
            gray = rgb[0]
        elif rgb.shape[0] == 3:
            weights = torch.tensor(
                [0.299, 0.587, 0.114], device=rgb.device, dtype=rgb.dtype
            )
            gray = (rgb * weights[:, None, None]).sum(dim=0)
        else:
            raise ValueError("gt_rgb must have 1 or 3 channels")

        finite = torch.isfinite(gray)
        safe_gray = torch.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0)
        dx = torch.zeros_like(safe_gray)
        dy = torch.zeros_like(safe_gray)
        dx[:, :-1] = safe_gray[:, 1:] - safe_gray[:, :-1]
        dy[:-1, :] = safe_gray[1:, :] - safe_gray[:-1, :]
        magnitude = torch.sqrt(dx * dx + dy * dy)
        finite = finite & torch.isfinite(magnitude)
        if not finite.any():
            return torch.zeros_like(finite), None, magnitude.masked_fill(~finite, float("nan"))

        threshold = torch.quantile(magnitude[finite].float(), float(stable_quantile))
        stable = finite & (magnitude < threshold.to(magnitude))
        return stable.detach(), float(threshold.item()), magnitude.masked_fill(
            ~finite, float("nan")
        ).detach()


def compute_depthpro_evidence_maps(rendered_depth, reference_depth, valid_mask=None, eps=1e-8):
    """Compute detached observation evidence using direct DepthPro depth."""
    rendered = _as_depth_map(rendered_depth)
    reference = adapt_depthpro_reference(reference_depth, rendered)
    valid = torch.isfinite(rendered) & torch.isfinite(reference)
    if valid_mask is not None:
        supplied = _as_depth_map(valid_mask, detach=True).to(rendered.device).bool()
        if supplied.shape != rendered.shape:
            raise ValueError("valid_mask must match depth map shape")
        valid = valid & supplied
    valid = valid.detach()

    z_render = torch.full_like(rendered, float("nan"))
    z_reference = torch.full_like(rendered, float("nan"))
    evidence = torch.full_like(rendered, float("nan"))
    normalized_valid = False

    if int(valid.sum().item()) >= 2:
        rendered_values = rendered[valid]
        reference_values = reference[valid]
        rendered_centered = rendered_values - rendered_values.mean()
        reference_centered = reference_values - reference_values.mean()
        rendered_scale = torch.sqrt((rendered_centered.square()).mean())
        reference_scale = torch.sqrt((reference_centered.square()).mean())
        rendered_scale_ok = bool(
            torch.isfinite(rendered_scale.detach()) and rendered_scale.detach().item() > eps
        )
        reference_scale_ok = bool(
            torch.isfinite(reference_scale.detach()) and reference_scale.detach().item() > eps
        )
        if rendered_scale_ok and reference_scale_ok:
            normalized_valid = True
            z_render_values = rendered_centered / rendered_scale.clamp_min(eps)
            z_reference_values = (
                reference_centered / reference_scale.clamp_min(eps)
            ).detach()
            z_render[valid] = z_render_values
            z_reference[valid] = z_reference_values
            evidence[valid] = 1.0 / (
                1.0 + (z_render_values.detach() - z_reference_values).abs()
            )

    return {
        "valid_mask": valid,
        "normalized_valid": normalized_valid,
        "normalization": "centered_rms",
        "reference_mode": "depthpro_direct",
        "selected_reference_depth": reference,
        "z_render": z_render,
        "z_reference": z_reference.detach(),
        "evidence": evidence.detach(),
    }


def _pair_terms(rendered, reference, evidence, stable, valid, dim):
    if dim == 1:
        first = (slice(None), slice(None, -1))
        second = (slice(None), slice(1, None))
    elif dim == 0:
        first = (slice(None, -1), slice(None))
        second = (slice(1, None), slice(None))
    else:
        raise ValueError("dim must be 0 or 1")

    loss = ((rendered[second] - rendered[first]) - (reference[second] - reference[first])).abs()
    pair_evidence = torch.minimum(evidence[first], evidence[second]).detach()
    pair_stable = (stable[first] & stable[second]).detach()
    pair_valid = (
        valid[first]
        & valid[second]
        & torch.isfinite(loss).detach()
        & torch.isfinite(pair_evidence)
    ).detach()
    return loss.reshape(-1), pair_evidence.reshape(-1), pair_stable.reshape(-1), pair_valid.reshape(-1)


def compute_i1_structural_loss(
    rendered_depth,
    reference_depth,
    stable_mask,
    gate_mode,
    valid_mask=None,
    eps=1e-8,
):
    """Compare adjacent normalized depth gradients under the frozen I1 gate."""
    if gate_mode not in I1_GATE_MODES:
        raise ValueError("gate_mode must be stable_only or stable_evidence")
    maps = compute_depthpro_evidence_maps(
        rendered_depth, reference_depth, valid_mask=valid_mask, eps=eps
    )
    rendered = maps["z_render"]
    zero = torch.nan_to_num(_as_depth_map(rendered_depth)).sum() * 0.0
    if not maps["normalized_valid"]:
        return zero, _empty_stats(gate_mode), maps

    reference = maps["z_reference"]
    evidence = maps["evidence"]
    valid = maps["valid_mask"]
    stable = _as_depth_map(stable_mask, detach=True).to(valid.device).bool()
    if stable.shape != valid.shape:
        raise ValueError("stable_mask must match depth map shape")

    horizontal = _pair_terms(rendered, reference, evidence, stable, valid, dim=1)
    vertical = _pair_terms(rendered, reference, evidence, stable, valid, dim=0)
    pair_loss = torch.cat((horizontal[0], vertical[0]))
    pair_evidence = torch.cat((horizontal[1], vertical[1])).detach()
    pair_stable = torch.cat((horizontal[2], vertical[2])).detach()
    pair_valid = torch.cat((horizontal[3], vertical[3])).detach()
    valid_pair_count = int(pair_valid.sum().item())
    if valid_pair_count == 0:
        return zero, _empty_stats(gate_mode), maps

    losses = pair_loss[pair_valid]
    evidences = pair_evidence[pair_valid]
    stable_pairs = pair_stable[pair_valid]
    if gate_mode == "stable_only":
        weights = stable_pairs.to(losses.dtype)
    else:
        weights = stable_pairs.to(losses.dtype) * evidences
    weights = weights.detach()
    denominator = weights.sum()
    if denominator.detach().item() <= eps:
        loss = zero
    else:
        loss = (weights * losses).sum() / denominator.clamp_min(eps)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    stats = {
        "gate_mode": gate_mode,
        "structural_loss": _safe_float(loss),
        "valid_pair_count": valid_pair_count,
        "active_pair_count": int((weights > 0).sum().item()),
        "mean_pair_evidence": _safe_float(evidences.mean()),
        "reference_mode": "depthpro_direct",
    }
    return loss, stats, maps


def combine_i1_loss(baseline_loss, structural_loss, weight, gate_mode):
    """Keep the disabled path as the exact baseline loss object."""
    if gate_mode is None:
        return baseline_loss
    return baseline_loss + float(weight) * structural_loss


def should_preserve_i1_densification_stats(enabled, gate_mode, iteration, densify_until_iter):
    return bool(enabled) and gate_mode is not None and int(iteration) < int(densify_until_iter)


def format_i1_log(iteration, camera_id, stats, weight):
    return (
        "[I1] "
        f"iter={int(iteration)} camera_id={camera_id} gate_mode={stats['gate_mode']} "
        f"structural_loss={stats['structural_loss']:.6g} weight={float(weight):.6g} "
        f"valid_pairs={stats['valid_pair_count']} active_pairs={stats['active_pair_count']} "
        "reference_mode=depthpro_direct"
    )


def _empty_stats(gate_mode):
    return {
        "gate_mode": gate_mode,
        "structural_loss": 0.0,
        "valid_pair_count": 0,
        "active_pair_count": 0,
        "mean_pair_evidence": None,
        "reference_mode": "depthpro_direct",
    }


def _safe_float(value):
    value = float(value.detach().cpu().item() if torch.is_tensor(value) else value)
    return value if math.isfinite(value) else None
