"""FSGS Observation-Evidence Guided Structural Loss (Innovation 1)."""

import math

import torch

from utils.evidence_support import compute_real_depth_evidence_maps

OE_STRUCTURE_GATE_MODES = ("none", "stable_only", "evidence_only", "stable_evidence", "matched_stable")


def add_oe_structural_arguments(parser):
    parser.add_argument("--enable_oe_structural_loss", action="store_true", default=False)
    parser.add_argument(
        "--oe_preserve_baseline_densification_stats",
        action="store_true",
        default=False,
    )
    parser.add_argument("--oe_structural_weight", type=float, default=0.0)
    parser.add_argument("--oe_structural_start_iter", type=int, default=0)
    parser.add_argument("--oe_structural_end_iter", type=int, default=30_000)
    parser.add_argument("--oe_stable_quantile", type=float, default=0.80)
    parser.add_argument(
        "--oe_structure_gate_mode",
        type=str,
        choices=OE_STRUCTURE_GATE_MODES,
        default="stable_evidence",
    )
    parser.add_argument("--oe_depth_cache_dir", type=str, default="")


def should_apply_oe_structural_loss(enabled, iteration, start_iter, end_iter, weight):
    return (
        bool(enabled)
        and float(weight) > 0.0
        and int(start_iter) <= int(iteration) <= int(end_iter)
    )


def compute_stable_mask_from_gt_rgb(gt_rgb, stable_quantile=0.80):
    with torch.no_grad():
        rgb = torch.as_tensor(gt_rgb).detach().float()
        if rgb.ndim != 3:
            raise ValueError("gt_rgb must be a 3D tensor.")
        if rgb.shape[0] not in (1, 3) and rgb.shape[-1] in (1, 3):
            rgb = rgb.permute(2, 0, 1)
        if rgb.shape[0] == 1:
            gray = rgb[0]
        elif rgb.shape[0] == 3:
            weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device, dtype=rgb.dtype)
            gray = (rgb * weights[:, None, None]).sum(dim=0)
        else:
            raise ValueError("gt_rgb must have 1 or 3 channels.")

        finite = torch.isfinite(gray)
        safe_gray = torch.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0)
        dx = torch.zeros_like(safe_gray)
        dy = torch.zeros_like(safe_gray)
        dx[:, :-1] = safe_gray[:, 1:] - safe_gray[:, :-1]
        dy[:-1, :] = safe_gray[1:, :] - safe_gray[:-1, :]
        magnitude = torch.sqrt(dx * dx + dy * dy)
        finite = finite & torch.isfinite(magnitude)
        if not finite.any():
            empty = torch.zeros_like(finite, dtype=torch.bool)
            return empty, None, magnitude.masked_fill(~finite, float("nan"))

        threshold = torch.quantile(magnitude[finite].float(), float(stable_quantile))
        stable = finite & (magnitude < threshold.to(device=magnitude.device, dtype=magnitude.dtype))
        return stable.detach(), _safe_float(threshold), magnitude.masked_fill(~finite, float("nan")).detach()


def compute_oe_structural_loss(
    rendered_depth,
    reference_depth,
    gt_rgb=None,
    stable_mask=None,
    valid_mask=None,
    gate_mode="stable_evidence",
    stable_quantile=0.80,
    eps=1e-8,
):
    if gate_mode not in OE_STRUCTURE_GATE_MODES:
        raise ValueError(f"gate_mode must be one of {', '.join(OE_STRUCTURE_GATE_MODES)}")

    maps = compute_real_depth_evidence_maps(
        rendered_depth,
        reference_depth,
        valid_mask=valid_mask,
        eps=eps,
        detach_rendered=False,
    )
    rendered = maps["z_render"]
    zero = torch.as_tensor(rendered_depth, device=rendered.device).float().sum() * 0.0
    if not maps["normalized_valid"]:
        stats = _empty_stats(gate_mode)
        stats["reference_transform"] = maps["reference_transform"]
        return zero, stats, maps

    reference = maps["z_reference"].detach()
    evidence = maps["evidence"].detach()
    valid = maps["valid_mask"].detach()

    if stable_mask is None:
        if gt_rgb is None:
            stable = torch.ones_like(valid, dtype=torch.bool)
            stable_threshold = None
        else:
            stable, stable_threshold, _ = compute_stable_mask_from_gt_rgb(gt_rgb, stable_quantile=stable_quantile)
            stable = stable.to(device=valid.device)
    else:
        stable = torch.as_tensor(stable_mask, device=valid.device).detach().bool()
        if stable.ndim == 3 and stable.shape[0] == 1:
            stable = stable[0]
        if stable.ndim == 3 and stable.shape[-1] == 1:
            stable = stable[..., 0]
        stable_threshold = None
    if stable.shape != valid.shape:
        raise ValueError("stable_mask must match depth map shape.")
    stable = stable.detach()

    horizontal = _pair_terms(rendered, reference, evidence, stable, valid, dim=1)
    vertical = _pair_terms(rendered, reference, evidence, stable, valid, dim=0)
    pair_loss = torch.cat((horizontal["loss"], vertical["loss"]))
    pair_evidence = torch.cat((horizontal["evidence"], vertical["evidence"]))
    pair_stable = torch.cat((horizontal["stable"], vertical["stable"]))
    pair_valid = torch.cat((horizontal["valid"], vertical["valid"]))

    valid_pair_count = int(pair_valid.sum().item())
    if valid_pair_count == 0:
        stats = _empty_stats(gate_mode)
        stats["stable_pixel_fraction"] = _fraction(stable & valid, valid)
        stats["stable_threshold"] = stable_threshold
        stats["reference_transform"] = maps["reference_transform"]
        return zero, stats, maps

    valid_loss = pair_loss[pair_valid]
    valid_evidence = pair_evidence[pair_valid].detach()
    valid_stable = pair_stable[pair_valid].detach()
    evidence_diagnostic_stats = _evidence_distribution_stats(
        valid_evidence,
        valid_stable if gate_mode in ("stable_only", "stable_evidence", "matched_stable") else None,
    )
    raw_loss = valid_loss.mean()
    stable_only_loss, stable_only_weights = _weighted_loss(
        valid_loss,
        _gate_weights(valid_evidence, valid_stable, "stable_only").detach(),
        zero,
        eps,
    )
    stable_evidence_loss, _ = _weighted_loss(
        valid_loss,
        _gate_weights(valid_evidence, valid_stable, "stable_evidence").detach(),
        zero,
        eps,
    )
    if gate_mode == "matched_stable":
        alpha = torch.nan_to_num(
            (stable_evidence_loss / (stable_only_loss + eps)).detach(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        loss = alpha * stable_only_loss
        weights = stable_only_weights
        matched_gated_loss = loss
    else:
        alpha = None
        weights = _gate_weights(valid_evidence, valid_stable, gate_mode).detach()
        loss, weights = _weighted_loss(valid_loss, weights, zero, eps)
        matched_gated_loss = None

    stats = {
        "gate_mode": gate_mode,
        "raw_struct_loss": _safe_float(raw_loss.detach()),
        "gated_struct_loss": _safe_float(loss.detach()),
        "matched_strength_alpha": _safe_float(alpha),
        "stable_only_gated_loss": _safe_float(stable_only_loss.detach()),
        "stable_evidence_gated_loss": _safe_float(stable_evidence_loss.detach()),
        "matched_gated_loss": _safe_float(matched_gated_loss),
        "active_pair_fraction": _safe_float((weights > 0).float().mean()),
        "mean_pair_evidence": _safe_float(valid_evidence.mean()),
        **evidence_diagnostic_stats,
        "stable_pixel_fraction": _fraction(stable & valid, valid),
        "valid_pair_count": valid_pair_count,
        "active_pair_count": int((weights > 0).sum().item()),
        "stable_threshold": stable_threshold,
        "reference_transform": maps["reference_transform"],
    }
    return loss, stats, maps


def _weighted_loss(valid_loss, weights, zero, eps):
    denominator = weights.sum()
    if float(denominator.detach().item()) <= eps:
        return zero, weights
    loss = (weights * valid_loss).sum() / denominator.clamp_min(eps)
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    return loss, weights


def _pair_terms(rendered, reference, evidence, stable, valid, dim):
    if dim == 1:
        left = (slice(None), slice(None, -1))
        right = (slice(None), slice(1, None))
    elif dim == 0:
        left = (slice(None, -1), slice(None))
        right = (slice(1, None), slice(None))
    else:
        raise ValueError("dim must be 0 or 1.")

    grad_rendered = rendered[right] - rendered[left]
    grad_reference = reference[right] - reference[left]
    loss_map = (grad_rendered - grad_reference).abs()
    evidence_map = torch.minimum(evidence[left], evidence[right]).detach()
    stable_map = (stable[left] & stable[right]).detach()
    valid_map = (
        valid[left]
        & valid[right]
        & torch.isfinite(loss_map).detach()
        & torch.isfinite(evidence_map)
    ).detach()
    return {
        "loss": loss_map.reshape(-1),
        "evidence": evidence_map.reshape(-1),
        "stable": stable_map.reshape(-1),
        "valid": valid_map.reshape(-1),
    }


def _gate_weights(evidence, stable, gate_mode):
    if gate_mode == "none":
        return torch.ones_like(evidence)
    if gate_mode == "stable_only":
        return stable.to(dtype=evidence.dtype)
    if gate_mode == "evidence_only":
        return evidence
    if gate_mode == "stable_evidence":
        return stable.to(dtype=evidence.dtype) * evidence
    raise ValueError(f"Unsupported gate_mode: {gate_mode}")


def _evidence_distribution_stats(evidence, population_mask=None):
    detached = evidence.detach()
    if population_mask is not None:
        detached = detached[population_mask.detach().bool()]
    detached = detached[torch.isfinite(detached)]
    if detached.numel() == 0:
        return {
            "pair_evidence_std": None,
            "pair_evidence_p10": None,
            "pair_evidence_p50": None,
            "pair_evidence_p90": None,
            "pair_evidence_spread": None,
        }

    values = detached.float()
    p10 = torch.quantile(values, 0.10)
    p50 = torch.quantile(values, 0.50)
    p90 = torch.quantile(values, 0.90)
    return {
        "pair_evidence_std": _safe_float(values.std(unbiased=False)),
        "pair_evidence_p10": _safe_float(p10),
        "pair_evidence_p50": _safe_float(p50),
        "pair_evidence_p90": _safe_float(p90),
        "pair_evidence_spread": _safe_float(p90 - p10),
    }


def _empty_stats(gate_mode):
    return {
        "gate_mode": gate_mode,
        "raw_struct_loss": 0.0,
        "gated_struct_loss": 0.0,
        "matched_strength_alpha": None,
        "stable_only_gated_loss": 0.0,
        "stable_evidence_gated_loss": 0.0,
        "matched_gated_loss": 0.0 if gate_mode == "matched_stable" else None,
        "active_pair_fraction": 0.0,
        "mean_pair_evidence": None,
        "pair_evidence_std": None,
        "pair_evidence_p10": None,
        "pair_evidence_p50": None,
        "pair_evidence_p90": None,
        "pair_evidence_spread": None,
        "stable_pixel_fraction": 0.0,
        "valid_pair_count": 0,
        "active_pair_count": 0,
        "stable_threshold": None,
        "reference_transform": None,
    }


def _fraction(numerator_mask, denominator_mask):
    denominator = int(denominator_mask.sum().item())
    if denominator == 0:
        return 0.0
    return int(numerator_mask.sum().item()) / denominator


def _safe_float(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("Only scalar tensors can be converted to float.")
        value = value.detach().cpu().item()
    value = float(value)
    return value if math.isfinite(value) else None
