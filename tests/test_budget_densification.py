from argparse import Namespace
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn

from utils.budget_densification import (
    build_official_clone_split_masks,
    compute_gradient_score,
    select_gradient_topk,
)


def test_gradient_score_matches_official_accum_over_denom_and_sanitizes_invalid():
    accum = torch.tensor([[1.0], [4.0], [3.0], [float("nan")], [float("inf")]])
    denom = torch.tensor([[2.0], [0.0], [float("inf")], [1.0], [1.0]])

    score = compute_gradient_score(accum, denom)

    assert score.shape == (5,)
    assert torch.isfinite(score).all()
    assert torch.allclose(score, torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0]))


def test_gradient_topk_selects_highest_positive_indices():
    score = torch.tensor([0.1, 0.9, 0.4, 0.8])

    selected = select_gradient_topk(score, budget=2)

    assert selected.tolist() == [1, 3]


def test_zero_gradient_does_not_consume_budget():
    score = torch.tensor([0.9, 0.3, 0.0, 0.0])

    selected = select_gradient_topk(score, budget=4)

    assert selected.tolist() == [0, 1]


def test_candidate_mask_excludes_highest_gradient():
    score = torch.tensor([0.1, 0.9, 0.4, 0.8])
    candidate = torch.tensor([True, False, True, True])

    selected = select_gradient_topk(score, budget=2, candidate_mask=candidate)

    assert selected.tolist() == [3, 2]


def test_clone_split_masks_are_disjoint_and_follow_official_scale_rule():
    selected = torch.tensor([True, True, False, True])
    scaling = torch.tensor(
        [
            [0.01, 0.02, 0.03],
            [0.20, 0.01, 0.01],
            [0.30, 0.30, 0.30],
            [0.05, 0.05, 0.05],
        ]
    )

    clone_mask, split_mask = build_official_clone_split_masks(
        selected,
        scaling,
        percent_dense=0.1,
        scene_extent=1.0,
    )

    assert not torch.logical_and(clone_mask, split_mask).any()
    assert torch.equal(clone_mask | split_mask, selected)
    assert clone_mask.tolist() == [True, False, False, True]
    assert split_mask.tolist() == [False, True, False, False]


def test_negative_nan_and_inf_gradient_are_not_selected():
    score = torch.tensor([0.5, -1.0, float("nan"), float("inf"), -float("inf"), 0.2])

    selected = select_gradient_topk(score, budget=6)

    assert selected.tolist() == [0, 5]


def _training_args(percent_dense=0.1):
    return Namespace(
        percent_dense=percent_dense,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30000,
        feature_lr=0.0025,
        opacity_lr=0.025,
        scaling_lr=0.005,
        rotation_lr=0.001,
        exposure_lr_init=0.01,
        exposure_lr_final=0.001,
        exposure_lr_delay_steps=0,
        exposure_lr_delay_mult=0.0,
        iterations=30000,
    )


@pytest.fixture
def cuda_gaussians():
    if not torch.cuda.is_available():
        pytest.skip("GaussianModel densification path is CUDA-only in the official implementation.")

    GaussianModel = _load_gaussian_model_class()

    def make_model(scales, gradient_scores=None):
        device = torch.device("cuda")
        n = len(scales)
        model = GaussianModel(sh_degree=0)
        model.spatial_lr_scale = 1.0
        model._xyz = nn.Parameter(torch.arange(n * 3, dtype=torch.float32, device=device).reshape(n, 3) * 0.01)
        model._features_dc = nn.Parameter(torch.zeros((n, 1, 3), dtype=torch.float32, device=device))
        model._features_rest = nn.Parameter(torch.zeros((n, 0, 3), dtype=torch.float32, device=device))
        model._scaling = nn.Parameter(torch.log(torch.tensor(scales, dtype=torch.float32, device=device)))
        model._rotation = nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(n, 1)
        )
        model._opacity = nn.Parameter(torch.zeros((n, 1), dtype=torch.float32, device=device))
        model._exposure = nn.Parameter(torch.eye(3, 4, dtype=torch.float32, device=device)[None])
        model.training_setup(_training_args(percent_dense=0.1))
        model.tmp_radii = torch.zeros((n,), dtype=torch.float32, device=device)
        model.ensure_visibility_history(num_views=3)
        if gradient_scores is None:
            gradient_scores = torch.arange(n, 0, -1, dtype=torch.float32)
        model.xyz_gradient_accum = gradient_scores.reshape(n, 1).to(device)
        model.denom = torch.ones((n, 1), dtype=torch.float32, device=device)
        return model

    return make_model


def _load_gaussian_model_class():
    if "plyfile" not in sys.modules:
        plyfile_stub = types.ModuleType("plyfile")
        plyfile_stub.PlyData = object
        plyfile_stub.PlyElement = object
        sys.modules["plyfile"] = plyfile_stub
    if "simple_knn" not in sys.modules:
        sys.modules["simple_knn"] = types.ModuleType("simple_knn")
    if "simple_knn._C" not in sys.modules:
        simple_knn_c_stub = types.ModuleType("simple_knn._C")
        simple_knn_c_stub.distCUDA2 = lambda *args, **kwargs: None
        sys.modules["simple_knn._C"] = simple_knn_c_stub

    module_name = "_budget_test_gaussian_model"
    if module_name in sys.modules:
        return sys.modules[module_name].GaussianModel
    module_path = Path(__file__).resolve().parents[1] / "scene" / "gaussian_model.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.GaussianModel


def _small_scales(n):
    return [[0.02, 0.02, 0.02] for _ in range(n)]


def test_strict_fixed_budget_when_candidates_are_sufficient(cuda_gaussians):
    model = cuda_gaussians(_small_scales(8))
    before = model.get_xyz.shape[0]

    stats = model.densify_gradient_topk_with_budget(
        budget=4,
        scene_extent=1.0,
        max_gaussians=100,
        return_stats=True,
    )

    assert stats["densification_net_added"] == 4
    assert model.get_xyz.shape[0] - before == 4


def test_max_gaussians_remaining_capacity_caps_budget(cuda_gaussians):
    model = cuda_gaussians(_small_scales(6))
    before = model.get_xyz.shape[0]

    stats = model.densify_gradient_topk_with_budget(
        budget=10,
        scene_extent=1.0,
        max_gaussians=before + 2,
        return_stats=True,
    )

    assert stats["effective_budget"] == 2
    assert stats["densification_net_added"] == 2
    assert model.get_xyz.shape[0] <= before + 2


def test_budget_zero_performs_no_clone_or_split(cuda_gaussians):
    model = cuda_gaussians(_small_scales(5))
    before = model.get_xyz.shape[0]

    stats = model.densify_gradient_topk_with_budget(
        budget=0,
        scene_extent=1.0,
        max_gaussians=100,
        return_stats=True,
    )

    assert stats["densification_net_added"] == 0
    assert model.get_xyz.shape[0] == before


def test_insufficient_gradient_candidates_leave_budget_unused(cuda_gaussians):
    model = cuda_gaussians(_small_scales(6), gradient_scores=torch.tensor([0.9, 0.7, 0.5, 0.0, 0.0, 0.0]))
    before = model.get_xyz.shape[0]

    stats = model.densify_gradient_topk_with_budget(
        budget=10,
        scene_extent=1.0,
        max_gaussians=100,
        return_stats=True,
    )

    assert stats["gradient_selected"] == 3
    assert stats["densification_net_added"] == 3
    assert model.get_xyz.shape[0] - before == 3


def test_structural_state_stays_synchronized_after_clone_and_split(cuda_gaussians):
    scales = [
        [0.02, 0.02, 0.02],
        [0.20, 0.20, 0.20],
        [0.02, 0.02, 0.02],
        [0.20, 0.20, 0.20],
    ]
    model = cuda_gaussians(scales, gradient_scores=torch.tensor([4.0, 3.0, 0.0, 0.0]))

    stats = model.densify_gradient_topk_with_budget(
        budget=2,
        scene_extent=1.0,
        max_gaussians=100,
        return_stats=True,
    )

    count = model.get_xyz.shape[0]
    assert stats["clone_selected"] == 1
    assert stats["split_selected"] == 1
    assert stats["densification_net_added"] == 2
    assert model.visibility_history.shape[0] == count
    assert model.visible_view_count.shape[0] == count
    assert model.birth_iteration.shape[0] == count
    assert model.source_type.shape[0] == count
    assert model.parent_index.shape[0] == count
    assert model.completion_support_count.shape[0] == count
    assert model.completion_age.shape[0] == count
    assert model.is_completion.shape[0] == count


def test_mask_versions_preserve_official_clone_and_split_generation(cuda_gaussians):
    model = cuda_gaussians([[0.02, 0.02, 0.02], [0.20, 0.20, 0.20]])
    original_xyz = model.get_xyz.detach().clone()
    original_scaling = model.get_scaling.detach().clone()
    clone_mask = torch.tensor([True, False], dtype=torch.bool, device=model.get_xyz.device)
    split_mask = torch.tensor([False, True], dtype=torch.bool, device=model.get_xyz.device)

    stats = model.densify_with_budget(
        clone_mask,
        split_mask,
        budget=2,
        max_gaussians=10,
        scene_extent=1.0,
        split_N=2,
        return_stats=True,
    )

    assert stats["densification_net_added"] == 2
    cloned_xyz = model.get_xyz[1].detach()
    assert torch.allclose(cloned_xyz, original_xyz[0])
    split_children_scaling = model.get_scaling[-2:].detach()
    expected_split_scaling = original_scaling[1].repeat(2, 1) / 1.6
    assert torch.allclose(split_children_scaling, expected_split_scaling, atol=1e-6)


def test_split_n_three_uses_net_cost_two(cuda_gaussians):
    scales = [[0.20, 0.20, 0.20] for _ in range(6)]
    gradients = torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

    model = cuda_gaussians(scales, gradient_scores=gradients)
    before = model.get_xyz.shape[0]

    stats = model.densify_gradient_topk_with_budget(
        budget=4,
        scene_extent=1.0,
        max_gaussians=100,
        split_N=3,
        return_stats=True,
    )

    assert stats["split_selected"] == 2
    assert stats["densification_net_added"] == 4
    assert model.get_xyz.shape[0] - before == 4

    model = cuda_gaussians(scales, gradient_scores=gradients)
    before = model.get_xyz.shape[0]

    stats = model.densify_gradient_topk_with_budget(
        budget=3,
        scene_extent=1.0,
        max_gaussians=100,
        split_N=3,
        return_stats=True,
    )

    assert stats["split_selected"] == 1
    assert stats["densification_net_added"] == 2
    assert stats["densification_net_added"] <= 3
    assert model.get_xyz.shape[0] - before == 2
