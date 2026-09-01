#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import time
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.budget_densification import (
    build_gradient_candidate_mask,
    build_official_clone_split_masks,
    compute_gradient_score,
    select_gradient_gated_demand_value_topk,
    select_gradient_gated_value_topk,
    select_gradient_topk,
)
from utils.edge_support import aggregate_multiview_edge_support, compute_edge_map
from utils.gestalt_loss import (
    compute_normal_continuity_loss,
    compute_plane_continuity_loss,
    select_same_surface_edges,
    subsample_edges,
)
from utils.structural_graph import (
    build_knn_graph,
    compute_continuity_defect,
    compute_geometric_turning,
    compute_redundancy,
    estimate_gaussian_normals,
)
from utils.training_mode_utils import (
    compute_effective_densification_budget,
    should_run_densification,
    validate_densification_mode,
)
from utils.value_allocation import (
    compute_demand_weighted_value_score,
    compute_observation_scarcity,
    compute_refine_utility,
    compute_structural_value,
    robust_normalize,
    select_gradient_priority_value_rerank_topk,
)
from utils.value_diagnostics import (
    build_multiview_value_diagnostics,
    compute_value_selection_diagnostics,
    compute_value_rerank_diagnostics,
    format_multiview_value_diagnostics_log,
    format_value_diagnostics_log,
    format_value_rerank_diagnostics_log,
)
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")
    densification_mode = validate_densification_mode(opt)
    value_densification_mode = densification_mode in (
        "value_allocation",
        "value_gestalt",
        "demand_value",
        "demand_value_gestalt",
        "value_rerank",
        "value_rerank_gestalt",
    )
    use_gestalt_loss = densification_mode in ("value_gestalt", "demand_value_gestalt", "value_rerank_gestalt")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    train_cameras = scene.getTrainCameras()
    gaussians.ensure_visibility_history(len(train_cameras))
    camera_uid_to_train_index = {
        camera.uid: index
        for index, camera in enumerate(train_cameras)
    }
    value_edge_maps = initialize_value_edge_maps(train_cameras) if value_densification_mode else None
    gestalt_neighbor_indices = None
    gestalt_graph_last_refresh = None

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = train_cameras.copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gaussians.update_visibility(vind, radii > 0)

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        gestalt_stats = None
        if use_gestalt_loss and iteration > opt.gestalt_warmup:
            gestalt_graph_build_time = 0.0
            graph_cache_stale = (
                gestalt_neighbor_indices is None
                or gestalt_graph_last_refresh is None
                or iteration - gestalt_graph_last_refresh >= opt.gestalt_graph_refresh_interval
            )
            if graph_cache_stale:
                _sync_cuda_for_timing()
                gestalt_graph_start = time.perf_counter()
                gestalt_neighbor_indices = build_knn_graph(gaussians.get_xyz, k=opt.knn_k)
                _sync_cuda_for_timing()
                gestalt_graph_build_time = time.perf_counter() - gestalt_graph_start
                gestalt_graph_last_refresh = iteration

            _sync_cuda_for_timing()
            gestalt_loss_start = time.perf_counter()
            normals = estimate_gaussian_normals(gaussians.get_scaling, gaussians.get_rotation)
            edges = select_same_surface_edges(
                gaussians.get_xyz.detach(),
                normals.detach(),
                gestalt_neighbor_indices,
                visible_view_count=gaussians.visible_view_count,
                normal_threshold=0.9,
                max_distance=None,
            )
            edges = subsample_edges(edges, opt.gestalt_edge_sample_num)
            plane_loss = compute_plane_continuity_loss(gaussians.get_xyz, normals, edges)
            normal_loss = compute_normal_continuity_loss(normals, edges)
            gestalt_loss = plane_loss + float(opt.lambda_normal) * normal_loss
            weighted_gestalt_loss = float(opt.lambda_gestalt) * gestalt_loss
            loss = loss + weighted_gestalt_loss
            _sync_cuda_for_timing()
            gestalt_loss_time = time.perf_counter() - gestalt_loss_start
            gestalt_stats = {
                "edge_count": int(edges.shape[0]),
                "plane_loss": plane_loss,
                "normal_loss": normal_loss,
                "gestalt_loss": gestalt_loss,
                "weighted_gestalt_loss": weighted_gestalt_loss,
                "gestalt_graph_build_time": gestalt_graph_build_time,
                "gestalt_loss_time": gestalt_loss_time,
            }

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if gestalt_stats is not None:
                if tb_writer:
                    tb_writer.add_scalar("gestalt/edge_count", gestalt_stats["edge_count"], iteration)
                    tb_writer.add_scalar("gestalt/plane_loss", gestalt_stats["plane_loss"].item(), iteration)
                    tb_writer.add_scalar("gestalt/normal_loss", gestalt_stats["normal_loss"].item(), iteration)
                    tb_writer.add_scalar("gestalt/loss", gestalt_stats["gestalt_loss"].item(), iteration)
                    tb_writer.add_scalar("gestalt/weighted_loss", gestalt_stats["weighted_gestalt_loss"].item(), iteration)
                    tb_writer.add_scalar("gestalt/graph_build_time", gestalt_stats["gestalt_graph_build_time"], iteration)
                    tb_writer.add_scalar("gestalt/loss_time", gestalt_stats["gestalt_loss_time"], iteration)
                if iteration % opt.structural_log_interval == 0:
                    print_gestalt_log(iteration, gestalt_stats)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if should_run_densification(iteration, opt.densify_from_iter, opt.densify_until_iter, opt.densification_interval):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if densification_mode == "official":
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                    elif densification_mode == "budget_gradient":
                        apply_budget_gradient_densification(gaussians, opt, scene.cameras_extent, size_threshold, radii, iteration)
                    elif densification_mode in ("value_allocation", "value_gestalt", "demand_value", "demand_value_gestalt", "value_rerank", "value_rerank_gestalt"):
                        apply_value_allocation_densification(
                            gaussians,
                            opt,
                            train_cameras,
                            value_edge_maps,
                            camera_uid_to_train_index,
                            scene.cameras_extent,
                            size_threshold,
                            radii,
                            iteration,
                            mode=densification_mode,
                        )
                    else:
                        raise ValueError(f"Unknown densification_mode '{densification_mode}'.")
                    if use_gestalt_loss:
                        gestalt_neighbor_indices = None
                        gestalt_graph_last_refresh = None
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    print_visibility_statistics(gaussians)

def initialize_value_edge_maps(train_cameras):
    edge_maps = []
    with torch.no_grad():
        for camera in train_cameras:
            edge_maps.append(compute_edge_map(camera.original_image.detach().cpu()).cpu())
    return edge_maps

def apply_budget_gradient_densification(gaussians, opt, extent, size_threshold, radii, iteration):
    _sync_cuda_for_timing()
    densification_start = time.perf_counter()
    gaussians.tmp_radii = radii
    try:
        densify_stats = gaussians.densify_gradient_topk_with_budget(
            budget=opt.densification_budget,
            scene_extent=extent,
            max_gaussians=opt.max_gaussians,
            return_stats=True,
        )
        prune_stats = gaussians.apply_standard_pruning(0.005, extent, size_threshold, return_stats=True)
    finally:
        gaussians.tmp_radii = None
    _sync_cuda_for_timing()
    densification_time = time.perf_counter() - densification_start

    densify_stats["pruned"] = prune_stats["pruned"]
    densify_stats["final_count"] = gaussians.get_xyz.shape[0]
    print_densification_log(
        mode="budget_gradient",
        iteration=iteration,
        stats=densify_stats,
        densification_time=densification_time,
    )

def apply_value_allocation_densification(
    gaussians,
    opt,
    train_cameras,
    edge_maps,
    camera_uid_to_train_index,
    extent,
    size_threshold,
    radii,
    iteration,
    mode="value_allocation",
):
    device = gaussians.get_xyz.device
    _sync_cuda_for_timing()
    graph_start = time.perf_counter()
    neighbor_indices = build_knn_graph(gaussians.get_xyz, k=opt.knn_k)
    normals = estimate_gaussian_normals(gaussians.get_scaling, gaussians.get_rotation)
    turning = compute_geometric_turning(gaussians.get_xyz, normals, neighbor_indices)
    defect = compute_continuity_defect(gaussians.get_xyz, normals, neighbor_indices)
    redundancy = compute_redundancy(gaussians.get_xyz, gaussians.get_scaling, neighbor_indices)
    _sync_cuda_for_timing()
    graph_build_time = time.perf_counter() - graph_start

    value_start = time.perf_counter()
    observation = compute_observation_scarcity(gaussians.visible_view_count, opt.tau_e, opt.tau_s).to(device=device)
    boundary = aggregate_multiview_edge_support(
        xyz=gaussians.get_xyz,
        cameras=train_cameras,
        edge_maps=edge_maps,
        visibility_history=gaussians.visibility_history,
        camera_uid_to_train_index=camera_uid_to_train_index,
    )
    boundary_norm = robust_normalize(boundary, opt.normalization_low_quantile, opt.normalization_high_quantile)
    turning_norm = robust_normalize(turning, opt.normalization_low_quantile, opt.normalization_high_quantile)
    defect_norm = robust_normalize(defect, opt.normalization_low_quantile, opt.normalization_high_quantile)
    redundancy_norm = robust_normalize(redundancy, opt.normalization_low_quantile, opt.normalization_high_quantile)
    structural_value = compute_structural_value(
        boundary_norm,
        turning_norm,
        defect_norm,
        opt.omega_boundary,
        opt.omega_turning,
        opt.omega_defect,
    )
    utility = compute_refine_utility(
        observation,
        structural_value,
        redundancy_norm,
        opt.lambda_redundancy,
    )
    effective_budget = compute_effective_densification_budget(
        opt.densification_budget,
        opt.max_gaussians,
        gaussians.get_xyz.shape[0],
    )
    gradient_score = compute_gradient_score(gaussians.xyz_gradient_accum, gaussians.denom)
    allocation_score = None
    value_selected_mask = None
    value_rerank_diagnostics = None
    if mode in ("demand_value", "demand_value_gestalt"):
        allocation_score = compute_demand_weighted_value_score(gradient_score, utility)
        selected_mask = select_gradient_gated_demand_value_topk(
            utility,
            effective_budget,
            gradient_score,
            opt.densify_grad_threshold,
            return_mask=True,
        )
        value_selected_mask = select_gradient_gated_value_topk(
            utility,
            effective_budget,
            gradient_score,
            opt.densify_grad_threshold,
            return_mask=True,
        )
    elif mode in ("value_rerank", "value_rerank_gestalt"):
        gradient_candidate_mask = build_gradient_candidate_mask(gradient_score, opt.densify_grad_threshold)
        selected_mask = select_gradient_priority_value_rerank_topk(
            gradient_score,
            utility,
            gradient_candidate_mask,
            effective_budget,
            rerank_fraction=opt.value_rerank_fraction,
            return_mask=True,
        )
        value_rerank_diagnostics = compute_value_rerank_diagnostics(
            gradient_score=gradient_score,
            utility=utility,
            candidate_mask=gradient_candidate_mask,
            selected_mask=selected_mask,
            budget=effective_budget,
            rerank_fraction=opt.value_rerank_fraction,
        )
    else:
        selected_mask = select_gradient_gated_value_topk(
            utility,
            effective_budget,
            gradient_score,
            opt.densify_grad_threshold,
            return_mask=True,
        )
    clone_mask, split_mask = build_official_clone_split_masks(
        selected_mask,
        gaussians.get_scaling,
        gaussians.percent_dense,
        extent,
    )
    gradient_selected_mask = select_gradient_topk(
        gradient_score,
        effective_budget,
        candidate_mask=None,
        return_mask=True,
    )
    value_diagnostics = compute_value_selection_diagnostics(
        scaling=gaussians.get_scaling,
        selected_mask=selected_mask,
        clone_mask=clone_mask,
        split_mask=split_mask,
        percent_dense=gaussians.percent_dense,
        scene_extent=extent,
        gradient_score=gradient_score,
        gradient_selected_mask=gradient_selected_mask,
        official_gradient_threshold=opt.densify_grad_threshold,
        source_type=gaussians.source_type,
        boundary=boundary,
        turning=turning,
        defect=defect,
        redundancy=redundancy,
        utility=utility,
        allocation_score=allocation_score,
    )
    print(format_value_diagnostics_log(iteration, value_diagnostics))
    if value_rerank_diagnostics is not None:
        print(format_value_rerank_diagnostics_log(iteration, value_rerank_diagnostics))
    multiview_value_diagnostics = build_multiview_value_diagnostics(
        scaling=gaussians.get_scaling,
        selected_mask=selected_mask,
        effective_budget=effective_budget,
        gradient_score=gradient_score,
        official_gradient_threshold=opt.densify_grad_threshold,
        visible_view_count=gaussians.visible_view_count,
        observation=observation,
        boundary=boundary_norm,
        turning=turning_norm,
        defect=defect_norm,
        redundancy=redundancy_norm,
        utility=utility,
        allocation_score=allocation_score,
        structural_value=structural_value,
        lambda_redundancy=opt.lambda_redundancy,
        source_type=gaussians.source_type,
        value_reference_mask=value_selected_mask,
    )
    print(format_multiview_value_diagnostics_log(iteration, multiview_value_diagnostics))
    _sync_cuda_for_timing()
    value_compute_time = time.perf_counter() - value_start

    densification_start = time.perf_counter()
    gaussians.tmp_radii = radii
    try:
        densify_stats = gaussians.densify_with_budget(
            clone_mask,
            split_mask,
            budget=opt.densification_budget,
            max_gaussians=opt.max_gaussians,
            scene_extent=extent,
            return_stats=True,
        )
        prune_stats = gaussians.apply_standard_pruning(0.005, extent, size_threshold, return_stats=True)
    finally:
        gaussians.tmp_radii = None
    _sync_cuda_for_timing()
    densification_time = time.perf_counter() - densification_start

    densify_stats["value_selected"] = int(selected_mask.sum().item())
    densify_stats["pruned"] = prune_stats["pruned"]
    densify_stats["final_count"] = gaussians.get_xyz.shape[0]
    value_stats = {
        "mean_O": observation.mean().item() if observation.numel() else 0.0,
        "mean_B": boundary.mean().item() if boundary.numel() else 0.0,
        "mean_K": turning.mean().item() if turning.numel() else 0.0,
        "mean_D": defect.mean().item() if defect.numel() else 0.0,
        "mean_R": redundancy.mean().item() if redundancy.numel() else 0.0,
        "mean_U": utility.mean().item() if utility.numel() else 0.0,
        "max_U": utility.max().item() if utility.numel() else 0.0,
    }
    if allocation_score is not None:
        value_stats["mean_A"] = allocation_score.mean().item() if allocation_score.numel() else 0.0
        value_stats["max_A"] = allocation_score.max().item() if allocation_score.numel() else 0.0

    print_densification_log(
        mode=mode,
        iteration=iteration,
        stats=densify_stats,
        value_stats=value_stats,
        graph_build_time=graph_build_time,
        value_compute_time=value_compute_time,
        densification_time=densification_time,
    )

def print_gestalt_log(iteration, stats):
    message = (
        f"\n[ITER {iteration}] gestalt "
        f"edge_count={stats['edge_count']} "
        f"plane_loss={stats['plane_loss'].item():.6f} "
        f"normal_loss={stats['normal_loss'].item():.6f} "
        f"gestalt_loss={stats['gestalt_loss'].item():.6f} "
        f"weighted_gestalt_loss={stats['weighted_gestalt_loss'].item():.6f} "
        f"gestalt_graph_build_time={stats['gestalt_graph_build_time']:.4f}s "
        f"gestalt_loss_time={stats['gestalt_loss_time']:.4f}s"
    )
    print(message)

def print_densification_log(
    mode,
    iteration,
    stats,
    value_stats=None,
    graph_build_time=None,
    value_compute_time=None,
    densification_time=None,
):
    message = (
        f"\n[ITER {iteration}] densification mode={mode} "
        f"count={stats['final_count']} "
        f"requested_budget={stats['requested_budget']} "
        f"effective_budget={stats['effective_budget']} "
        f"clone={stats['clone_selected']} "
        f"split={stats['split_selected']} "
        f"net_added={stats['densification_net_added']} "
        f"pruned={stats['pruned']}"
    )
    if value_stats is not None:
        message += (
            f" mean_O={value_stats['mean_O']:.6f}"
            f" mean_B={value_stats['mean_B']:.6f}"
            f" mean_K={value_stats['mean_K']:.6f}"
            f" mean_D={value_stats['mean_D']:.6f}"
            f" mean_R={value_stats['mean_R']:.6f}"
            f" mean_U={value_stats['mean_U']:.6f}"
            f" max_U={value_stats['max_U']:.6f}"
        )
        if "mean_A" in value_stats:
            message += (
                f" mean_A={value_stats['mean_A']:.6f}"
                f" max_A={value_stats['max_A']:.6f}"
            )
    if graph_build_time is not None:
        message += f" graph_build_time={graph_build_time:.4f}s"
    if value_compute_time is not None:
        message += f" value_compute_time={value_compute_time:.4f}s"
    if densification_time is not None:
        message += f" densification_time={densification_time:.4f}s"
    print(message)

def _sync_cuda_for_timing():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def print_visibility_statistics(gaussians):
    visible_view_count = gaussians.visible_view_count
    if visible_view_count is None or visible_view_count.numel() == 0:
        print("\nVisible statistics: unavailable")
        return

    counts = visible_view_count.float()
    print("\nGaussian number:")
    print(gaussians.get_xyz.shape[0])
    print("\nVisible statistics:")
    print("mean visible views:")
    print(counts.mean().item())
    print("max visible views:")
    print(visible_view_count.max().item())
    print("min visible views:")
    print(visible_view_count.min().item())

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
