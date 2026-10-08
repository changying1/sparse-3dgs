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
import torchvision
import random
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils import depth_utils
from utils.runtime_compat import detached_tensor_to, should_sample_pseudo
from utils.i1_evidence_structural_loss import (
    combine_i1_loss,
    compute_i1_structural_loss,
    compute_stable_mask_from_gt_rgb,
    format_i1_log,
    resolve_i1_gate_mode,
    should_preserve_i1_densification_stats,
)
from utils.image_utils import normalize_depth
from torchmetrics.functional.regression import pearson_corrcoef
import matplotlib.pyplot as plt
from imageio import imwrite
import json
# from guidance.sd_utils import StableDiffusion
import torch.nn.functional as F
from torchvision import transforms as T

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    depth_utils.offload_depth_pro()
    print("[Speed Audit] DepthPro moved to CPU after camera depth precomputation.")
    # ---------------- Training Setup ----------------
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack, pseudo_stack = None, None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    bg_mask = None
    loss_accum = 0
    i1_stable_mask_cache = {}
    
    train_start_time = time.time()
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        gt_image = viewpoint_cam.original_image.cuda()   
             
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, is_train=True, iteration=iteration)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        Ll1 = l1_loss(image, gt_image) # (3,H,W)
        ssim_value = ssim(image, gt_image)
        loss = Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
                
        if args.depth_loss:
            rendered_depth = render_pkg["depth"][0]     
            depth_pro_depth = detached_tensor_to(viewpoint_cam.depth_image, rendered_depth.device)
            depth_pro_depth = depth_pro_depth.reshape(-1, 1)
            rendered_depth = rendered_depth.reshape(-1, 1)
            depth_loss = (1 - pearson_corrcoef(depth_pro_depth, rendered_depth))
             
            loss += args.depth_weight * depth_loss 
            
            if iteration > args.end_sample_pseudo:
                args.depth_weight = 0.001
                
            if should_sample_pseudo(
                iteration,
                args.sample_pseudo_interval,
                args.start_sample_pseudo,
                args.end_sample_pseudo,
            ):
                if not pseudo_stack:
                    pseudo_stack = scene.getPseudoCameras().copy()
                pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))

                render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background)
                rendered_depth_pseudo = render_pkg_pseudo["depth"][0]               
                with depth_utils.depth_pro_on_cuda():
                    depth_pro_depth_pseudo = depth_utils.estimate_depth_pro(
                        render_pkg_pseudo["render"], mode='train'
                    )
                depth_pro_depth_pseudo = detached_tensor_to(
                    depth_pro_depth_pseudo, rendered_depth_pseudo.device
                )
                rendered_depth_pseudo = rendered_depth_pseudo.reshape(-1, 1)
                depth_pro_depth_pseudo = depth_pro_depth_pseudo.reshape(-1, 1)
                depth_loss_pseudo = (1 - pearson_corrcoef(rendered_depth_pseudo, depth_pro_depth_pseudo))
                
                if torch.isnan(depth_loss_pseudo).sum() == 0:
                    loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
                    loss += loss_scale * args.depth_pseudo_weight * depth_loss_pseudo
                            
        baseline_loss = loss
        i1_gate_mode = resolve_i1_gate_mode(
            iteration,
            enabled=getattr(opt, "enable_i1", False),
            phase1_start=getattr(opt, "i1_phase1_start", 1600),
            phase2_start=getattr(opt, "i1_phase2_start", 2000),
            end_iter=getattr(opt, "i1_end_iter", 5000),
        )
        i1_active = i1_gate_mode is not None and viewpoint_cam.depth_image is not None
        i1_structural_loss = None
        if i1_active:
            camera_id = viewpoint_cam.uid
            if camera_id not in i1_stable_mask_cache:
                stable_mask, _, _ = compute_stable_mask_from_gt_rgb(
                    gt_image.detach(),
                    stable_quantile=getattr(opt, "i1_stable_quantile", 0.80),
                )
                i1_stable_mask_cache[camera_id] = stable_mask.detach()
            i1_structural_loss, i1_stats, _ = compute_i1_structural_loss(
                render_pkg["depth"][0],
                viewpoint_cam.depth_image,
                stable_mask=i1_stable_mask_cache[camera_id].to(render_pkg["depth"].device),
                gate_mode=i1_gate_mode,
            )
            if iteration % 100 == 0:
                print(
                    format_i1_log(
                        iteration,
                        camera_id,
                        i1_stats,
                        getattr(opt, "i1_structural_weight", 0.05),
                    ),
                    flush=True,
                )
        loss = combine_i1_loss(
            baseline_loss,
            i1_structural_loss,
            getattr(opt, "i1_structural_weight", 0.05),
            i1_gate_mode if i1_active else None,
        )
        preserve_i1_densification_stats = should_preserve_i1_densification_stats(
            getattr(opt, "i1_preserve_baseline_densification_stats", True),
            i1_gate_mode if i1_active else None,
            iteration,
            opt.densify_until_iter,
        )
        baseline_viewspace_grad = None
        if preserve_i1_densification_stats:
            baseline_viewspace_grad = torch.autograd.grad(
                baseline_loss,
                viewspace_point_tensor,
                retain_graph=True,
                allow_unused=False,
            )[0].detach()
            viewspace_point_tensor.grad = None

        loss.backward()
        iter_end.record()
                            
        with torch.no_grad():
            # Progress bar
            if iteration > opt.densify_from_iter:
                loss_accum += loss.clone().detach().item()

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                dataset,
                tb_writer,
                iteration,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
            )
            if tb_writer:
                torch.cuda.synchronize()
                total_elapsed = time.time() - train_start_time
                tb_writer.add_scalar('train_time/elapsed_sec', total_elapsed, iteration)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(
                    viewspace_point_tensor,
                    visibility_filter,
                    grad_override=(
                        baseline_viewspace_grad
                        if preserve_i1_densification_stats
                        else None
                    ),
                )
                        
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


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

def training_report(args, tb_writer, iteration, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        # tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(len(scene.getTrainCameras()))]})

        with depth_utils.depth_pro_on_cuda():
            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0.0
                    psnr_test = 0.0
                    for idx, viewpoint in enumerate(config['cameras']):
                        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                        image = render_pkg["render"]
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        rendered_depth = render_pkg["depth"]

                        # render_image -> estimated_depth
                        render_depth_pro = depth_utils.estimate_depth_pro(image)

                        # normalize depth
                        rendered_depth = normalize_depth(rendered_depth)
                        render_depth_pro = normalize_depth(render_depth_pro)[None]

                        if tb_writer and (idx < 5): # default 5 plot
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/render_depth".format(viewpoint.image_name), rendered_depth[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/render_depth_pro".format(viewpoint.image_name), render_depth_pro[None], global_step=iteration)

                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                                gt_depth_pro = depth_utils.estimate_depth_pro(gt_image)
                                gt_depth_pro = normalize_depth(gt_depth_pro)[None]
                                tb_writer.add_images(config['name'] + "_view_{}/ground_truth_depth_pro".format(viewpoint.image_name), gt_depth_pro[None], global_step=iteration)

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    seed_everything(42)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
