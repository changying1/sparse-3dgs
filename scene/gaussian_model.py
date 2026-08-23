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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.budget_densification import (
    build_official_clone_split_masks,
    compute_gradient_score,
    select_gradient_topk,
)

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.visibility_history = None
        self.visible_view_count = None
        self.birth_iteration = None
        self.source_type = None
        self.parent_index = None
        self.completion_support_count = None
        self.completion_age = None
        self.is_completion = None
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._capture_structural_state(),
        )
    
    def restore(self, model_args, training_args):
        if isinstance(model_args, dict):
            official_state = model_args["official_state"]
            structural_state = model_args.get("structural_state", None)
        elif len(model_args) == 13:
            official_state = model_args[:12]
            structural_state = model_args[12]
        else:
            official_state = model_args
            structural_state = None

        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = official_state
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self._restore_structural_state(structural_state)

    def _structural_state_names(self):
        return (
            "visibility_history",
            "visible_view_count",
            "birth_iteration",
            "source_type",
            "parent_index",
            "completion_support_count",
            "completion_age",
            "is_completion",
        )

    def _capture_structural_state(self):
        return {name: getattr(self, name) for name in self._structural_state_names()}

    def _restore_structural_state(self, structural_state):
        if structural_state is None:
            self.visibility_history = None
            self._initialize_structural_state(self.get_xyz.shape[0], device=self.get_xyz.device, num_views=None)
            return

        for name in self._structural_state_names():
            setattr(self, name, structural_state.get(name, None))
        self._ensure_structural_state_count(self.get_xyz.shape[0], device=self.get_xyz.device)

    def _initialize_structural_state(self, count, device=None, num_views=None):
        if device is None:
            device = self.get_xyz.device
        if num_views is None:
            self.visibility_history = None
        else:
            self.visibility_history = torch.zeros((count, num_views), dtype=torch.bool, device=device)
        self.visible_view_count = torch.zeros((count), dtype=torch.long, device=device)
        self.birth_iteration = torch.zeros((count), dtype=torch.long, device=device)
        self.source_type = torch.zeros((count), dtype=torch.long, device=device)
        self.parent_index = torch.full((count,), -1, dtype=torch.long, device=device)
        self.completion_support_count = torch.zeros((count), dtype=torch.long, device=device)
        self.completion_age = torch.zeros((count), dtype=torch.long, device=device)
        self.is_completion = torch.zeros((count), dtype=torch.bool, device=device)

    def _ensure_structural_state_count(self, count, device=None):
        if device is None:
            device = self.get_xyz.device
        if self.visible_view_count is None:
            self.visible_view_count = torch.zeros((count), dtype=torch.long, device=device)
        if self.birth_iteration is None:
            self.birth_iteration = torch.zeros((count), dtype=torch.long, device=device)
        if self.source_type is None:
            self.source_type = torch.zeros((count), dtype=torch.long, device=device)
        if self.parent_index is None:
            self.parent_index = torch.full((count,), -1, dtype=torch.long, device=device)
        if self.completion_support_count is None:
            self.completion_support_count = torch.zeros((count), dtype=torch.long, device=device)
        if self.completion_age is None:
            self.completion_age = torch.zeros((count), dtype=torch.long, device=device)
        if self.is_completion is None:
            self.is_completion = torch.zeros((count), dtype=torch.bool, device=device)

    def _prune_structural_state(self, valid_points_mask):
        for name in self._structural_state_names():
            state = getattr(self, name)
            if state is not None:
                setattr(self, name, state[valid_points_mask])

    def _append_structural_state(self, selected_pts_mask, source_type, repeat_count=1):
        self._ensure_structural_state_count(selected_pts_mask.shape[0], device=selected_pts_mask.device)
        parent_indices = torch.nonzero(selected_pts_mask, as_tuple=False).squeeze(1)
        if repeat_count != 1:
            parent_indices = parent_indices.repeat(repeat_count)
        new_count = parent_indices.shape[0]

        if self.visibility_history is not None:
            new_visibility_history = self.visibility_history[selected_pts_mask]
            if repeat_count != 1:
                new_visibility_history = new_visibility_history.repeat(repeat_count, 1)
            self.visibility_history = torch.cat((self.visibility_history, new_visibility_history), dim=0)

        if self.visible_view_count is not None:
            new_visible_view_count = self.visible_view_count[selected_pts_mask]
            if repeat_count != 1:
                new_visible_view_count = new_visible_view_count.repeat(repeat_count)
            self.visible_view_count = torch.cat((self.visible_view_count, new_visible_view_count), dim=0)

        device = self.get_xyz.device
        self.birth_iteration = torch.cat((self.birth_iteration, torch.zeros((new_count), dtype=torch.long, device=device)), dim=0)
        self.source_type = torch.cat((self.source_type, torch.full((new_count,), source_type, dtype=torch.long, device=device)), dim=0)
        self.parent_index = torch.cat((self.parent_index, parent_indices.to(device=device, dtype=torch.long)), dim=0)
        self.completion_support_count = torch.cat((self.completion_support_count, torch.zeros((new_count), dtype=torch.long, device=device)), dim=0)
        self.completion_age = torch.cat((self.completion_age, torch.zeros((new_count), dtype=torch.long, device=device)), dim=0)
        self.is_completion = torch.cat((self.is_completion, torch.zeros((new_count), dtype=torch.bool, device=device)), dim=0)

    def ensure_visibility_history(self, num_views):
        count = self.get_xyz.shape[0]
        device = self.get_xyz.device
        self._ensure_structural_state_count(count, device=device)
        if self.visibility_history is None:
            self.visibility_history = torch.zeros((count, num_views), dtype=torch.bool, device=device)
            self.visible_view_count = torch.zeros((count), dtype=torch.long, device=device)
            return
        if self.visibility_history.shape[0] != count:
            raise ValueError("visibility_history first dimension must match the current Gaussian count.")
        if self.visibility_history.shape[1] != num_views:
            raise ValueError("visibility_history second dimension must match the training view count.")
        if self.visible_view_count is None or self.visible_view_count.shape[0] != count:
            self.visible_view_count = self.visibility_history.sum(dim=1).to(dtype=torch.long)

    def update_visibility(self, view_id, visible_mask):
        visible_mask = visible_mask.to(device=self.get_xyz.device, dtype=torch.bool).reshape(-1)
        if visible_mask.shape[0] != self.get_xyz.shape[0]:
            raise ValueError("visible_mask first dimension must match the current Gaussian count.")
        if self.visibility_history is None:
            self.ensure_visibility_history(view_id + 1)
        if view_id < 0 or view_id >= self.visibility_history.shape[1]:
            raise ValueError("view_id must be within the visibility_history view dimension.")

        previous_visible = self.visibility_history[:, view_id]
        newly_visible = torch.logical_and(visible_mask, torch.logical_not(previous_visible))
        self.visibility_history[:, view_id] = torch.logical_or(previous_visible, visible_mask)
        self.visible_view_count += newly_visible.to(dtype=self.visible_view_count.dtype)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
        self._initialize_structural_state(self.get_xyz.shape[0], device=self.get_xyz.device, num_views=len(cam_infos))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]
        self._prune_structural_state(valid_points_mask)

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def _validate_densification_mask(self, selected_mask):
        if selected_mask.shape[0] != self.get_xyz.shape[0]:
            raise ValueError("selected_mask first dimension must match the current Gaussian count.")
        return selected_mask.to(device=self.get_xyz.device, dtype=torch.bool)

    def _trim_mask_to_count(self, selected_mask, max_count):
        if max_count is None:
            return selected_mask
        if max_count < 0:
            raise ValueError("max_count must be non-negative.")
        selected_indices = torch.nonzero(selected_mask, as_tuple=False).squeeze(1)
        if selected_indices.shape[0] <= max_count:
            return selected_mask
        trimmed_mask = torch.zeros_like(selected_mask, dtype=torch.bool)
        trimmed_mask[selected_indices[:max_count]] = True
        return trimmed_mask

    def _tmp_radii_for_mask(self, selected_mask):
        if not hasattr(self, "tmp_radii") or self.tmp_radii is None:
            return torch.zeros((selected_mask.sum()), device="cuda")
        return self.tmp_radii[selected_mask]

    def densify_and_split_by_mask(self, selected_mask, grads=None, scene_extent=None, N=2):
        selected_pts_mask = self._validate_densification_mask(selected_mask)
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self._tmp_radii_for_mask(selected_pts_mask).repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)
        self._append_structural_state(selected_pts_mask, source_type=2, repeat_count=N)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        self.densify_and_split_by_mask(selected_pts_mask, grads, scene_extent, N)

    def densify_and_clone_by_mask(self, selected_mask, grads=None, scene_extent=None):
        selected_pts_mask = self._validate_densification_mask(selected_mask)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self._tmp_radii_for_mask(selected_pts_mask)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)
        self._append_structural_state(selected_pts_mask, source_type=1)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        self.densify_and_clone_by_mask(selected_pts_mask, grads, scene_extent)

    def action_cost(self, action, N=2):
        if action == "clone":
            return 1
        if action == "split":
            return N - 1
        raise ValueError(f"Unknown densification action: {action}")

    def densify_with_budget(self, clone_mask, split_mask, budget, max_gaussians=None, scene_extent=None, split_N=2, return_stats=False):
        clone_mask = self._validate_densification_mask(clone_mask)
        split_mask = self._validate_densification_mask(split_mask)
        if clone_mask.shape[0] != split_mask.shape[0]:
            raise ValueError("clone_mask and split_mask must have the same length.")
        if torch.logical_and(clone_mask, split_mask).any():
            raise ValueError("clone_mask and split_mask cannot select the same Gaussian.")
        if budget < 0:
            raise ValueError("budget must be non-negative.")
        if split_N < 2:
            raise ValueError("split_N must be at least 2.")

        before_count = self.get_xyz.shape[0]
        action_cost = {"clone": self.action_cost("clone"), "split": self.action_cost("split", N=split_N)}
        requested_budget = int(budget)
        effective_budget = requested_budget
        if max_gaussians is not None:
            effective_budget = min(effective_budget, max_gaussians - before_count)
        effective_budget = max(0, effective_budget)

        selected_clone_mask = torch.zeros_like(clone_mask, dtype=torch.bool)
        selected_split_mask = torch.zeros_like(split_mask, dtype=torch.bool)
        remaining_budget = effective_budget
        for action, mask in (("clone", clone_mask), ("split", split_mask)):
            cost = action_cost[action]
            if cost <= 0:
                raise ValueError("densification action cost must be positive.")
            if remaining_budget < cost:
                continue
            count = min(int(mask.sum().item()), remaining_budget // cost)
            trimmed_mask = self._trim_mask_to_count(mask, count)
            if action == "clone":
                selected_clone_mask = trimmed_mask
            else:
                selected_split_mask = trimmed_mask
            remaining_budget -= int(trimmed_mask.sum().item()) * cost

        clone_mask = selected_clone_mask
        split_mask = selected_split_mask
        net_added = int(clone_mask.sum().item()) * action_cost["clone"] + int(split_mask.sum().item()) * action_cost["split"]
        stats = {
            "requested_budget": requested_budget,
            "effective_budget": effective_budget,
            "clone_selected": int(clone_mask.sum().item()),
            "split_selected": int(split_mask.sum().item()),
            "densification_net_added": net_added,
            "before_count": before_count,
            "after_densification_count": before_count,
            "pruned": 0,
        }
        if not clone_mask.any() and not split_mask.any():
            return stats if return_stats else None

        n_init_points = self.get_xyz.shape[0]
        self.densify_and_clone_by_mask(clone_mask, scene_extent=scene_extent)
        if self.get_xyz.shape[0] > n_init_points:
            split_mask = torch.cat((split_mask, torch.zeros(self.get_xyz.shape[0] - n_init_points, device="cuda", dtype=bool)))
        self.densify_and_split_by_mask(split_mask, scene_extent=scene_extent, N=split_N)
        stats["after_densification_count"] = self.get_xyz.shape[0]
        return stats if return_stats else None

    def select_gradient_densification_masks(self, budget, scene_extent, max_gaussians=None, candidate_mask=None):
        gradient_score = compute_gradient_score(self.xyz_gradient_accum, self.denom)
        effective_budget = int(budget)
        if max_gaussians is not None:
            effective_budget = min(effective_budget, max_gaussians - self.get_xyz.shape[0])
        effective_budget = max(0, effective_budget)
        selected_mask = select_gradient_topk(gradient_score, effective_budget, candidate_mask=candidate_mask, return_mask=True)
        clone_mask, split_mask = build_official_clone_split_masks(
            selected_mask,
            self.get_scaling,
            self.percent_dense,
            scene_extent,
        )
        if torch.logical_and(clone_mask, split_mask).any():
            raise RuntimeError("clone and split masks must be disjoint.")
        return clone_mask, split_mask, gradient_score, selected_mask

    def densify_gradient_topk_with_budget(self, budget, scene_extent, max_gaussians=None, candidate_mask=None, split_N=2, return_stats=False):
        clone_mask, split_mask, gradient_score, selected_mask = self.select_gradient_densification_masks(
            budget=budget,
            scene_extent=scene_extent,
            max_gaussians=max_gaussians,
            candidate_mask=candidate_mask,
        )
        stats = self.densify_with_budget(
            clone_mask,
            split_mask,
            budget=budget,
            max_gaussians=max_gaussians,
            scene_extent=scene_extent,
            split_N=split_N,
            return_stats=True,
        )
        stats["gradient_selected"] = int(selected_mask.sum().item())
        stats["gradient_score"] = gradient_score
        return stats if return_stats else None

    def apply_densification(self, grads, max_grad, extent):
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

    def apply_standard_pruning(self, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.apply_densification(grads, max_grad, extent)
        self.apply_standard_pruning(min_opacity, extent, max_screen_size)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
