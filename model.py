import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from MinkowskiEngine import SparseTensor
import MinkowskiEngine as ME
from utils.model import get_grid, ChannelPool, Flatten, NNBase
from utils.distributions import Categorical, DiagGaussian

import envs.utils.depth_utils as du
from plyfile import PlyData, PlyElement
from model3D.disnet import DisNet as model3d
from torch.utils import model_zoo
def get_model(args):
    '''Get the 3D model.'''

    model = model3d(cfg=args)
    return model


class Semantic_Mapping(nn.Module):

    """
    Semantic_Mapping
    """

    def __init__(self, args):
        super(Semantic_Mapping, self).__init__()

        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.cat_pred_threshold = args.cat_pred_threshold
        self.exp_pred_threshold = args.exp_pred_threshold
        self.map_pred_threshold = args.map_pred_threshold
        self.num_sem_categories = args.num_sem_categories

        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = args.camera_height * 100.
        self.shift_loc = [self.vision_range *
                          self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)

        self.pool = ChannelPool(1)

        vr = self.vision_range
        if args.use_3dmodel:
            self.model3 = get_model(args)
            checkpoint = model_zoo.load_url(args.model_path, progress=True)
            self.model3.load_state_dict(checkpoint['state_dict'], strict=True)
        self.init_grid = torch.zeros(
            args.num_processes, 1 + self.num_sem_categories, vr, vr,
            self.max_height - self.min_height
        ).float().to(self.device)
        self.feat = torch.ones(
            args.num_processes, 1 + self.num_sem_categories,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)
        self.init_grid_sim = torch.zeros(
        args.num_processes,  
        1,                  
        vr, vr, 
        self.max_height - self.min_height
        ).float().to(self.device)
        self.feat_sim = torch.zeros(
        args.num_processes,  
        1,                  
        self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)

    def forward(self, obs, pose_obs, maps_last, poses_last, sim_maps_last, text_features):
        bs, c, h, w = obs.size() #28

        depth = obs[:, 3, :, :]

        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale)
        agent_view_t = du.transform_camera_view_t(
            point_cloud_t, self.agent_height, 0, self.device)

        agent_view_centered_t = du.transform_pose_t(
            agent_view_t, self.shift_loc, self.device)

        points_3d = agent_view_centered_t.reshape(bs, -1, 3)
        all_point_features = []
  
        for i in range(bs):
            current_points = points_3d[i]
            batch_index = torch.zeros(current_points.shape[0], 1, device=self.device)
       
            coords = torch.cat([batch_index, current_points], dim=1).int()    

            unique_coords, inverse_indices = torch.unique(coords, dim=0, return_inverse=True)

            feat = torch.ones(unique_coords.shape[0], 3, device=self.device)

            sinput = SparseTensor(feat, unique_coords)

            with torch.no_grad():
                self.model3.eval()
                current_point_features = self.model3(sinput)
           
            current_point_features = current_point_features[inverse_indices]
     
            similarity = current_point_features.half() @ text_features[i].t()
            similarity_scores = (similarity - similarity.min()) / (similarity.max() - similarity.min())  # [N]
            all_point_features.append(similarity_scores)
      
        similarity_scores = torch.cat(all_point_features, dim=0)
        self.feat_sim[:, 0, :] = similarity_scores.view(bs, -1)
        

        
        
        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               vision_range // 2.) / vision_range * 2.
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
                              (max_h + min_h) // 2.) / (max_h - min_h) * 2.
      
        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(
            obs[:, 4:4+(self.num_sem_categories), :, :]
        ).view(bs, self.num_sem_categories, h // self.du_scale * w // self.du_scale)

        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

        voxels = du.splat_feat_nd(
            self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3)
        
        voxels_sim = du.splat_feat_nd(
        self.init_grid_sim * 0., 
        self.feat_sim,
        XYZ_cm_std
        ).transpose(2, 3)
        min_z = int(25 / z_resolution - min_h)
        max_z = int((self.agent_height + 1) / z_resolution - min_h)

        agent_height_proj = voxels[..., min_z:max_z].sum(4)
     
        all_height_proj = voxels.sum(4)
        similarity_height_proj = voxels_sim[..., min_z:max_z].sum(4)

        fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        pose_pred = poses_last

        agent_view = torch.zeros(bs, c-2,
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device)

        sim_view = torch.zeros(bs, 1,
                             self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution
                             ).to(self.device)
        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0)

        sim_view[:, 0:1, y1:y2, x1:x2] = torch.clamp(
                similarity_height_proj, min=0.0, max=1.0)
        corrected_pose = pose_obs
      
        def get_new_pose_batch(pose, rel_pose_change):

            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()
     
        
        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                      self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)
       
        rotated_sim = F.grid_sample(sim_view, rot_mat, align_corners=True)
        translated_sim = F.grid_sample(rotated_sim, trans_mat, align_corners=True)
        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)
        sim_maps2 = torch.cat((sim_maps_last.unsqueeze(1), 
                                 translated_sim.unsqueeze(1)), 1)
        sim_map_pred, _ = torch.max(sim_maps2, 1)
        map_pred, _ = torch.max(maps2, 1)

        return fp_map_pred, map_pred, pose_pred, current_poses, sim_map_pred
