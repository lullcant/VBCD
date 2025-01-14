import torch
from models.backbone import ResidualUNet3D
from torch import nn
from pytorch3d.ops import sample_farthest_points
import torch
import torch.nn.functional as F


def volume_to_point_cloud_tensor(
    volume: torch.Tensor, 
    voxel_size=(0.2, 0.2, 0.2), 
  
):


    device = volume.device
    B, C, D, H, W = volume.shape
    assert C == 1

    
    if not torch.is_tensor(voxel_size):
        voxel_size_tensor = torch.tensor(voxel_size, 
                                         device=device, 
                                         dtype=torch.float32)  # [3]
    else:
        voxel_size_tensor = voxel_size.to(device=device, dtype=torch.float32)     


    foreground_prob = torch.sigmoid(volume)                     # [B, 1, D, H, W]
    foreground_mask = (foreground_prob > 0.5).squeeze(1)        # [B, D, H, W]
   
    nonzero_indices = torch.nonzero(foreground_mask, as_tuple=False)
   
    b = nonzero_indices[:, 0].long()  # batch index
    d = nonzero_indices[:, 1].long()  # depth index
    h = nonzero_indices[:, 2].long()  # height index
    w = nonzero_indices[:, 3].long()  # width index

    coords_dhw = torch.stack([d ,h, w], dim=-1).float()  # [N, 3]

    coords_xyz =  coords_dhw * voxel_size_tensor  # [N, 3]

    
    b_float = b.float().unsqueeze(-1)  # [N, 1]
    point_cloud = torch.cat([b_float,coords_xyz], dim=-1)  # [N, 4]

    return point_cloud


class CrownMVM(nn.Module):
    def __init__(self,scale=128,in_channels=1,out_channels=4):
        super(CrownMVM, self).__init__()
        self.unet = ResidualUNet3D(in_channels=in_channels,out_channels=out_channels,is_segmentation=False)
        self.offset_normal_predictor = nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,6))
        #self.feature_project = nn.Linear(64,6)
    def mask_feature_to_nxc(self,mask, feature):
        '''
        Featurr selection
        '''    
    
        foreground_prob = torch.sigmoid(mask[:,:1,:,:,:])            # [B, C, D, H, W]
        foreground_mask = (foreground_prob > 0.5).squeeze(1)  # [B, D, H, W]
        nonzero_indices = torch.nonzero(foreground_mask, as_tuple=False)  # [N, 4]
    
        b_idx = nonzero_indices[:, 0]  # [N]
        d_idx = nonzero_indices[:, 1]  # [N]
        h_idx = nonzero_indices[:, 2]  # [N]
        w_idx = nonzero_indices[:, 3]  # [N]
        selected_feat = feature[b_idx, :, d_idx, h_idx, w_idx]  # [N, C]
        offset_normal = self.offset_normal_predictor(selected_feat)
        return offset_normal
    
    def forward(self,batched_pna_voxel,prompt=None):
        voxel,final_feature = self.unet(batched_pna_voxel,prompt)
        offset_normal = self.mask_feature_to_nxc(voxel,final_feature)
        offset = offset_normal[:,:3]
        normals = offset_normal[:,3:]
        voxel_ind = voxel[:,:1,:,:,:]
        voxel_normal = voxel[:,1:,:,:,:]
        pred_pc = volume_to_point_cloud_tensor(volume=voxel_ind,voxel_size=(0.15625,0.15625,0.15625))
        refined_pos = offset + pred_pc[:,1:]
        refined_pos_with_normal = torch.cat([refined_pos,normals],dim=1)
        batch_x = pred_pc[:,0]
        return voxel_ind,voxel_normal,refined_pos_with_normal,batch_x

