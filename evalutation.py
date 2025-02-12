import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from accelerate import Accelerator
from mydataset.Dentaldataset import *
from models.crownmvm2 import CrownMVM
import SimpleITK as sitk
import os
import numpy as np
import random
from pytorch3d.loss import chamfer_distance
import torch.nn.functional as F
import open3d as o3d
from scipy.spatial import distance_matrix
from models.metric import fidelity,fidelity_ratio_and_point_cloud
from SAP.src.dpsr import DPSR
from SAP.src.utils import *
teeth_dict = {
    'Incisors': { 'count': 0, 'total_chamfer_distance': 0,'total_fidelity':0,'total_f_value':0},
    'Canines': { 'count': 0, 'total_chamfer_distance': 0,'total_fidelity':0,'total_f_value':0},
    'Premolars': { 'count': 0, 'total_chamfer_distance': 0,'total_fidelity':0,'total_f_value':0},
    'Molars': {'count': 0, 'total_chamfer_distance': 0,'total_fidelity':0,'total_f_value':0}
}
def update_teeth_statistics(teeth_dict, file_name, chamfer_dist,fidelity,f_value):
   
    subdir = file_name.split('/')[-3]  
 
    if subdir.endswith(('1', '2')):
        tooth_type = 'Incisors'
    elif subdir.endswith('3'):
        tooth_type = 'Canines'
    elif subdir.endswith(('4', '5')):
        tooth_type = 'Premolars'
    elif subdir.endswith(('6', '7')):
        tooth_type = 'Molars'
    else:
        raise ValueError(f"Unrecognized subdir format: {subdir}")

    
    teeth_dict[tooth_type]['count'] += 1
    teeth_dict[tooth_type]['total_chamfer_distance'] += chamfer_dist
    teeth_dict[tooth_type]['total_fidelity'] += fidelity
    teeth_dict[tooth_type]['total_f_value'] += f_value
save_file = True
accelerator = Accelerator()



import pyvista as pv



def voxel_grid_to_point_cloud(voxel_grid, min_bound_crop, voxel_size):
    """
    Convert a 3D voxel grid back to point cloud coordinates.
    
    Args:
        voxel_grid (np.ndarray): The 3D voxel grid with marked voxels.
        min_bound_crop (np.ndarray): The minimum bound of the crop region in original coordinates.
        voxel_size (np.ndarray): The size of each voxel (in mm).
    
    Returns:
        np.ndarray: The point cloud (N, 3) representing the coordinates of the filled voxels.
    """
    # Find the indices of non-zero voxels
    filled_indices = np.argwhere(voxel_grid > 0)  # Shape (N, 3), where N is the number of filled voxels
    # Map voxel indices back to the original point cloud coordinates
    # Coordinate = min_bound_crop + (voxel indices + 0.5) * voxel_size
    point_cloud = min_bound_crop + (filled_indices + 0.5) * voxel_size

    return point_cloud



def voxel_grid_to_point_cloud_with_normals(voxel_grid: np.ndarray, 
                                           min_bound_crop: np.ndarray, 
                                           voxel_size: np.ndarray) -> np.ndarray:
    # min bound crop is the minimum point (left front down) on the mesh bouding box
    #check for shape
    if voxel_grid.ndim != 4 or voxel_grid.shape[0] != 4:
        raise ValueError(f"expected voxel_grid 形状为 (4, D, H, W)，but {voxel_grid.shape}")
    if min_bound_crop.shape != (3,):
        raise ValueError(f"expected voxel_grid min_bound_crop 形状为 (3,)，but {min_bound_crop.shape}")
    if voxel_size.shape != (3,):
        raise ValueError(f"expected voxel_grid voxel_size 形状为 (3,)，but {voxel_size.shape}")
    
    
    occupancy = voxel_grid[0]  
    
   
    filled_indices = np.argwhere(occupancy > 0)  
    
    if filled_indices.size == 0:
      
        return np.empty((0, 6), dtype=voxel_grid.dtype)
    
    # 将体素索引转换为实际坐标
    d_indices = filled_indices[:, 0]
    h_indices = filled_indices[:, 1]
    w_indices = filled_indices[:, 2]

    x_coords = min_bound_crop[0] + (d_indices + 0.5) * voxel_size[0]
    y_coords = min_bound_crop[1] + (h_indices + 0.5) * voxel_size[1]
    z_coords = min_bound_crop[2] + (w_indices + 0.5) * voxel_size[2]
    
    
    coordinates = np.stack((x_coords, y_coords, z_coords), axis=1)  
    
 
    normals = voxel_grid[1:4, d_indices, h_indices, w_indices] 
    normals = normals.T 
    
   
    norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    
    valid = norm_lengths > 0
    normals[valid[:, 0]] /= norm_lengths[valid[:, 0]]
    
    normals[~valid[:, 0]] = 0.0
    
    
    point_cloud_with_normals = np.hstack((coordinates, normals)) 
    
    return point_cloud_with_normals

def normalize_point_cloud(point_cloud, point_cloud_center, crop_size=20.0):
    """
    Normalize a point cloud by shifting its geometric center to crop_size/2 and scaling to [0, 1].

    Args:
        point_cloud (np.ndarray): Point cloud with shape (num_points, 3).
        point_cloud_center (np.ndarray): Center of the point cloud with shape (3,).
        crop_size (float): The crop size for normalization.

    Returns:
        np.ndarray: Normalized point cloud.
    """
    # Define crop center and scale
    crop_center = np.array([crop_size / 2, crop_size / 2, crop_size / 2], dtype=np.float32)
    crop_scale = np.array([crop_size, crop_size, crop_size], dtype=np.float32)
    
    # Translate point cloud to crop center and normalize
    normalized_point_cloud = (point_cloud - point_cloud_center + crop_center) / crop_scale
    
    return normalized_point_cloud


def denormalize_point_cloud(normalized_point_cloud, point_cloud_center, crop_size):
    """
    De-normalize a point cloud from normalized coordinates back to original scale and position.

    Args:
        normalized_point_cloud (np.ndarray): Normalized point cloud with shape (num_points, 3).
        crop_center (np.ndarray): Original crop center with shape (3,).
        crop_scale (np.ndarray): Crop scale with shape (3,).

    Returns:
        np.ndarray: De-normalized point cloud.
    """
    # Reverse the scale and translation
    crop_center = np.array([crop_size / 2, crop_size / 2, crop_size / 2], dtype=np.float32)
    point_cloud = normalized_point_cloud * crop_size - crop_center + point_cloud_center
    
    return point_cloud

def extract_and_rename(path):
   
    parts = path.split(os.sep)
    
    
    third_last_part = parts[-3]
    last_part = parts[-1]       
    
    
    file_name, file_extension = os.path.splitext(last_part)
    
   
    new_file_name = f"{third_last_part}+{file_name}"
    
    return new_file_name

def test(model, test_loader, voxel_size=(0.15625,0.15625,0.15625), save_path='./test_outputs', save_batches=4):
    
    model.eval() 
    test_hausdorff = 0.0
    test_fidelity = 0.0
    test_f_value = 0.0
    dpsr = DPSR(res=(128,128, 128), sig = 2)

    
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        for batch_idx,(inputs,targets,pointcloud_inform,batch_y,min_bound_crop,prompt,file_dir) in enumerate(tqdm(test_loader)):
            

            with accelerator.autocast():
                voxel_ind,refined_pos_with_normal,batch_x = model(inputs.cuda(),min_bound_crop,prompt.cuda())
            position_indicator = F.sigmoid(voxel_ind)
            position_indicator = (position_indicator>0.5).float()   
                # fidelity_val = fidelity(pred_points=refined_pos_with_normal[:,:3].cuda(),gt_points=pointcloud_inform[:,:3].cuda(),batch_x=batch_x.cuda(),batch_y=batch_y.cuda())
                # test_fidelity += fidelity_val
                
            for batch_id in torch.unique(batch_x):
                mask_gt = batch_y.cuda() == batch_id
                batch_gt = pointcloud_inform.cuda()[mask_gt,:3]
                mask = batch_x == batch_id
                batch_points = refined_pos_with_normal[mask, :3]  # 提取 [x, y, z]
                batch_normals = refined_pos_with_normal[mask, 3:]  # 提取 [nx, ny, nz]
                
              
                chamfer = chamfer_distance(batch_gt[None,:],batch_points[None,:])[0]
                fidelity_val = fidelity(batch_gt,batch_points)
                f_value,feature_pc = fidelity_ratio_and_point_cloud(batch_gt,batch_points)
                print(chamfer)
               
                test_f_value += f_value
                test_hausdorff += chamfer
                test_fidelity += fidelity_val
                
               
                update_teeth_statistics(teeth_dict, file_dir[int(batch_id.item())], chamfer,fidelity=fidelity_val,f_value=f_value)
              
                    

    
    print(teeth_dict)
    test_hausdorff /= len(test_loader.dataset)
    print(test_fidelity / len(test_loader.dataset))
    print(test_f_value / len(test_loader.dataset))
    return test_hausdorff


def load_test_data(batch_size=2, test_path='./test_data'):
    test_dataset = IOS_Datasetv5(test_path, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=8,collate_fn=test_dataset.collate_fn)
    return test_loader



def test_main():
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--test_path', type=str, default='./test_data', help='Path to test data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--save_path', type=str, default='./test_outputs', help='Path to save the test results')
    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)  
    
    model = CrownMVM(in_channels=1,out_channels=1)
 
    ckpt = torch.load(args.model_path)
    ckpt = {k[7:] if k.startswith('module.') else k: v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    model.to(accelerator.device)
   
    test_loader = load_test_data(batch_size=args.batch_size, test_path=args.test_path)
   
    test_hausdorff = test(model, test_loader, voxel_size=(0.15625,0.15625,0.15625),save_path=args.save_path)

    print(f"Test hausdorff Coefficient:{test_hausdorff:.4f}")
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    print(torch.cuda.is_available())
    seed_everything(42)
    test_main()
