import os
import open3d as o3d
import torch
import time
import multiprocessing
import numpy as np
from tqdm import tqdm

#from easy_mesh_vtk import Easy_Mesh
import h5py
import os
from src.dpsr import DPSR

def process_all_h5_files(base_dir, resolution=128, sig=0):
    # 初始化 DPSR
    dpsr = DPSR(res=(resolution, resolution, resolution), sig=sig)
    
    # 遍历目录
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'crown.h5':
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}...")
                
                # 读取 crown.h5 文件
                with h5py.File(file_path, 'r') as f:
                    vertices = f['vertices'][:]
                    normals = f['normals'][:]
                
                # 计算 psr_gt
                vertices_tensor = torch.from_numpy(vertices.astype(np.float32))[None]  # 批量维度
                normals_tensor = torch.from_numpy(normals.astype(np.float32))[None]   # 批量维度
                
                psr_gt = dpsr(vertices_tensor, normals_tensor).squeeze().cpu().numpy().astype(np.float16)
                exit()
                # 保存 psr_gt 为 .npz 文件
                out_path = os.path.join(root, 'psr.npz')
                np.savez(out_path, psr=psr_gt)
                print(f"Saved PSR to {out_path}")

# 主函数
if __name__ == "__main__":
    base_dir = "simplified_processed_h5_data"  # 根目录路径
    process_all_h5_files(base_dir, resolution=128, sig=0)












