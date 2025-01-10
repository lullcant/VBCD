from torch_geometric.nn import knn
import torch
from torch import nn
import torch.nn.functional as F

# def fidelity(pred_points,gt_points,batch_x,batch_y):
#     assign_index_x2y = knn(pred_points,gt_points,1,batch_x,batch_y)
#     gt_to_pred_dist = torch.sqrt(torch.sum((gt_points - pred_points[assign_index_x2y[1]])**2, dim=1)) 
#     return torch.mean(gt_to_pred_dist)
def fidelity(pred_points,gt_points):
    assign_index_x2y = knn(pred_points,gt_points,1,batch_size=1)
    gt_to_pred_dist = torch.sqrt(torch.sum((gt_points - pred_points[assign_index_x2y[1]])**2, dim=1)) 
    return torch.mean(gt_to_pred_dist)


def fidelity_ratio_and_point_cloud(
    x: torch.Tensor,       # 需要找最近邻的点集 (e.g. GT 点云)
    y: torch.Tensor,       # 用来做最近邻搜索的点集 (e.g. Pred 点云)
    fraction: float = 0.03 # 距离阈值 (单位 mm, 或你的坐标系单位)
):
    """
    1. 对 x 中每个点，求其在 y 中的最近邻距离。
    2. 若距离 <= threshold 则记为 inlier，否则记为 outlier。
    3. 返回 (ratio, new_point_cloud):
       - ratio = (inlier 数量) / (outlier 数量)
       - new_point_cloud: (N, 4) 张量
         [ x_coord, y_coord, z_coord, dist_to_nearest ]
    """
    min_xyz = y.min(dim=0)[0]
    max_xyz = y.max(dim=0)[0]
    diag_length = torch.sqrt(torch.sum((max_xyz - min_xyz) ** 2))
    threshold = fraction * diag_length
    # 使用 knn 找最近邻（k=1）
    # 注意 knn(x, y, k=1) 或 knn(y, x, k=1) 的查询/引用点顺序
    # 根据 PyTorch Geometric 版本不同，若结果不对可对调
    assign_index_x2y = knn(y, x, k=1, batch_size=1)
    
    # 计算 x 中各点与最近邻 y[*] 的欧几里得距离
    nearest_dists = torch.sqrt(torch.sum((x - y[assign_index_x2y[1]])**2, dim=1))
    
    # 根据 threshold 分为 inliers 与 outliers
    inliers_mask = (nearest_dists <= threshold)  # True/False
    outliers_mask = ~inliers_mask

    num_inliers = inliers_mask.sum().item()   # <= threshold 的点数
    num_outliers = outliers_mask.sum().item() # > threshold 的点数

    # 计算比值: (inlier 数量 / outlier 数量)
    if num_outliers == 0:
        ratio = 1
    else:
        ratio = num_inliers / (num_outliers+num_inliers)
    
    # 生成新的 (N, 4) 点云: 前三列是 x 的原始坐标，最后一列是与最近邻的距离
    new_point_cloud = torch.cat([x, nearest_dists.unsqueeze(1)], dim=1)
    
    return ratio, new_point_cloud