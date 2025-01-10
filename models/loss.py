from torch_geometric.nn import knn
import torch
from torch import nn
import torch.nn.functional as F


def curvature_and_margine_penalty_loss(pred_points_with_normal,gt_points_with_normal_and_curv,batch_x,batch_y):
    pred_points,pred_normals = pred_points_with_normal[:,:3],pred_points_with_normal[:,3:6]
    #pred_normals = F.normalize(pred_normals,dim=1)
    gt_points,gt_normals,curvatures,whether_margin = gt_points_with_normal_and_curv[:,:3],gt_points_with_normal_and_curv[:,3:6],gt_points_with_normal_and_curv[:,6],gt_points_with_normal_and_curv[:,-1]
    assign_index_x2y = knn(pred_points,gt_points,1,batch_x,batch_y) # for each element y, the nearst point in x
    assign_index_y2x = knn(gt_points,pred_points,1,batch_y,batch_x) # for each element x, the nearst point in y
    gt_to_pred_dist = torch.sum((gt_points - pred_points[assign_index_x2y[1]])**2, dim=1)  # (M,)
    gt_to_pred_weighted = gt_to_pred_dist * (torch.exp(torch.abs(curvatures )) + whether_margin)
    pred_to_gt_dist = torch.sum((pred_points - gt_points[assign_index_y2x[1]])**2, dim=1)  # (N,)
    pred_to_gt_weighted = pred_to_gt_dist * (torch.exp(torch.abs(curvatures[assign_index_y2x[1]])) +whether_margin[assign_index_y2x[1]])
    normal_loss = F.mse_loss(pred_normals, gt_normals[assign_index_y2x[1]])
    return torch.mean(gt_to_pred_weighted)+torch.mean(pred_to_gt_weighted),normal_loss


if __name__ == "__main__":
    pred_point = torch.randn(8421,6).cuda()
    gt_point = torch.randn(7625,7).cuda()
    '''
    In order to support abitrary number of point in input and output, and perform knn using torch_geometric, the point cloud in our framework is represented in
    [N,C] with N=N1+N2+....+Nb , where b is the batch size, and a corresponding batch index array is followed to tell which batch a certain point belongs to.
    (B,N,C is not applicable since N is different)
    '''
    batch_x = torch.cat([torch.zeros(3581, dtype=torch.long), torch.ones(8421-3581, dtype=torch.long)]).cuda()
    batch_y = torch.cat([torch.zeros(3814, dtype=torch.long), torch.ones(7625-3814, dtype=torch.long)]).cuda()
    print(curvature_and_margine_penalty_loss(pred_point,gt_point,batch_x,batch_y))
