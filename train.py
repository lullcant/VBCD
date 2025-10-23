import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from accelerate import Accelerator
from models.crownmvm import CrownMVM,volume_to_point_cloud_tensor
from models.loss import curvature_penalty_loss
import os
import random
import numpy as np
import torch.nn.functional as F
import re
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_farthest_points
import pyvista as pv
from mydataset.Dentaldataset import *
from accelerate import DataLoaderConfiguration,DistributedDataParallelKwargs
focal_loss = True
curvature_weight = 2
curvature_weighted_bce = False
dataloader_config = DataLoaderConfiguration(split_batches=True)
kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(dataloader_config=dataloader_config,kwargs_handlers=[kwargs])

def setup_logging(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Training started.")

def cycle(dl):
    while True:
        for data in dl:
            yield data

def train(model, train_loader, val_loader,args,log_file):
    model.to(accelerator.device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_steps)
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    train_loader = cycle(train_loader)
    
    
    
    initial_step = 0
    step = initial_step
    best_val_dice = 0.0
    with tqdm(total=args.num_steps, desc="Training", unit="step") as pbar:
        while step < args.num_steps + initial_step:
            model.train()
            total_loss = 0.0
            for _ in range(args.accumulation_steps):
                inputs,targets,pointcloud_inform,batch_y,min_bound_crop,_ = next(train_loader)
                if curvature_weighted_bce:
                    curvatures = targets[:,-1,:,:,:]
                    non_zero_mask = curvatures != 0
                    curvatures_weighted = torch.where(non_zero_mask, 1+curvatures, curvatures)
                    criterition = nn.BCEWithLogitsLoss(weight=torch.exp(curvatures_weighted.unsqueeze(1)))
                else:
                    criterition = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0))
                with accelerator.autocast():
                    voxel_ind,voxel_normal,refined_pos_with_normal,batch_x = model(inputs,min_bound_crop)
                    if step % 2500 ==0  : logging.info(f'The front points has {refined_pos_with_normal.shape[0]} points')
                    bce_loss = criterition(voxel_ind,targets[:,:1,:,:,:])
                    cpl,normal_loss = curvature_penalty_loss(refined_pos_with_normal,pointcloud_inform,batch_x=batch_x,batch_y=batch_y) if step>100000 else (0,0)
                    refine_loss = 0.1*cpl + normal_loss
                    loss = bce_loss + F.mse_loss(voxel_normal,targets[:,1:4,:,:,:]) + refine_loss 
                    loss = loss/ args.accumulation_steps
                    total_loss += loss.item()
                accelerator.backward(loss)
           
            pbar.set_description(f'loss: {total_loss:.4f}')
            accelerator.wait_for_everyone()
            accelerator.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            accelerator.wait_for_everyone()    

            step += 1
      
            
            if accelerator.is_main_process:
                if step % args.validation_interval == 0 or step == args.num_steps:
                    val_dice= validate(model, val_loader, step=step,save_path=args.save_path)
                    logging.info(f"Step [{step}/{args.num_steps}], Validation hausdorff: {val_dice:.4f}")
                    
                    if step % args.validation_interval == 0:
                        valmodel = accelerator.unwrap_model(model)
                        step_model_path = os.path.join('checkpoints',f"{args.save_path.replace('.pth', f'_step_{step}.pth')}")
                        torch.save(valmodel.state_dict(), step_model_path)
                        logging.info(f"Saved model at step {step}")
                    
                    if val_dice < best_val_dice:
                        best_val_dice = val_dice
                        valmodel = accelerator.unwrap_model(model)
                        best_model_path = os.path.join('checkpoints',f"{args.save_path.replace('.pth', '_best.pth')}")
                        torch.save(valmodel.state_dict(), best_model_path)
                        print(f"Saved best model at step {step}")
            pbar.update(1)
            if step % args.log_interval == 0:
                logging.info(f"Step {step} - BCE: {bce_loss:.4f} - Normal: {normal_loss:.4f} - Chamfer {cpl:.4f}")

def dice_coefficient(tensor1, tensor2, epsilon=1e-6):
    assert tensor1.shape == tensor2.shape, "两个输入张量的形状必须相同"
    tensor1 = tensor1.float()
    tensor2 = tensor2.float()

    intersection = (tensor1 * tensor2).sum(dim=(2, 3, 4))
    volumes_sum = tensor1.sum(dim=(2, 3, 4)) + tensor2.sum(dim=(2, 3, 4))

    dice = (2.0 * intersection + epsilon) / (volumes_sum + epsilon)
    return dice.mean()

def validate(model, val_loader, step,save_path='./chamfer_validation_outputs'):
    model.eval()
    val_hausdorff = 0.0
    with torch.no_grad():
         
        for batch_idx,(inputs,targets,pointcloud_inform,batch_y,min_bound_crop,file_dir) in enumerate(val_loader):
           
            with accelerator.autocast():
                voxel_ind,voxel_normal,refined_pos_with_normal,batch_x = model(inputs,min_bound_crop)   
            position_indicator = F.sigmoid(voxel_ind)
            position_indicator = (position_indicator>0.5).float()

            outputs_pc = volume_to_point_cloud(volume=position_indicator,voxel_size=(0.15625,0.15625,0.15625),origin=min_bound_crop.cpu())
            hausdorff = dice_coefficient(position_indicator,targets[:,:1,:,:,:]).item()
            val_hausdorff += hausdorff
            if batch_idx < 2:
                targets_np = targets.cpu().numpy()
                for i in range(len(outputs_pc)):
                    mask = (batch_y == i)
                    gtpoints = pointcloud_inform[mask][:,:3].cpu().numpy()
                    point_cloud_gt = pv.PolyData(gtpoints)
                    point_cloud = pv.PolyData(outputs_pc[i])
                    output_filename = os.path.join(save_path[:-4], f"output_batch{batch_idx+1}_sample{i+1}_step{step}.ply")
                    gt_filename = os.path.join(save_path[:-4], f"output_batch{batch_idx+1}_gt{i+1}_step{step}.ply")
                    point_cloud.save(output_filename)
                    point_cloud_gt.save(gt_filename)
                    logging.info(f"Saved: {output_filename}")
                    logging.info(f"Saved: {gt_filename}")
                
    val_hausdorff/=len(val_loader)
  
    return val_hausdorff

def load_data(batch_size=4, train_path='./train_data'):
    train_dataset = IOS_Datasetv2(train_path)
    val_dataset = IOS_Datasetv2(train_path,is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=8, shuffle=True,collate_fn=train_dataset.collate_fn)
    logging.info(f"Length of train dataloader {len(train_loader)}")
    val_loader = DataLoader(val_dataset, batch_size=2,num_workers=8, shuffle=False,collate_fn=train_dataset.collate_fn)
    logging.info(f"Length of val dataloader {len(val_loader)}")
    return train_loader, val_loader

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_steps', type=int, default=1000, help='Total number of steps for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Number of steps for gradient accumulation')
    parser.add_argument('--save_path', type=str, default='./unet3d_model.pth', help='Path to save the trained model')
    parser.add_argument('--train_path', type=str, default='./train_data', help='Path to training data')
    parser.add_argument('--val_path', type=str, default='./val_data', help='Path to validation data')
    parser.add_argument('--validation_interval',type=int,default=4000,help='interval steps to validate')
    parser.add_argument('--continue_ckpt_dir',type=str,required=False,help='whether to use exist ckpt')
    parser.add_argument('--log_interval', type=int, default=50, help='Interval steps to log training information')
    args = parser.parse_args()
    model = CrownMVM(in_channels=1,out_channels=4)
    if args.continue_ckpt_dir:
        ckpt = torch.load(args.continue_ckpt_dir)
        ckpt = {k[7:] if k.startswith('module.') else k: v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        print('load checkpoint complete')
    if not os.path.exists(args.save_path[:-4]):
        os.makedirs(args.save_path[:-4])
    log_file = os.path.join(args.save_path[:-4],'training.log')
    setup_logging(log_file)
    train_loader, val_loader = load_data(batch_size=args.batch_size, train_path=args.train_path)
    train(model, train_loader, val_loader, args,log_file=log_file)
if __name__== "__main__":
    seed_everything(42)
    main()
