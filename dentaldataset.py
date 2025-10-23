class IOS_Datasetv2(Dataset):
    def __init__(self, root_dir, is_train=True, crop_size=(2.00, 2.00, 2.00), voxel_size=(0.15625, 0.15625, 0.15625)):
        self.root_dir = Path(root_dir)
        self.crop_size = crop_size
        self.voxel_size = voxel_size
        self.subdirs = ['11', '12', '13','14', '15', '16', '17', 
 '21', '22','23', '24', '25', '26', '27', 
 '31', '32','33','34', '35', '36', '37', 
 '41', '42', '43','44', '45', '46', '47']
        self.data_paths = []
        for subdir in self.subdirs:
            subdir_path = self.root_dir / subdir
            if is_train:
                case_dir = subdir_path / 'train' 
            else:
                case_dir = subdir_path / 'test'
            for case in os.listdir(case_dir):
                abs_case = os.path.join(case_dir, case)
                crown_file = os.path.join(abs_case, 'crown.h5')
                pna_crop_file = os.path.join(abs_case, 'pna_crop.h5')
                self.data_paths.append((abs_case, crown_file, pna_crop_file))

    def crop_mesh(self, mesh, center, crop_size):
        points = np.asarray(mesh)

        positions = points[:, :3]
        normals = points[:, 3:]

        half_crop_size = 10*np.array(crop_size) / 2
        min_bound = center - half_crop_size
        max_bound = center + half_crop_size
        mask = np.all((positions >= min_bound) & (positions <= max_bound), axis=1)
        cropped_positions = positions[mask]
        cropped_normals = normals[mask]

        cropped_points = np.hstack((cropped_positions, cropped_normals))
        return cropped_points
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        dirpath, crown_file, pna_crop_file = self.data_paths[idx]
        with h5py.File(crown_file, 'r') as f:
            crown_vertices = np.array(f['vertices'])
            crown_normals = np.array(f['normals'])
            curvatures = np.array(f['curvatures'])
        with h5py.File(pna_crop_file, 'r') as f:
            pna_crop_vertices = np.array(f['vertices'])
            pna_crop_normals = np.array(f['normals'])
        crown_min_bound = crown_vertices.min(axis=0)
        crown_max_bound = crown_vertices.max(axis=0)
        crown_center = (crown_min_bound + crown_max_bound) / 2

        half_crop_size = np.array(self.crop_size) * 10 / 2
        min_bound_crop = crown_center - half_crop_size
        max_bound_crop = crown_center + half_crop_size
        point_cloud_full_inform = np.concatenate([crown_vertices,crown_normals,curvatures.reshape(-1,1)],axis=1)
        crown_voxel_grid = create_voxelwithnormal_grid(point_cloud_full_inform, min_bound_crop, max_bound_crop, self.voxel_size)
        crown_tensor = torch.from_numpy(crown_voxel_grid).float()
        pna_crop = self.crop_mesh(np.concatenate([pna_crop_vertices,pna_crop_normals],axis=1),crown_center, self.crop_size)
        pna_crop_tensor = torch.tensor(pna_crop, dtype=torch.float32)
        pna_crop_tensor = self.normalize_point_cloud(pna_crop_tensor,cropsize=self.crop_size)
        point_cloud_crown_inform = torch.tensor(point_cloud_full_inform,dtype=torch.float32)
        return pna_crop_tensor[:,:3], crown_tensor, point_cloud_crown_inform, torch.tensor(min_bound_crop,dtype=torch.float32),dirpath
    
    def collate_fn(self, batch):
        pna_crop_tensor,crown_tensor,point_cloud_crown_tensor,min_bound_crop,dirpath = zip(*batch)
        point_cloud_crown_tensor = [pc for pc in point_cloud_crown_tensor]
        combined_point_cloud = torch.cat(point_cloud_crown_tensor, dim=0)

        batch_sizes = [pc.shape[0] for pc in point_cloud_crown_tensor]
        batch_indices = torch.cat([torch.full((size,), i, dtype=torch.long) for i, size in enumerate(batch_sizes)])

        return pna_crop_tensor,torch.stack(crown_tensor),combined_point_cloud,batch_indices,torch.stack(min_bound_crop),dirpath
    
    def normalize_point_cloud(self, point_cloud,cropsize):
        if not isinstance(point_cloud, torch.Tensor):
            point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
        if point_cloud.shape[1] != 6:
            raise ValueError("Point cloud should have shape (num_points, 6)")
        
        positions = point_cloud[:, :3]
        normals = point_cloud[:, 3:]

        point_cloud_center = (torch.min(positions, dim=0)[0] + torch.max(positions, dim=0)[0]) / 2
        crop_center = 10*torch.tensor(cropsize, dtype=torch.float32) / 2
        crop_scale = 10*torch.tensor(cropsize, dtype=torch.float32)

        normalized_positions = (positions - point_cloud_center + crop_center) / crop_scale
        normalized_positions = (normalized_positions - 0.5) * 2

        normalized_point_cloud = torch.cat((normalized_positions, normals), dim=1)

        return normalized_point_cloud
    
