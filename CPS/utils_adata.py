import numpy as np
import torch
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import add_self_loops, dense_to_sparse, to_undirected
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
from collections import namedtuple
import scipy.sparse
from scipy.spatial import cKDTree


BatchData = namedtuple('BatchData', ['x', 'y', 'pos'])
class AdataDataset(Dataset):
    def __init__(self, x, y, pos, edge_index, num_nodes, num_genes):
        self.x = x
        self.y = y
        self.pos = pos
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.num_genes = num_genes
        
    def __len__(self):
        return self.num_nodes
    
    def __getitem__(self, idx):
        return BatchData(
            x=self.x[idx], 
            y=self.y[idx], 
            pos=self.pos[idx]
        )


# construct spatial graph using location
class SpatialGraphBuilder:
    def __init__(self, args):
        self.max_neighbors = args.max_neighbors
        self.radius = args.radius
        self.self_loops = args.self_loops
        self.flow = args.flow
    
    # randomly mask the spots
    def spots_perturb(self, adata, mask_ratio, method='rknn'):
        """
        mask test spots, return train and test data
        
        Args:
            adata: AnnData, contains spatial coordinates and hvg_features
            mask_ratio: mask ratio of spots (0.0-1.0)
            method: construct method of graphs, 'knn' or 'rknn'
            
        Returns:
            train_data: train data with graph
            test_data: test data without graph
            train_indices: train indices of spots
            test_indices: test indices of spots
        """
        n_spots = adata.n_obs
        n_test = int(n_spots * mask_ratio)
        
        # randomly select test spots
        all_indices = np.arange(n_spots)
        test_indices = np.random.choice(all_indices, size=n_test, replace=False)
        train_indices = np.setdiff1d(all_indices, test_indices)
        
        # get position and features (and counts)
        all_pos = torch.FloatTensor(adata.obsm['spatial'])
        min_val = all_pos.min(dim=0).values
        max_val = all_pos.max(dim=0).values
        center = (min_val + max_val) / 2
        span = max_val - min_val
        scale = span.max() / 2
        all_pos_norm = (all_pos - center) / scale
        all_features = torch.tensor(adata.obsm['hvg_features'], dtype=torch.float)
        all_counts = torch.tensor(adata.obsm['hvg_counts'], dtype=torch.float)
        
        # train data
        train_pos = all_pos_norm[train_indices]
        train_features = all_features[train_indices]
        train_counts = all_counts[train_indices]
        
        # build graph
        if method == 'knn':
            train_edge_index = knn_graph(train_pos, k=self.max_neighbors, flow=self.flow)
            train_edge_index = to_undirected(train_edge_index)
        elif method == 'rknn':
            train_edge_index = radius_graph(train_pos, r=self.radius,
                                           max_num_neighbors=self.max_neighbors, flow=self.flow)
            train_edge_index = to_undirected(train_edge_index)
        if self.self_loops:
            train_edge_index, _ = add_self_loops(train_edge_index, num_nodes=len(train_indices))
        
        # train dataset
        train_data = Data(x=train_features,
                          y=train_counts,
                         pos=train_pos,
                         edge_index=train_edge_index,
                         num_nodes=len(train_indices),
                         num_genes=train_features.shape[1])
        
        # test data
        test_pos = all_pos_norm[test_indices]
        test_features = all_features[test_indices]
        test_counts = all_counts[test_indices]
        
        # test dataset
        test_data = Data(x=test_features,
                         y=test_counts,
                        pos=test_pos,
                        edge_index=None,  # without graph
                        num_nodes=len(test_indices),
                        num_genes=test_features.shape[1])
        
        return train_data, test_data, train_indices, test_indices
    
    # randomly mask the genes in spots
    def genes_perturb(self, adata, mask_ratio, method='rknn', mask_value=0.0):
        """
        Randomly mask a certain proportion of genes in each spot, return training and test data
        
        Args:
            adata: AnnData object containing spatial coordinates and hvg_features
            mask_ratio: Proportion of genes to mask in each spot (0.0-1.0)
            method: Graph construction method, 'knn' or 'rknn'
            mask_value: Value to replace masked genes (default 0.0)
        Returns:
            train_data: Training data (partially masked genes, with complete graph structure)
            test_data: Test data (complete genes) to compute metrics
            mask_pattern: Noolean matrix of shape (n_spots, n_genes)
        """
        
        n_spots = adata.n_obs
        n_genes = adata.obsm['hvg_features'].shape[1]
        
        # Get positions and features of all spots
        all_pos = torch.FloatTensor(adata.obsm['spatial'])
        min_val = all_pos.min(dim=0).values
        max_val = all_pos.max(dim=0).values
        center = (min_val + max_val) / 2
        span = max_val - min_val
        scale = span.max() / 2
        all_pos_norm = (all_pos - center) / scale
        
        feat_data = adata.obsm['hvg_features']
        if scipy.sparse.issparse(feat_data):
            feat_data = feat_data.toarray()
        all_features = torch.tensor(feat_data, dtype=torch.float)
        count_data = adata.obsm['hvg_counts']
        if scipy.sparse.issparse(count_data):
            count_data = count_data.toarray()
        all_counts = torch.tensor(count_data, dtype=torch.float)
        
        # Create random mask pattern: each spot independently selects genes
        mask_pattern = torch.zeros((n_spots, n_genes), dtype=torch.bool)
        n_mask_per_spot = int(n_genes * mask_ratio)
        
        for i in range(n_spots):
            # Randomly select genes to mask for each spot
            gene_indices = np.random.choice(n_genes, size=n_mask_per_spot, replace=False)
            mask_pattern[i, gene_indices] = True
        
        # Training data: partially masked genes
        train_features = all_features.clone()
        train_features[mask_pattern] = mask_value
        train_counts = all_counts.clone()
        train_counts[mask_pattern] = mask_value
        
        # Test data: use complete features
        test_features = all_features.clone()
        test_counts = all_counts.clone()
        
        
        # Build graph for training data (using all spots)
        if method == 'knn':
            train_edge_index = knn_graph(all_pos, k=self.max_neighbors, flow=self.flow)
            train_edge_index = to_undirected(train_edge_index)
        elif method == 'rknn':
            train_edge_index = radius_graph(all_pos, r=self.radius,
                                           max_num_neighbors=self.max_neighbors, flow=self.flow)
            train_edge_index = to_undirected(train_edge_index)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if self.self_loops:
            train_edge_index, _ = add_self_loops(train_edge_index, num_nodes=n_spots)
        
        # Create training data (with graph structure)
        train_data = Data(x=train_features, 
                          y=train_counts, 
                          pos=all_pos_norm, 
                          edge_index=train_edge_index, 
                          num_nodes=n_spots, 
                          num_genes=n_genes)
        
        # Create test data (without graph structure)
        test_data = Data(x=test_features,
                         y=test_counts,
                         pos=all_pos_norm,
                         edge_index=train_edge_index, # not used in test
                         num_nodes=n_spots,
                         num_genes=n_genes)
        
        return train_data, test_data, mask_pattern
    
    # simple single graph construct
    def build_single_graph(self, adata, method='rknn'):
        
        pos = torch.FloatTensor(adata.obsm['spatial'])
        min_val = pos.min(dim=0).values
        max_val = pos.max(dim=0).values
        center = (min_val + max_val) / 2
        span = max_val - min_val
        scale = span.max() / 2
        pos_norm = (pos - center) / scale
        
        if method == 'knn':
            edge_index = knn_graph(pos, k=self.max_neighbors, flow=self.flow)
            edge_index = to_undirected(edge_index)
        elif method == 'rknn':
            edge_index = radius_graph(pos, r=self.radius, max_num_neighbors=self.max_neighbors, flow=self.flow)
            edge_index = to_undirected(edge_index)
        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=adata.n_obs)
        
        data = Data(x=torch.tensor(adata.obsm['hvg_features'], dtype=torch.float),
                    y=torch.tensor(adata.obsm['hvg_counts'], dtype=torch.float),
                    pos=pos_norm,
                    edge_index=edge_index,
                    num_nodes=adata.n_obs, 
                    num_genes=adata.obsm['hvg_features'].shape[1])
    
        return data
    # for Visium HD et.al, num of spots > 1e6
    def build_large_graph(self, adata, method='rknn'):
        
        pos = torch.FloatTensor(adata.obsm['spatial'])
        min_val = pos.min(dim=0).values
        max_val = pos.max(dim=0).values
        center = (min_val + max_val) / 2
        span = max_val - min_val
        scale = span.max() / 2
        pos_norm = (pos - center) / scale
        
        if method == 'knn':
            edge_index = knn_graph(pos, k=self.max_neighbors, flow=self.flow)
            edge_index = to_undirected(edge_index)
        elif method == 'rknn':
            edge_index = radius_graph(pos, r=self.radius, max_num_neighbors=self.max_neighbors, flow=self.flow)
            edge_index = to_undirected(edge_index)
        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=adata.n_obs)
        
        data = AdataDataset(x=torch.tensor(adata.obsm['hvg_features'], dtype=torch.float),
                            y=torch.tensor(adata.obsm['hvg_counts'], dtype=torch.float),
                            pos=pos_norm,
                            edge_index=edge_index,
                            num_nodes=adata.n_obs, 
                            num_genes=adata.obsm['hvg_features'].shape[1])    
        
        return data




def generate_sr_coords(adata, upscale_factor=4, margin=0):
    """
    根据现有 adata 的坐标范围，生成更高分辨率的网格坐标。
    并且通过距离过滤，只保留组织覆盖区域内的点（去除背景）。
    
    Args:
        adata: 原始 AnnData 对象，包含 .obsm['spatial']
        upscale_factor: 放大倍数 (例如 2 表示宽高各 x2，总点数 x4)
        margin: 允许向外延伸的距离 (通常设为 0 或很小的值)
        
    Returns:
        new_coords (np.array): (N_new, 2) 新的高清坐标矩阵
    """
    # 1. 获取原始坐标和范围
    coords = adata.obsm['spatial']
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    
    # 2. 计算原始的分辨率 (大致的 Spot 间距)
    # 通过计算最近邻距离的平均值来估算
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2) # k=2 因为第1个是自己
    avg_step = np.mean(dists[:, 1]) # 原始的步长
    
    # 3. 生成新的高分辨率网格
    new_step = avg_step / upscale_factor
    
    # 使用 meshgrid 生成网格
    x_range = np.arange(min_x - margin, max_x + margin, new_step)
    y_range = np.arange(min_y - margin, max_y + margin, new_step)
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    
    # 拉平为 (N, 2)
    grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    print(f"原始点数: {len(coords)}")
    print(f"生成的全网格点数: {len(grid_coords)}")
    
    # 4. 关键步骤：背景过滤 (Masking Background)
    # 我们只保留那些“离原始数据点比较近”的网格点
    # 阈值通常设为原始步长的 0.6-0.8 倍，保证空隙被填满，但外部背景被切除
    dist_threshold = avg_step * 0.8 
    
    # 查询每个网格点离它最近的原始点的距离
    dists_to_real, _ = tree.query(grid_coords, k=1)
    
    # 只保留距离小于阈值的点
    mask = dists_to_real < dist_threshold
    final_coords = grid_coords[mask]
    
    print(f"过滤背景后的 SR 点数: {len(final_coords)}")
    
    return final_coords


from scipy.interpolate import griddata

def generate_sr_library_size(adata, sr_coords, mode='mean', k=5):
    """
    为超分辨率坐标生成对应的 Library Size。
    
    Args:
        adata: 原始 AnnData
        sr_coords: (N, 2) 新生成的超分辨率坐标
        mode: 
            - 'mean': 使用全数据均值 (去除测序深度偏好，图最干净)
            - 'median': 使用全数据中位数 (推荐，抗异常值)
            - 'nearest': 最近邻 (会有马赛克块状感，不推荐)
            - 'linear': 线性插值 (平滑保留细胞密度差异)
        k: 只有在某些自定义插值逻辑下才用，这里主要依赖 griddata
    
    Returns:
        sr_lib_size: (N, 1) Torch Tensor
    """
    # 1. 获取真实的 Library Size
    # 优先从 obs 取，如果没有就现算
    if 'total_counts' in adata.obs.columns:
        real_lib_size = adata.obs['total_counts'].values
    else:
        real_lib_size = np.array(adata.X.sum(axis=1)).flatten()
        
    real_coords = adata.obsm['spatial']
    
    print(f"Generating Library Size with mode: {mode}")
    
    if mode == 'mean':
        # 策略 A: 均值填充 (所有点一样)
        avg_val = np.mean(real_lib_size)
        sr_lib_size = np.full(len(sr_coords), avg_val)
        
    elif mode == 'median':
        # 策略 B: 中位数填充 (所有点一样，更鲁棒)
        med_val = np.median(real_lib_size)
        sr_lib_size = np.full(len(sr_coords), med_val)
        
    elif mode == 'linear' or mode == 'cubic':
        # 策略 C: 空间插值 (保留密度差异)
        # 注意：griddata 在凸包之外的点会产生 nan，需要 fill_value
        avg_val = np.mean(real_lib_size)
        sr_lib_size = griddata(
            points=real_coords, 
            values=real_lib_size, 
            xi=sr_coords, 
            method=mode, 
            fill_value=avg_val # 边缘外的点用均值填充
        )
        # 确保没有负数 (虽然插值通常不会产生负数)
        sr_lib_size = np.clip(sr_lib_size, a_min=1.0, a_max=None)
        
    elif mode == 'nearest':
        # 策略 D: 最近邻
        sr_lib_size = griddata(
            points=real_coords, 
            values=real_lib_size, 
            xi=sr_coords, 
            method='nearest'
        )
        
    else:
        raise ValueError("Unknown mode")

    # 转为 Tensor 格式 (N, 1) 适配模型输入
    sr_lib_size = torch.FloatTensor(sr_lib_size).unsqueeze(1)
    
    return sr_lib_size