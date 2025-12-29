import numpy as np
import torch
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import add_self_loops, dense_to_sparse, to_undirected
from torch_geometric.data import Data, Batch

# constract spatial graph using location
class SpatialGraphBuilder:
    def __init__(self, args):
        self.max_neighbors = args.max_neighbors
        self.radius = args.radius
        self.self_loops = args.self_loops
        self.flow = args.flow
    
    # randomly mask the spots
    def spots_perturb(self, adata, mask_ratio, method='rknn', seed=None):
        """
        随机mask一定比例的spots，返回训练数据和测试数据
        
        Args:
            adata: AnnData对象，包含spatial坐标和hvg_features
            mask_ratio: 被mask的spots比例 (0.0-1.0)
            method: 构图方法，'knn'或'rknn'
            seed: 随机种子，用于可重复性
            
        Returns:
            train_data: 训练数据（包含完整的图结构）
            test_data: 测试数据（只包含位置和特征，无图结构）
            train_indices: 训练spots的索引
            test_indices: 测试spots的索引
        """
        
        n_spots = adata.n_obs
        n_test = int(n_spots * mask_ratio)
        
        # 随机选择测试spots
        all_indices = np.arange(n_spots)
        test_indices = np.random.choice(all_indices, size=n_test, replace=False)
        train_indices = np.setdiff1d(all_indices, test_indices)
        
        # 获取所有spots的位置和特征
        all_pos = torch.FloatTensor(adata.obsm['spatial'])
        min_val = all_pos.min(dim=0).values
        max_val = all_pos.max(dim=0).values
        center = (min_val + max_val) / 2
        span = max_val - min_val
        scale = span.max() / 2
        all_pos_norm = (all_pos - center) / scale
        all_features = torch.tensor(adata.obsm['hvg_features'], dtype=torch.float)
        all_counts = torch.tensor(adata.obsm['hvg_counts'], dtype=torch.float)
        
        # 训练数据：使用训练spots构建图
        train_pos = all_pos_norm[train_indices]
        train_features = all_features[train_indices]
        train_counts = all_counts[train_indices]
        
        # 为训练数据构建图
        if method == 'knn':
            train_edge_index = knn_graph(train_pos, k=self.max_neighbors, flow=self.flow)
            train_edge_index = to_undirected(train_edge_index)
        elif method == 'rknn':
            train_edge_index = radius_graph(train_pos, r=self.radius,
                                           max_num_neighbors=self.max_neighbors, flow=self.flow)
            train_edge_index = to_undirected(train_edge_index)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if self.self_loops:
            train_edge_index, _ = add_self_loops(train_edge_index, num_nodes=len(train_indices))
        
        # 创建训练数据（包含图结构）
        train_data = Data(x=train_features,
                          y=train_counts,
                         pos=train_pos,
                         edge_index=train_edge_index,
                         num_nodes=len(train_indices),
                         num_genes=train_features.shape[1])
        
        # 测试数据：只包含位置和特征，无图结构
        test_pos = all_pos_norm[test_indices]
        test_features = all_features[test_indices]
        test_counts = all_counts[test_indices]
        
        # 创建测试数据（无图结构）
        test_data = Data(x=test_features,
                         y=test_counts,
                        pos=test_pos,
                        edge_index=None,  # 测试数据不需要图结构
                        num_nodes=len(test_indices),
                        num_genes=test_features.shape[1])
        
        return train_data, test_data, train_indices, test_indices
    
    # randomly mask the genes in spots
    def genes_perturb(self, adata, mask_ratio, method='rknn', seed=None,
                     train_mask_only=True, mask_value=0.0):
        """
        随机mask每个spot中一定比例的基因，返回训练数据和测试数据
        
        Args:
            adata: AnnData对象，包含spatial坐标和hvg_features
            mask_ratio: 每个spot中被mask的基因比例 (0.0-1.0)
            method: 构图方法，'knn'或'rknn'
            seed: 随机种子，用于可重复性
            train_mask_only: 如果True，只mask训练数据的基因；如果False，训练和测试都mask
            mask_value: 用于替换被mask基因的值（默认0.0）
            
        Returns:
            train_data: 训练数据（部分基因被mask，包含完整的图结构）
            test_data: 测试数据（如果train_mask_only=True则使用完整特征，否则部分基因被mask）
            mask_pattern: 掩码模式，形状为(n_spots, n_genes)的布尔矩阵
        """
        
        n_spots = adata.n_obs
        n_genes = adata.obsm['hvg_features'].shape[1]
        
        # 获取所有spots的位置和特征
        # all_pos = torch.FloatTensor(adata.obsm['spatial']) # !归一化
        all_pos = torch.FloatTensor(adata.obsm['spatial'])
        min_val = all_pos.min(dim=0).values
        max_val = all_pos.max(dim=0).values
        center = (min_val + max_val) / 2
        span = max_val - min_val
        scale = span.max() / 2
        all_pos_norm = (all_pos - center) / scale
        
        all_features = torch.tensor(adata.obsm['hvg_features'], dtype=torch.float)
        
        # 创建随机掩码模式：每个spot独立随机选择基因
        mask_pattern = torch.zeros((n_spots, n_genes), dtype=torch.bool)
        n_mask_per_spot = int(n_genes * mask_ratio)
        
        for i in range(n_spots):
            # 为每个spot随机选择要mask的基因
            gene_indices = np.random.choice(n_genes, size=n_mask_per_spot, replace=False)
            mask_pattern[i, gene_indices] = True
        
        # 创建训练数据的特征（应用掩码）
        if train_mask_only:
            # 训练数据：部分基因被mask
            train_features = all_features.clone()
            train_features[mask_pattern] = mask_value
            
            # 测试数据：使用完整特征
            test_features = all_features.clone()
        else:
            # 训练和测试数据都使用相同的mask模式
            train_features = all_features.clone()
            train_features[mask_pattern] = mask_value
            
            # 为测试数据创建不同的mask模式
            test_mask_pattern = torch.zeros((n_spots, n_genes), dtype=torch.bool)
            for i in range(n_spots):
                gene_indices = np.random.choice(n_genes, size=n_mask_per_spot, replace=False)
                test_mask_pattern[i, gene_indices] = True
            
            test_features = all_features.clone()
            test_features[test_mask_pattern] = mask_value
        
        # 为训练数据构建图（使用所有spots）
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
        
        # 创建训练数据（包含图结构）
        train_data = Data(x=train_features,
                         pos=all_pos_norm,
                         edge_index=train_edge_index,
                         num_nodes=n_spots,
                         num_genes=n_genes)
        
        # 创建测试数据（无图结构）
        test_data = Data(x=test_features,
                        pos=all_pos_norm,
                        edge_index=None,  # 测试数据不需要图结构
                        num_nodes=n_spots,
                        num_genes=n_genes)
        
        return train_data, test_data, mask_pattern
    
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

    def build_multi_graph(self, adata_list, method='rknn'):
        pass
        

# preproduce the multi-scale features using ssgconv (APPNP)
class MultiScaleNicheProcessor:
    def __init__(self, k_list, x, edge_list):
        self.k_list = k_list
        
    def compute(self):
        pass

# batch construct for training
class BatchCollater:
    def __init__(self, follow_batch=None, exclude_keys=None):
        self.follow_batch = follow_batch or []
        self.exclude_keys = exclude_keys or []
    
    def __call__(self, batch):
        if len(batch) == 1:
            return batch[0]
        
        return Batch.from_data_list(batch, 
                                   follow_batch=self.follow_batch,
                                   exclude_keys=self.exclude_keys)
        
        