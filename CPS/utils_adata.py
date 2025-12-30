import numpy as np
import torch
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import add_self_loops, dense_to_sparse, to_undirected
from torch_geometric.data import Data, Batch

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
    def genes_perturb(self, adata, mask_ratio, method='rknn', train_mask_only=True, mask_value=0.0):
        """
        Randomly mask a certain proportion of genes in each spot, return training and test data
        
        Args:
            adata: AnnData object containing spatial coordinates and hvg_features
            mask_ratio: Proportion of genes to mask in each spot (0.0-1.0)
            method: Graph construction method, 'knn' or 'rknn'
            seed: Random seed for reproducibility
            train_mask_only: If True, only mask genes in training data; if False, mask both training and test data
            mask_value: Value to replace masked genes (default 0.0)
            
        Returns:
            train_data: Training data (partially masked genes, with complete graph structure)
            test_data: Test data (if train_mask_only=True, use complete features; otherwise partially masked)
            mask_pattern: Mask pattern, boolean matrix of shape (n_spots, n_genes)
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
        
        all_features = torch.tensor(adata.obsm['hvg_features'], dtype=torch.float)
        
        # Create random mask pattern: each spot independently selects genes
        mask_pattern = torch.zeros((n_spots, n_genes), dtype=torch.bool)
        n_mask_per_spot = int(n_genes * mask_ratio)
        
        for i in range(n_spots):
            # Randomly select genes to mask for each spot
            gene_indices = np.random.choice(n_genes, size=n_mask_per_spot, replace=False)
            mask_pattern[i, gene_indices] = True
        
        # Create features for training data (apply mask)
        if train_mask_only:
            # Training data: partially masked genes
            train_features = all_features.clone()
            train_features[mask_pattern] = mask_value
            
            # Test data: use complete features
            test_features = all_features.clone()
        else:
            # Both training and test data use the same mask pattern
            train_features = all_features.clone()
            train_features[mask_pattern] = mask_value
            
            # Create different mask pattern for test data
            test_mask_pattern = torch.zeros((n_spots, n_genes), dtype=torch.bool)
            for i in range(n_spots):
                gene_indices = np.random.choice(n_genes, size=n_mask_per_spot, replace=False)
                test_mask_pattern[i, gene_indices] = True
            
            test_features = all_features.clone()
            test_features[test_mask_pattern] = mask_value
        
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
                         pos=all_pos_norm,
                         edge_index=train_edge_index,
                         num_nodes=n_spots,
                         num_genes=n_genes)
        
        # Create test data (without graph structure)
        test_data = Data(x=test_features,
                        pos=all_pos_norm,
                        edge_index=None,  # Test data doesn't need graph structure
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

    def bulid_multi_graph(self, adatas, method='rknn'):
        pass
