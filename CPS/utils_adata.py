import numpy as np
import torch
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import add_self_loops, dense_to_sparse, to_undirected
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
from collections import namedtuple
import scipy.sparse
from scipy.spatial import cKDTree
import scanpy as sc


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
    Based on the coordinate range of the existing data, a higher-resolution grid of coordinates is generated.
    Furthermore, distance filtering is applied to retain only the points within the tissue-covered area (removing the background).
    
    Args:
        adata: raw AnnData, include.obsm['spatial']
        upscale_factor: 2x2
        margin: 0 or smale value
        
    Returns:
        new_coords (np.array): (N_new, 2) 
    """
    coords = adata.obsm['spatial']
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2) 
    avg_step = np.mean(dists[:, 1])

    new_step = avg_step / upscale_factor
    
    x_range = np.arange(min_x - margin, max_x + margin, new_step)
    y_range = np.arange(min_y - margin, max_y + margin, new_step)
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    
    grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    print(f"num of raw spots: {len(coords)}")
    print(f"num of generation spots: {len(grid_coords)}")
    
    dist_threshold = avg_step * 0.8 

    dists_to_real, _ = tree.query(grid_coords, k=1)
  
    mask = dists_to_real < dist_threshold
    final_coords = grid_coords[mask]
    
    print(f"filter bg SR num: {len(final_coords)}")
    
    return final_coords


# Visium HD 2um (subcellular) -> 16um (cellular) 
# lr_coords = generate_lr_coords(adata, downscale_factor=8)

def generate_lr_coords(adata, downscale_factor=4, margin=0):
    """
    Visium HD Binning
    Args:
        adata: AnnData (Visium HD raw data),  .obsm['spatial']
        downscale_factor: 
                        Visium HD raw 2um
                        4 -> 8um 
                        8 -> 16um 
        margin: 0
        
    Returns:
        final_coords (np.array): (N_new, 2) 
    """
    coords = adata.obsm['spatial']
    

    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    
    if len(coords) > 10000:
        sample_indices = np.random.choice(len(coords), 10000, replace=False)
        sample_coords = coords[sample_indices]
        tree_calc = cKDTree(sample_coords)
        dists, _ = tree_calc.query(sample_coords, k=2)
    else:
        tree_calc = cKDTree(coords)
        dists, _ = tree_calc.query(coords, k=2)
        
    avg_step = np.mean(dists[:, 1])
    
    new_step = avg_step * downscale_factor
    
    print(f"Original Step (approx): {avg_step:.2f}")
    print(f"Target LR Step: {new_step:.2f}")
    
    x_range = np.arange(min_x - margin, max_x + margin + new_step, new_step)
    y_range = np.arange(min_y - margin, max_y + margin + new_step, new_step)
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    
    grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    print(f"Num of raw HD spots: {len(coords)}")
    print(f"Num of coarse grid spots (before filtering): {len(grid_coords)}")
    
    tree = cKDTree(coords)

    dist_threshold = new_step * 0.6 

    dists_to_real, _ = tree.query(grid_coords, k=1)
    
    mask = dists_to_real < dist_threshold
    final_coords = grid_coords[mask]
    
    print(f"Final LR coords num (Tissue Covered): {len(final_coords)}")
    
    return final_coords


from scipy.interpolate import griddata

def generate_sr_library_size(adata, sr_coords, mode='mean', k=0):
    """
    generate Library Size for super-res spots
    
    Args:
        adata: raw AnnData
        sr_coords: (N, 2) super-res coords
        mode: 
            - 'mean': Using the mean of all data (removes sequencing depth bias, resulting in the cleanest graph)
            - 'median': Use the median of all data (recommended, robust against outliers)
            - 'nearest': Nearest neighbor (will result in a blocky, pixelated appearance; not recommended)
            - 'linear': Linear interpolation (smoothly preserves cell density differences)
        k: This is only used under certain custom interpolation logic; it primarily relies on `griddata`.
    
    Returns:
        sr_lib_size: (N, 1) Torch Tensor
    """
    if 'total_counts' in adata.obs.columns:
        real_lib_size = adata.obs['total_counts'].values
    else:
        real_lib_size = np.array(adata.X.sum(axis=1)).flatten()
        
    real_coords = adata.obsm['spatial']
    
    print(f"Generating Library Size with mode: {mode}")
    
    if mode == 'mean':
        avg_val = np.mean(real_lib_size)
        sr_lib_size = np.full(len(sr_coords), avg_val)
        
    elif mode == 'median':
        med_val = np.median(real_lib_size)
        sr_lib_size = np.full(len(sr_coords), med_val)
        
    elif mode == 'linear' or mode == 'cubic':
        avg_val = np.mean(real_lib_size)
        sr_lib_size = griddata(
            points=real_coords, 
            values=real_lib_size, 
            xi=sr_coords, 
            method=mode, 
            fill_value=avg_val 
        )
        sr_lib_size = np.clip(sr_lib_size, a_min=1.0, a_max=None)
        
    elif mode == 'nearest':
        sr_lib_size = griddata(
            points=real_coords, 
            values=real_lib_size, 
            xi=sr_coords, 
            method='nearest'
        )
        
    else:
        raise ValueError("Unknown mode")
    if k > 0:
        print(f'raw lib min={sr_lib_size.min()}, max={sr_lib_size.max()}')
        sr_lib_size = sr_lib_size * k

    sr_lib_size = torch.FloatTensor(sr_lib_size).unsqueeze(1)
    
    return sr_lib_size


def construct_sr_adata(adata_raw, sr_expression, sr_latent, sr_coords, use_log1p=False):
    if torch.is_tensor(sr_expression):
        X_sr = sr_expression.detach().cpu().numpy()
    else:
        X_sr = sr_expression
    if use_log1p:
        X_sr = np.log1p(X_sr)
    
    adata_sr = sc.AnnData(X=X_sr)
    adata_sr.var = adata_raw.var.copy()
    adata_sr.obsm['spatial'] = sr_coords
    adata_sr.obsm['latent'] = sr_latent
    adata_sr.obs_names = [f"SR_{i}" for i in range(adata_sr.n_obs)]
    
    if 'spatial' in adata_raw.uns:
        adata_sr.uns['spatial'] = adata_raw.uns['spatial'].copy()
    
    return adata_sr