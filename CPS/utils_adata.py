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
    
    def build_single_graph(self, adata, method='rknn'):
        pos = torch.FloatTensor(adata.obsm['spatial'])
        
        if method == 'knn':
            edge_index = knn_graph(pos, k=self.max_neighbors, flow=self.flow)
            edge_index = to_undirected(edge_index)
            
        elif method == 'rknn':
            edge_index = radius_graph(pos, r=self.radius, max_num_neighbors=self.max_neighbors, flow=self.flow)
            edge_index = to_undirected(edge_index)
        
        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=adata.n_obs)
            
        data = Data(x=torch.tensor(adata.obsm['hvg_features'], dtype=torch.float),
                    pos=pos,
                    edge_index=edge_index,
                    num_nodes=adata.n_obs, 
                    num_genes=adata.obsm['hvg_features'].shape[1])
        return data
    

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
        
        
# mask for cell and genes
class DataPerturb:
    def __init__(self, adata, method='spatial_imputation'):
        self.adata = adata.copy()
        self.method = method
        
    def mask_spots(self, down_ratio=0.5):
        pass
        