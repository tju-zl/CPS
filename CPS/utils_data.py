import torch
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import add_self_loops, dense_to_sparse
from torch_geometric.data import Data, Batch

# constract spatial graph using location
class SpatialGraphBuilder:
    def __init__(self, k, radius, max_num_neighs, self_loops, norm):
        self.k = k
        self.radius = radius
        self.max_num_neighs = max_num_neighs
        self.self_loops = self_loops
        self.norm = norm
    
    def build_graph(self, coordinates, gene_exp, method):
        N = len(coordinates)
        pos = torch.tensor(coordinates, dtype=torch.float)
        
        if method == 'knn':
            edge_index = knn_graph(pos, k=self.k)
        elif method == 'rknn':
            edge_index = radius_graph(pos, r=self.radius, max_num_neighs=self.max_num_neighs)
        
        edge_index = edge_index.to(torch.long)
        
        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)
            
        data = Data(x=torch.tensor(gene_exp, dtype=torch.float),
                    pos=pos, 
                    edge_index=edge_index,
                    num_nodes=N)
        
        return data
    

# preproduce the multi-scale features using ssgconv (APPNP)
class MultiScaleNicheProcessor:
    def __init__(self, k_list, x, edge_list):
        self.k_list = k_list
        

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
    def __init__(self):
        pass