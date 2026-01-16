import torch
from torch import Tensor
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm


class MultiHopSGConv(MessagePassing):
    def __init__(self, in_channels: int=0, out_channels: int=0, k_list=[],
                 add_self_loops: bool = True, dropout=0.2, prep_scale=False,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.k_list = k_list
        self.add_self_loops = add_self_loops
        self.prep_scale = prep_scale
        
        if not self.prep_scale:
            self.proj = nn.Sequential(
                Linear(in_channels, out_channels, bias),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        if not self.prep_scale:
            x = self.proj(x)
        
        features = []
        if 0 in self.k_list:
            features.append(x)
        
        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)

        for k in range(1, max(self.k_list) + 1):
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
            if k in self.k_list:
                features.append(x)
        
        features = torch.stack(features, dim=1)
        return features

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, k_list={self.k_list})')

