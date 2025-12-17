import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv, SSGConv, GCNConv, BatchNorm, LayerNorm
import numpy as np


class FourierFeatureEncoding(nn.Module):
    def __init__(self, in_dim=2, num_frequencies=10, sigma=1.0):
        super().__init__()
        self.n_freq = num_frequencies
        self.out_dim = in_dim * (2*num_frequencies) + in_dim
        
        self.register_buffer('freq_weights', torch.randn(in_dim, num_frequencies) * sigma)
    
    def forward(self, coords):
        scaled = 2 * torch.pi * coords @ self.freq_weights # (N, n_freq)
        sin_enc = torch.sin(scaled)
        cos_enc = torch.cos(scaled)
        
        encoded = torch.cat([coords, 
                             sin_enc.reshape(coords.shape[0], -1), 
                             cos_enc.reshape(coords.shape[0], -1)], 
                            dim=-1)
        
        return encoded

# ! 解决 大图预读取问题 （12-14）
class MultiScaleSSGConv(nn.Module):
    def __init__(self, in_dim, out_dim, k_list, dropout=0.1):
        super().__init__()
        self.k_list = k_list
        
        self.convs = nn.ModuleList([SSGConv(in_channels=in_dim, 
                                            out_channels=out_dim,
                                            K=k,
                                            alpha=0.1, 
                                            cached=False) for k in k_list])
        
        self.norms = nn.ModuleList([BatchNorm(out_dim) for _ in k_list])
        
        self.dropout = nn.Dropout(dropout)
        
        self.activation = nn.GELU()
        
    def forward(self, x, edge_index):
        feature = []
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = norm(h)
            h = self.activation(h)
            h = self.dropout(h)
            feature.append(h)
        return feature
    

class NicheCrossAttention(nn.Module):
    def __init__(self, in_dim, out_dim, k_list, num_heads, dropout, share_weights=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k_list = k_list
        self.num_scales = len(k_list)
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.share_weights = share_weights
        
        self.multi_scale_convs = MultiScaleSSGConv(
            in_dim, out_dim, k_list, dropout)
        
        if share_weights:
            self.query_proj = nn.Linear(in_dim, out_dim)
            self.key_proj = nn.Linear(out_dim, out_dim)
            self.value_proj = nn.Linear(out_dim, out_dim)
        else:
            self.query_projs = nn.ModuleList([
                nn.Linear(in_dim, out_dim) for _ in k_list
            ])
            self.key_projs = nn.ModuleList([
                nn.Linear(out_dim, out_dim) for _ in k_list
            ])
            self.value_projs = nn.ModuleList([
                nn.Linear(out_dim, out_dim) for _ in k_list
            ])
        
        self.out_proj = nn.Linear(out_dim, out_dim)
        
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        
        self.norm = LayerNorm(out_dim)
        
    def forward(self, x, edge_index, return_weights=True):
        N = x.shape[0]
        multi_scale_features = self.multi_scale_convs(x, edge_index) # list[(N, l_dim)]
        scale_features = torch.stack(multi_scale_features, dim=1) # (N, S, D)
        
        if self.share_weights:
            query = self.query_proj(x).reshape(N, self.num_heads, self.head_dim) # (N, H, D_h)
            keys = self.key_proj(scale_features).reshape(N, self.num_scales, self. num_heads, self.head_dim)
            values = self.value_proj(scale_features).reshape(N, self.num_scales, self. num_heads, self.head_dim)
        else:
            queries, keys, values = [], [], []
            for i, (q_proj, k_proj, v_proj) in enumerate(zip(self.query_projs, self.key_projs, self.value_projs)):
                q = q_proj(x).reshape(N, self.num_heads, self.head_dim)
                k = k_proj(multi_scale_features[i]).reshape(N, self.num_heads, self.head_dim)
                v = v_proj(multi_scale_features[i]).reshape(N, self.num_heads, self.head_dim)
                queries.append(q.unsqueeze(1))  # (N, 1, H, D_h)
                keys.append(k.unsqueeze(1))
                values.append(v.unsqueeze(1))
            query = torch.sum(torch.stack(queries, dim=1), dim=1)  # (N, H, D_h)
            keys = torch.stack(keys, dim=1)  # (N, S, H, D_h)
            values = torch.stack(values, dim=1)  # (N, S, H, D_h)
            
        attn_scores = torch.einsum('nhd,nshd->nsh', query, keys) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=1)  # (N, S, H)
        attn_weights = self.dropout(attn_weights)
        
        attended = torch.einsum('nsh,nshd->nhd', attn_weights, values)
        attended = attended.reshape(N, -1)  # (N, l_dim)
        
        output = self.out_proj(attended)
        output = self.dropout(output)
        
        residual = self.residual(x)
        output = self.norm(output + residual)
        
        if return_weights:
            return output, attn_weights
        else:
            return output, None
        

class SharedDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dims=[256,512]):
        super().__init__()
        layers = []
        
        for _ ,h_dim in enumerate(hid_dims):
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            in_dim = h_dim
            
        layers.append(nn.Linear(hid_dims[-1], out_dim))
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.decoder(z)
    
class CPSModel(nn.Module):
    def __init__(self, num_genes, latent_dim, 
                 teacher_hidden, teacher_k_list, teacher_num_layers, teacher_num_heads,
                 student_num_layers, student_num_heads, student_hid_dims,
                 decoder_hid_dims, num_frequencies, lambda_distill, dropout):
        super().__init__()
        
        self.lambda_distill = lambda_distill
        self.num_genes = num_genes
        
        self.teacher = NicheCrossAttention(in_dim=num_genes, out_dim=teacher_hidden,
                                           k_list=teacher_k_list, num_heads=teacher_num_heads,
                                           dropout=dropout, share_weights=True)
        
        self.student = FourierFeatureEncoding(in_dim=2, num_frequencies=num_frequencies,
                                              sigma=1.0)
        
        self.decoder = SharedDecoder(in_dim=latent_dim, out_dim=num_genes,
                                     hid_dims=decoder_hid_dims)
        
        self.projection_head = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                             nn.ReLU(),
                                             nn.Linear(latent_dim, latent_dim) 
        ) if lambda_distill > 0 else None
        
    def forward(self, mode='train', data=None, coordinates=None, return_attn=True):
        results = {}
        if mode == 'train':
            z_teacher, attn_weights = self.teacher(data.x, data.edge_index)
            recon_teacher = self.decoder(z_teacher)

            z_student = self.student(coordinates)
            recon_student = self.decoder(z_student)
            
            if self.projection_head is not None:
                z_teacher_proj = F.normalize(self.projection_head(z_teacher), dim=-1)
                z_student_proj = F.normalize(self.projection_head(z_student), dim=-1)
                distill_loss = 1 - F.cosine_similarity(z_teacher_proj, z_student_proj).mean()
            else:
                distill_loss = F.mse_loss(z_student, z_teacher.detach())
                
            results.update({
                'z_teacher': z_teacher,
                'z_student': z_student,
                'recon_teacher': recon_teacher,
                'recon_student': recon_student,
                'distill_loss': distill_loss
            })
            if return_attn and attn_weights is not None:
                results['attn_weights'] = attn_weights
                
        elif mode == 'inference':
            z_student = self.student(coordinates)
            recon_student = self.decoder(z_student)
            
            results.update({
                'z_student': z_student,
                'recon_student': recon_student
            })
        
        return results

    def compute_losses(self, pred_dict, gene_expr, recon_weight):
        losses = {}
        if 'recon_teacher' in pred_dict:
            recon_loss_teacher = F.mse_loss(pred_dict['recon_teacher'], gene_expr)
            losses['recon_teacher'] = recon_weight[0] * recon_loss_teacher
        
        if 'recon_student' in pred_dict:
            recon_loss_student = F.mse_loss(pred_dict['recon_student'], gene_expr)
            losses['recon_student'] = recon_weight[1] * recon_loss_student
            
        if 'distill_loss' in pred_dict:
            losses['distill'] = self.lambda_distill * pred_dict['distill_loss']

        total_loss = sum([losses[k] for k in losses])
        losses['total'] = total_loss
        
        return losses
