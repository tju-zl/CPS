import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SSGConv, BatchNorm, LayerNorm
import numpy as np
from .module import *


class FourierFeatureEncoding(nn.Module):
    def __init__(self, in_dim=2, num_frequencies=32, sigma=1.0):
        super().__init__()
        self.in_dim = in_dim
        self.n_freq = num_frequencies
        self.sigma = sigma
        
        B = torch.randn(num_frequencies, in_dim) * sigma
        self.register_buffer('B', B, persistent=True)
    
    def out_dim(self):
        return self.in_dim * self.num_frequencies

    def forward(self, coords):  # coords: [N, in_dim]
        scaled = (2.0 * torch.pi) * (coords @ self.B.t())       # (N, n_freq)
        encoded = torch.cat([torch.cos(scaled), torch.sin(scaled)], dim=-1)
        return encoded          # (N, in_dim*n_freq)


class StudentINR(nn.Module):
    def __init__(self, coord_dim, latent_dim, num_freq, fourier_sigma):
        super().__init__()
        self.fourier = FourierFeatureEncoding(in_dim=coord_dim,
                                              num_frequencies=num_freq,
                                              sigma=fourier_sigma)
        enc_dim = self.fourier.out_dim()
        self.mlp = self._make_mlp(hidden_dims=[256, 256, 256], act=nn.SiLU(), 
                                  in_dim=coord_dim, out_dim=latent_dim)
    
    def _make_mlp(self, hidden_dims, act, in_dim, out_dim):
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), act]
            d = h
        layers += [nn.Linear(d, out_dim)]
        return nn.Sequential(*layers)
    
    def forward(self, pos):
        enc = self.fourier(pos)
        return self.mlp(enc)    # (N, l_dim)


# ! consider large scale dataset
class MultiScaleSSGConv(nn.Module):
    def __init__(self, in_dim, out_dim, k_list, dropout, add_self_loops=True):
        super().__init__()
        self.k_list = k_list
        
        self.convs = nn.ModuleList([SSGConv(in_channels=in_dim, 
                                            out_channels=out_dim,
                                            K=k,
                                            alpha=0.1, 
                                            cached=False,
                                            add_self_loops=add_self_loops) for k in k_list])
        
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
    

class TeacherNicheAttention(nn.Module):
    def __init__(self, in_dim, out_dim, k_list, num_heads, dropout, share_weights=False, prep_scale=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k_list = k_list
        self.num_scales = len(k_list)
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.share_weights = share_weights
        self.prep_scale = prep_scale
        
        if not prep_scale:
            self.multi_scale_convs = MultiScaleSSGConv(
                in_dim, out_dim, k_list, dropout)
        else:
            self.gene_proj = nn.Linear(in_dim, out_dim)
        
        if share_weights:
            self.query_proj = nn.Linear(out_dim, out_dim)
            self.key_proj = nn.Linear(out_dim, out_dim)
            self.value_proj = nn.Linear(out_dim, out_dim)
        else:
            self.query_projs = nn.ModuleList([
                nn.Linear(out_dim, out_dim) for _ in k_list
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
        
    def forward(self, x, edge_index=None, return_weights=True):
        N = x.shape[0]
        if not self.prep_scale:
            multi_scale_features = self.multi_scale_convs(x, edge_index) # list[(N, D)]
            scale_features = torch.stack(multi_scale_features, dim=1) # (N, S, D)
        else:
            scale_features = self.gene_proj(x)  # (N, S, D)
        
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
        self.decoder = self._make_mlp(hidden_dims=hid_dims, act=nn.SiLU(),
                                      in_dim=in_dim, out_dim=out_dim)
    
    def _make_mlp(self, hidden_dims, act, in_dim, out_dim):
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), act]
            d = h
        layers += [nn.Linear(d, out_dim)]
        return nn.Sequential(*layers)
        
    def forward(self, z):
        return self.decoder(z)


class CPSModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.teacher = TeacherNicheAttention(in_dim=args.hvgs, out_dim=args.latent_dim,
                                           k_list=args.k_list, num_heads=args.num_heads,
                                           dropout=args.dropout, share_weights=True,
                                           prep_scale=args.prep_scale)
        
        self.student = StudentINR(coord_dim=args.coord_dim, latent_dim=args.latent_dim,
                                  num_freq=args.freq, fourier_sigma=args.sigma)
        
        self.decoder = SharedDecoder(in_dim=args.latent_dim, out_dim=args.hvgs,
                                     hid_dims=args.latent_dim)
        
        self.projection_head = nn.Sequential(nn.Linear(args.latent_dim, args.latent_dim),
                                             nn.ReLU(),
                                             nn.Linear(args.latent_dim, args.latent_dim) 
        ) if args.distill == 0 else None
        
    def forward(self, coords, x=None, edge_index=None, mode='infer', return_attn=False):
        results = {}
        if mode == 'train':
            z_teacher, attn_weights = self.teacher(x, edge_index)
            recon_teacher = self.decoder(z_teacher)

            z_student = self.student(coords)
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
                
        elif mode == 'infer':
            z_student = self.student(coords)
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
