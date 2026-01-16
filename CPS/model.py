import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, LayerNorm, SGConv, SSGConv
import numpy as np
import math
from .module import MultiHopSGConv


class FourierPositionMapping(nn.Module):
    def __init__(self, in_dim=2, num_frequencies=8, sigma=1.0):
        super().__init__()
        self.in_dim = in_dim
        self.n_freq = num_frequencies
        self.sigma = sigma
        
        B = torch.randn(num_frequencies, in_dim) * sigma
        self.register_buffer('B', B, persistent=True)
    
    def out_dim(self):
        return self.in_dim * self.n_freq

    def forward(self, coords):      # coords: [N, in_dim]
        scaled = (2.0 * torch.pi) * (coords @ self.B.t())       # (N, n_freq)
        encoded = torch.cat([torch.cos(scaled), torch.sin(scaled)], dim=-1)
        return encoded      # (N, in_dim*n_freq)


class StudentINR(nn.Module):
    def __init__(self, coord_dim, latent_dim, num_freq, fourier_sigma, inr_latent):
        super().__init__()
        self.fourier = FourierPositionMapping(in_dim=coord_dim, num_frequencies=num_freq, sigma=fourier_sigma)
        enc_dim = self.fourier.out_dim()
        self.mlp = self._make_mlp(hidden_dims=inr_latent, act=nn.SiLU(), 
                                  in_dim=enc_dim, out_dim=latent_dim)
    
    def _make_mlp(self, hidden_dims, act, in_dim, out_dim):
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), act, nn.LayerNorm(h)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        return nn.Sequential(*layers)
    
    def forward(self, pos):
        enc = self.fourier(pos)
        return self.mlp(enc)        # (N, l_dim)
    

# TeacherNetwork
class TeacherNicheAttention(nn.Module):
    def __init__(self, in_dim, out_dim, er_w, k_list, num_heads, dropout, prep_scale=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.er_w = er_w
        self.k_list = k_list
        self.num_scales = len(k_list)
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.prep_scale = prep_scale
        self.target_entropy = math.log(len(self.k_list)) * 0.5  # regularization ratio of entropy
        
        if not prep_scale:
            self.multi_scale_convs = MultiHopSGConv(in_channels=in_dim, 
                                                    out_channels=out_dim,
                                                    k_list=k_list,
                                                    prep_scale=prep_scale,
                                                    dropout=dropout)
        else:
            self.gene_proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout))
        
        self.query_proj = nn.Linear(out_dim, out_dim)
        self.key_proj = nn.Linear(out_dim, out_dim)
        self.value_proj = nn.Linear(out_dim, out_dim)
        
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm_input = nn.LayerNorm(out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale_type = nn.Parameter(torch.zeros(1, self.num_scales, out_dim))
        
        self.res_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim), 
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        
    def forward(self, x, edge_index=None):
        N = x.shape[0]
        
        if not self.prep_scale:
            scale_features = self.multi_scale_convs(x, edge_index) # (N, S, D)
        else:
            scale_features = self.gene_proj(x)  # (N, S, D)
        
        scale_features = scale_features + self.scale_type
        
        self_feature = self.norm_input(scale_features[:, 0, :])
        query = self.query_proj(self_feature).reshape(N, self.num_heads, self.head_dim)
        keys = self.key_proj(scale_features).reshape(N, self.num_scales, self.num_heads, self.head_dim)
        values = self.value_proj(scale_features).reshape(N, self.num_scales, self.num_heads, self.head_dim)

        attn_scores = torch.einsum('nhd,nshd->nsh', query, keys) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=1)  # (N, S, H)
        
        avg_weights = attn_weights.mean(dim=2)
        entropy = -torch.sum(avg_weights * torch.log(avg_weights + 1e-9), dim=1).mean()
        er_loss = self.er_w * F.relu(entropy - self.target_entropy)
        # print(er_loss.item(), entropy.item())
        
        attn_weights = self.dropout(attn_weights)
        
        context = torch.einsum('nsh,nshd->nhd', attn_weights, values).reshape(N, -1)
        context = self.out_proj(context)
        
        self_id = self.res_proj(self_feature)
        output = self_id + context

        return output, attn_weights, er_loss
    

# NB loss (or with MSE loss)
class SharedDecoder(nn.Module):
    def __init__(self,in_dim, out_dim, hid_dims, dropout):
        super().__init__()
        self.dec_mean = self._make_mlp(hidden_dims=hid_dims, 
                                       act=nn.SiLU(), 
                                       in_dim=in_dim,
                                       out_dim=out_dim, 
                                       dropout=dropout)
        # Gene-wise Dispersion
        self.px_r = nn.Parameter(torch.randn(out_dim))
    
    def _make_mlp(self, hidden_dims, act, in_dim, out_dim, dropout):
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), act, nn.LayerNorm(h), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        return nn.Sequential(*layers)
    
    def forward(self, z, library_size):
        # Mean (Softmax * Library Size)
        mean_prop = torch.softmax(self.dec_mean(z), dim=1)
        mean = mean_prop * library_size
        # Dispersion (Exp grantee to above 0)
        disp = torch.exp(self.px_r).unsqueeze(0).expand(z.size(0), -1)
        
        return mean, disp


class CPSModel(nn.Module):
    def __init__(self, args, in_dim=None):
        super().__init__()
        self.args = args
        torch.manual_seed(args.seed)
        in_dim = in_dim if in_dim is not None else args.hvgs
        self.teacher = TeacherNicheAttention(in_dim=in_dim, out_dim=args.latent_dim,
                                             er_w=args.er_w, k_list=args.k_list, 
                                             num_heads=args.num_heads, dropout=args.dropout, 
                                             prep_scale=args.prep_scale)
        torch.manual_seed(args.seed)
        self.student = StudentINR(coord_dim=args.coord_dim, latent_dim=args.latent_dim,
                                  num_freq=args.freq, fourier_sigma=args.sigma,
                                  inr_latent=args.inr_latent)
        torch.manual_seed(args.seed)
        self.decoder = SharedDecoder(in_dim=args.latent_dim, out_dim=args.hvgs,
                                     hid_dims=args.decoder_latent, dropout=args.dropout)
        
        self.loss_NB = NBLoss()
        self.loss_MSE = nn.MSELoss()
    
    def teacher_forward(self, x, edge_index, y, library_size):
        z_teacher, attn_weights, er_loss = self.teacher(x, edge_index)
        mean, disp = self.decoder(z_teacher, library_size)
        nb_loss = self.loss_NB(y, mean, disp) / self.args.hvgs
        mse_loss = F.mse_loss(torch.log1p(mean), torch.log1p(y))
        rec_loss = nb_loss + mse_loss + er_loss
        return z_teacher, mean, attn_weights, rec_loss
    
    def student_forward(self, coords, z_t, y, library_size):
        z_student = self.student(coords)
        pid_loss = F.mse_loss(z_student, z_t.detach())
        mean, disp = self.decoder(z_student, library_size)
        nb_loss = self.loss_NB(y, mean, disp) / self.args.hvgs
        mse_loss = F.mse_loss(torch.log1p(mean), torch.log1p(y))
        rec_loss = nb_loss + mse_loss
        return z_student, mean, pid_loss, rec_loss


class NBLoss(nn.Module):
    """
    Negative Binomial Loss
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y, mean, disp):
        # y: [batch, genes], count data
        # mean: predict mean (scaled by library size)
        # disp: dispersion theta
        mean = mean + self.eps
        disp = disp + self.eps
        
        t1 = torch.lgamma(disp + y)
        t2 = torch.lgamma(disp)
        t3 = torch.lgamma(y + 1.0)
        
        log_theta_mu = torch.log(disp + mean)
        t4 = disp * (torch.log(disp) - log_theta_mu)
        t5 = y * (torch.log(mean) - log_theta_mu)
        
        log_prob = t1 - t2 - t3 + t4 + t5
        loss = -log_prob
        return torch.mean(torch.sum(loss, dim=1))

