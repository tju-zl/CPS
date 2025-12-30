import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SSGConv, BatchNorm, LayerNorm
import numpy as np
from .module import *


class FourierFeatureEncoding(nn.Module):
    def __init__(self, in_dim=2, num_frequencies=8, sigma=1.0):
        super().__init__()
        self.in_dim = in_dim
        self.n_freq = num_frequencies
        self.sigma = sigma
        
        B = torch.randn(num_frequencies, in_dim) * sigma
        self.register_buffer('B', B, persistent=True)
    
    def out_dim(self):
        return self.in_dim * self.n_freq

    def forward(self, coords):  # coords: [N, in_dim]
        
        scaled = (2.0 * torch.pi) * (coords @ self.B.t())       # (N, n_freq)
        encoded = torch.cat([torch.cos(scaled), torch.sin(scaled)], dim=-1)
        return encoded          # (N, in_dim*n_freq)





class StudentINR(nn.Module):
    def __init__(self, coord_dim, latent_dim, num_freq, fourier_sigma, inr_latent):
        super().__init__()
        self.fourier = FourierFeatureEncoding(in_dim=coord_dim, num_frequencies=num_freq, sigma=fourier_sigma)
        enc_dim = self.fourier.out_dim()
        self.mlp = self._make_mlp(hidden_dims=inr_latent, act=nn.SiLU(), 
                                  in_dim=enc_dim, out_dim=latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
    
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
        return self.mlp(enc)    # (N, l_dim)


# ! consider large scale dataset
class MultiScaleSSGConv(nn.Module):
    def __init__(self, in_dim, out_dim, k_list, dropout, add_self_loops=True):
        super().__init__()
        self.k_list = k_list
        
        self.convs = nn.ModuleList([SSGConv(in_channels=in_dim, 
                                            out_channels=out_dim,
                                            K=k,
                                            alpha=0.2, 
                                            cached=False,
                                            add_self_loops=add_self_loops) for k in k_list])
        
        self.norms = nn.ModuleList([BatchNorm(out_dim) for _ in k_list])
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x, edge_index):
        feature = []
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = self.activation(h)
            h = norm(h)
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
        self.residual = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(out_dim)
        self.norm2 = LayerNorm(out_dim)
        # self.temperature = nn.Parameter(torch.ones(1) * 2.3)
        
    def forward(self, x, edge_index=None, return_weights=True):
        N = x.shape[0]
        if not self.prep_scale:
            multi_scale_features = self.multi_scale_convs(x, edge_index) # list[(N, D)]
            scale_features = torch.stack(multi_scale_features, dim=1) # (N, S, D)
        else:
            multi_scale_features = x
            scale_features = self.gene_proj(multi_scale_features)  # (N, S, D)
        
        # Pre-LN: 在计算注意力之前对特征进行归一化
        normed_features = self.norm(scale_features)
        
        if self.share_weights:
            query = self.query_proj(normed_features[:,0,:]).reshape(N, self.num_heads, self.head_dim) # (N, H, D_h)
            keys = self.key_proj(normed_features).reshape(N, self.num_scales, self.num_heads, self.head_dim)
            values = self.value_proj(normed_features).reshape(N, self.num_scales, self.num_heads, self.head_dim)
        else:
            queries, keys, values = [], [], []
            for i, (q_proj, k_proj, v_proj) in enumerate(zip(self.query_projs, self.key_projs, self.value_projs)):
                q = q_proj(normed_features[:,0,:]).reshape(N, self.num_heads, self.head_dim)
                k = k_proj(normed_features[:,i,:]).reshape(N, self.num_heads, self.head_dim)
                v = v_proj(normed_features[:,i,:]).reshape(N, self.num_heads, self.head_dim)
                queries.append(q)  # i+ [(N, H, D_h)]
                keys.append(k)
                values.append(v)
            query = torch.mean(torch.stack(queries, dim=1), dim=1)  # (N, H, D_h)
            keys = torch.stack(keys, dim=1)  # (N, S, H, D_h)
            values = torch.stack(values, dim=1)  # (N, S, H, D_h)
        
        # scale = torch.exp(self.temperature) / (self.head_dim ** 0.5)
        attn_scores = torch.einsum('nhd,nshd->nsh', query, keys) / (self.head_dim ** 0.5) #* scale
        attn_weights = F.softmax(attn_scores, dim=1)  # (N, S, H)
        attn_weights = self.dropout(attn_weights)
        
        attended = torch.einsum('nsh,nshd->nhd', attn_weights, values)
        output = attended.reshape(N, -1)  # (N, l_dim)
        
        # 添加残差连接：output = output + residual
        residual = self.residual(scale_features[:,0,:])
        output = output + residual
        
        # 应用输出投影
        output = self.out_proj(output)
        
        # output = self.norm2(output)
        if return_weights:
            return output, attn_weights
        else:
            return output, None


import torch.distributions as dist

class ZINBLoss(nn.Module):
    """
    Zero-Inflated Negative Binomial 损失函数
    专门用于基因表达数据（计数数据 + 零膨胀）
    """
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps
        
    def forward(self, x, mean, disp, pi, eps=1e-10):
        """
        计算ZINB负对数似然
        
        参数:
            x: 观测值 (batch_size, n_genes)
            mean: 负二项分布的均值 (batch_size, n_genes)
            disp: 离散度参数 (batch_size, n_genes) 或标量
            pi: 零膨胀概率 (batch_size, n_genes)
            
        返回:
            loss: 负对数似然
        """
        # 确保参数为正数
        mean = mean + eps
        disp = disp + eps
        pi = torch.clamp(pi, eps, 1-eps)
        
        # 负二项分布
        nb = dist.NegativeBinomial(
            total_count=1/disp,
            probs=disp/(mean + disp)
        )
        
        # 零膨胀负二项分布的对数似然
        # log likelihood = log(pi * I(x=0) + (1-pi) * NB(x|mean,disp))
        
        # 计算负二项分布的对数概率
        nb_log_prob = nb.log_prob(x)
        
        # 零膨胀部分
        if torch.any(x == 0):
            # 对于x=0的情况，考虑零膨胀
            zero_mask = (x == 0).float()
            # 零膨胀混合分布的对数似然
            log_likelihood = torch.log(
                pi * zero_mask + (1 - pi) * torch.exp(nb_log_prob) + eps
            )
        else:
            # 没有零值，直接使用负二项分布
            log_likelihood = torch.log(1 - pi + eps) + nb_log_prob
        
        # 负对数似然
        nll = -log_likelihood.mean()
        
        return nll


class ResidualBlock(nn.Module):
    """带跳层的残差块"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.norm2(x)
        x = self.act(x)
        return x + residual  # 跳层连接


class ZINBDecoder(nn.Module):
    """
    带跳层的ZINB解码器
    输出ZINB分布的三个参数：均值、离散度、零膨胀概率
    """
    def __init__(self, in_dim, out_dim, hid_dims=[512, 1024, 512], dropout=0.1):
        """
        参数:
            in_dim: 输入维度（隐空间维度）
            out_dim: 输出维度（基因数量）
            hid_dims: 隐藏层维度列表
            dropout: dropout概率
        """
        super().__init__()
        self.out_dim = out_dim
        
        # 构建编码网络（共享特征提取）
        layers = []
        d = in_dim
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hid_dims[0]),
            nn.LayerNorm(hid_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        d = hid_dims[0]
        
        # 残差块序列
        self.residual_blocks = nn.ModuleList()
        for h in hid_dims[1:]:
            self.residual_blocks.append(ResidualBlock(d, dropout))
            # 如果需要改变维度
            if h != d:
                self.residual_blocks.append(nn.Linear(d, h))
                d = h
        
        # 最后的残差块
        self.residual_blocks.append(ResidualBlock(d, dropout))
        
        # 三个输出头：均值、离散度、零膨胀概率
        self.mean_head = nn.Sequential(
            nn.Linear(d, out_dim),
            nn.Softplus()  # 均值必须为正
        )
        
        self.disp_head = nn.Sequential(
            nn.Linear(d, out_dim),
            nn.Softplus()  # 离散度必须为正
        )
        
        self.pi_head = nn.Sequential(
            nn.Linear(d, out_dim),
            nn.Sigmoid()  # 零膨胀概率在[0,1]之间
        )
        
    def forward(self, z):
        """
        前向传播，返回ZINB分布的三个参数
        
        返回:
            mean: 负二项分布的均值 (batch_size, n_genes)
            disp: 离散度参数 (batch_size, n_genes)
            pi: 零膨胀概率 (batch_size, n_genes)
        """
        # 输入投影
        x = self.input_proj(z)
        
        # 通过残差块
        for block in self.residual_blocks:
            x = block(x)
        
        # 三个输出头
        mean = self.mean_head(x)
        disp = self.disp_head(x)
        pi = self.pi_head(x)
        
        return mean, disp, pi


class SharedDecoder(nn.Module):
    """
    兼容的共享解码器（包装ZINB解码器）
    保持原有接口，但内部使用ZINB解码器
    """
    def __init__(self, in_dim, out_dim, hid_dims=[512, 1024, 512], dropout=0.1, use_zinb=True):
        super().__init__()
        self.use_zinb = use_zinb
        
        if use_zinb:
            # 使用ZINB解码器
            self.zinb_decoder = ZINBDecoder(in_dim, out_dim, hid_dims, dropout)
            # 为了保持兼容性，我们还需要一个简单的输出头
            self.simple_head = nn.Sequential(
                nn.Linear(hid_dims[-1], out_dim),
                nn.Softplus()
            )
        else:
            # 回退到原来的解码器（保持兼容）
            layers = []
            d = in_dim
            for h in hid_dims:
                layers.append(nn.Linear(d, h))
                layers.append(nn.LayerNorm(h))
                layers.append(nn.SiLU())
                layers.append(nn.Dropout(dropout))
                d = h
            layers.append(nn.Linear(d, out_dim))
            layers.append(nn.Softplus())
            self.decoder = nn.Sequential(*layers)
    
    def forward(self, z, return_params=False):
        """
        前向传播
        
        参数:
            z: 隐表示
            return_params: 是否返回ZINB参数
            
        返回:
            如果use_zinb=True且return_params=True: 返回(mean, disp, pi)
            否则: 返回基因表达预测值
        """
        if self.use_zinb:
            mean, disp, pi = self.zinb_decoder(z)
            if return_params:
                return mean, disp, pi
            else:
                # 返回期望值作为预测
                return mean * (1 - pi)  # 零膨胀调整后的期望
        else:
            return self.decoder(z)


class CPSModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.teacher = TeacherNicheAttention(in_dim=args.hvgs, out_dim=args.latent_dim,
                                           k_list=args.k_list, num_heads=args.num_heads,
                                           dropout=args.dropout, share_weights=args.sh_weights,
                                           prep_scale=args.prep_scale)
        
        self.student = StudentINR(coord_dim=args.coord_dim, latent_dim=args.latent_dim,
                                  num_freq=args.freq, fourier_sigma=args.sigma,
                                  inr_latent=args.inr_latent)
        
        self.decoder = SharedDecoder(in_dim=args.latent_dim, out_dim=args.hvgs,
                                     hid_dims=args.decoder_latent, dropout=args.dropout)
        
        self.projection_head = nn.Sequential(nn.Linear(args.latent_dim, args.latent_dim),
                                             nn.ReLU(),
                                             nn.Linear(args.latent_dim, args.latent_dim) 
        ) if args.distill == 0 else None
        
        
        
    def forward(self, coords, x=None, edge_index=None, return_attn=False, return_zinb_params=False):
        results = {}
        z_teacher, attn_weights = self.teacher(x, edge_index, return_attn)
        
        # 解码器输出（支持ZINB）
        if hasattr(self.decoder, 'use_zinb') and self.decoder.use_zinb:
            # ZINB解码器返回三个参数
            mean_teacher, disp_teacher, pi_teacher = self.decoder(z_teacher, return_params=True)
            recon_teacher = mean_teacher * (1 - pi_teacher)  # 期望值作为重建
            
            mean_student, disp_student, pi_student = self.decoder(self.student(coords), return_params=True)
            recon_student = mean_student * (1 - pi_student)
            
            # 存储ZINB参数
            if return_zinb_params:
                results.update({
                    'mean_teacher': mean_teacher,
                    'disp_teacher': disp_teacher,
                    'pi_teacher': pi_teacher,
                    'mean_student': mean_student,
                    'disp_student': disp_student,
                    'pi_student': pi_student
                })
        else:
            # 传统解码器
            recon_teacher = self.decoder(z_teacher)
            recon_student = self.decoder(self.student(coords))
        
        z_student = self.student(coords)
        
        if self.projection_head is not None:    # if not distill use contrastive alignment
            z_teacher_proj = F.normalize(self.projection_head(z_teacher), dim=-1)
            z_student_proj = F.normalize(self.projection_head(z_student), dim=-1)
            distill_loss = 1 - F.cosine_similarity(z_teacher_proj.detach(), z_student_proj).mean()
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

        return results

    def generate(self, coords):
        pass
    
    def interpret(self, x, edge_index=None, return_attn=True):
        pass