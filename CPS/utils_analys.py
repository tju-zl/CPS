import numpy as np
import scanpy as sc


def compute_attention_score(results):
    spot_attn = results['attn_weights'].mean(dim=2)
    
    scale_attention_stats = {
    'mean': results['attn_weights'].mean(dim=0),  # 每个尺度的平均关注度
    'std': results['attn_weights'].std(dim=0),    # 关注度的变异程度
    'max': results['attn_weights'].max(dim=0).values,  # 最大关注度
    'min': results['attn_weights'].min(dim=0).values,  # 最小关注度
    }


