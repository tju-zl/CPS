import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, ttest_ind, f_oneway
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class SpatialAttentionAnalyzer:
    def __init__(self, adata, results) -> None:
        pass
    
    def run_analyze(self, output_path=None):
        pass
    
    def print_basic_states(self):
        pass
    
    def plot_spatial_patterns(self):
        pass
    
    def plot_RH_patterns(self):
        pass


# plot scale attention
def plot_scale_attention_spatial(results, adata, plot_name=None, cmap='viridis'):
    if plot_name is None:
        scale_name = [f'Scale {i}' for i in range(results['attn_weights'].shape[1])]
        
    
# plot R/H metrics
def plot_RH_spatial(results, adata, cmap='viridis'):
    pass


# plot spatial cluster of attention
def plot_cluster_pattern(results, adata, cmap='viridis'):
    pass


# plot gene-attention patterns
def correlate_attention_to_gene(results, adata, gene_list=None, top_n=20):
    pass


# plot spatial-attention patterns
def correlate_attention_to_spatial(results, adata, domain_column='domain'):
    pass


# analyze attention trajectory
def plot_attention_trajectory(results, spatial_coords, n_intervals=10):
    pass


