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


# ============================================================================
# 填补指标可视化函数
# ============================================================================

def plot_imputation_metrics(metrics_dict, output_path=None, figsize=(15, 10)):
    """
    可视化填补指标
    
    Args:
        metrics_dict: compute_imputation_metrics返回的指标字典
        output_path: 保存图像的路径，如果为None则显示图像
        figsize: 图像大小
    """
    import os
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    
    # 1. 主要误差指标条形图
    ax1 = plt.subplot(2, 3, 1)
    main_metrics = ['mse', 'rmse', 'mae', 'r2_score']
    main_values = [metrics_dict.get(m, 0) for m in main_metrics]
    main_labels = ['MSE', 'RMSE', 'MAE', 'R²']
    
    bars = ax1.bar(main_labels, main_values)
    ax1.set_title('主要误差指标', fontsize=12, fontweight='bold')
    ax1.set_ylabel('值')
    ax1.grid(True, alpha=0.3)
    
    # 为条形添加数值标签
    for bar, val in zip(bars, main_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(main_values),
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 2. 相关系数图
    ax2 = plt.subplot(2, 3, 2)
    corr_metrics = ['pearson_correlation', 'spearman_correlation']
    corr_values = [metrics_dict.get(m, 0) for m in corr_metrics]
    corr_labels = ['Pearson', 'Spearman']
    
    bars2 = ax2.bar(corr_labels, corr_values, color=['#2E86AB', '#A23B72'])
    ax2.set_title('相关系数', fontsize=12, fontweight='bold')
    ax2.set_ylabel('相关系数')
    ax2.set_ylim(-1, 1)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    for bar, val in zip(bars2, corr_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Spots填补指标（如果存在）
    ax3 = plt.subplot(2, 3, 3)
    spots_metrics = []
    spots_values = []
    spots_labels = []
    
    for key in ['test_spots_mse', 'test_spots_rmse', 'test_spots_mae', 'test_spots_r2']:
        if key in metrics_dict:
            spots_metrics.append(key)
            spots_values.append(metrics_dict[key])
            spots_labels.append(key.replace('test_spots_', '').upper())
    
    if spots_metrics:
        bars3 = ax3.bar(spots_labels, spots_values, color=['#F18F01', '#C73E1D', '#6B2737', '#3F88C5'])
        ax3.set_title('Spots填补指标', fontsize=12, fontweight='bold')
        ax3.set_ylabel('值')
        ax3.grid(True, alpha=0.3)
        
        for bar, val in zip(bars3, spots_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(spots_values),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    else:
        ax3.text(0.5, 0.5, '无Spots填补数据', ha='center', va='center', fontsize=12)
        ax3.set_title('Spots填补指标', fontsize=12, fontweight='bold')
    
    # 4. Genes填补指标（如果存在）
    ax4 = plt.subplot(2, 3, 4)
    genes_metrics = []
    genes_values = []
    genes_labels = []
    
    for key in ['masked_genes_mse', 'masked_genes_rmse', 'masked_genes_mae', 'masked_genes_r2']:
        if key in metrics_dict:
            genes_metrics.append(key)
            genes_values.append(metrics_dict[key])
            genes_labels.append(key.replace('masked_genes_', '').upper())
    
    if genes_metrics:
        bars4 = ax4.bar(genes_labels, genes_values, color=['#44AF69', '#F8333C', '#FCAB10', '#2B9EB3'])
        ax4.set_title('Genes填补指标', fontsize=12, fontweight='bold')
        ax4.set_ylabel('值')
        ax4.grid(True, alpha=0.3)
        
        for bar, val in zip(bars4, genes_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(genes_values),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    else:
        ax4.text(0.5, 0.5, '无Genes填补数据', ha='center', va='center', fontsize=12)
        ax4.set_title('Genes填补指标', fontsize=12, fontweight='bold')
    
    # 5. 误差分布图（如果存在）
    ax5 = plt.subplot(2, 3, 5)
    
    # 尝试获取误差分布数据
    error_data = []
    error_labels = []
    
    if 'test_spots_error_mean' in metrics_dict:
        error_data.extend([
            metrics_dict.get('test_spots_error_mean', 0),
            metrics_dict.get('test_spots_error_std', 0),
            metrics_dict.get('test_spots_error_min', 0),
            metrics_dict.get('test_spots_error_max', 0)
        ])
        error_labels = ['均值', '标准差', '最小值', '最大值']
        
        bars5 = ax5.bar(error_labels, error_data, color=['#6A0572', '#AB83A1', '#3C91E6', '#A2D729'])
        ax5.set_title('Spots误差分布', fontsize=12, fontweight='bold')
        ax5.set_ylabel('误差值')
        ax5.grid(True, alpha=0.3)
        
        for bar, val in zip(bars5, error_data):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(error_data),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    else:
        ax5.text(0.5, 0.5, '无误差分布数据', ha='center', va='center', fontsize=12)
        ax5.set_title('误差分布', fontsize=12, fontweight='bold')
    
    # 6. 实验信息文本
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    experiment_info = f"实验名称: {metrics_dict.get('experiment_name', 'N/A')}\n"
    experiment_info += f"时间戳: {metrics_dict.get('timestamp', 'N/A')}\n\n"
    
    # 添加关键指标摘要
    experiment_info += "关键指标摘要:\n"
    experiment_info += f"• MSE: {metrics_dict.get('mse', 0):.6f}\n"
    experiment_info += f"• RMSE: {metrics_dict.get('rmse', 0):.6f}\n"
    experiment_info += f"• MAE: {metrics_dict.get('mae', 0):.6f}\n"
    experiment_info += f"• R²: {metrics_dict.get('r2_score', 0):.6f}\n"
    experiment_info += f"• Pearson: {metrics_dict.get('pearson_correlation', 0):.6f}\n"
    
    if 'test_spots_mse' in metrics_dict:
        experiment_info += f"• 测试Spots R²: {metrics_dict.get('test_spots_r2', 0):.6f}\n"
    
    if 'masked_genes_r2' in metrics_dict:
        experiment_info += f"• 被Mask基因 R²: {metrics_dict.get('masked_genes_r2', 0):.6f}\n"
    
    ax6.text(0.05, 0.95, experiment_info, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 调整布局
    plt.suptitle(f"填补性能评估: {metrics_dict.get('experiment_name', '实验')}",
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存或显示图像
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"可视化图像已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)


def plot_gene_level_metrics(metrics_dict, output_path=None, figsize=(12, 8)):
    """
    可视化基因级别的填补指标
    
    Args:
        metrics_dict: compute_imputation_metrics返回的指标字典
        output_path: 保存图像的路径，如果为None则显示图像
        figsize: 图像大小
    """
    import os
    
    if 'per_gene_metrics' not in metrics_dict or not metrics_dict['per_gene_metrics']:
        print("警告: 没有基因级别的指标数据")
        return
    
    # 获取基因级别数据
    gene_metrics = metrics_dict['per_gene_metrics']
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. 基因MSE分布直方图
    ax1 = axes[0, 0]
    mse_values = [g['mse'] for g in gene_metrics]
    ax1.hist(mse_values, bins=30, edgecolor='black', alpha=0.7, color='#2E86AB')
    ax1.set_xlabel('MSE')
    ax1.set_ylabel('基因数量')
    ax1.set_title('基因MSE分布', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_mse = np.mean(mse_values)
    median_mse = np.median(mse_values)
    ax1.axvline(mean_mse, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_mse:.4f}')
    ax1.axvline(median_mse, color='green', linestyle='--', linewidth=2, label=f'中位数: {median_mse:.4f}')
    ax1.legend()
    
    # 2. 基因相关系数分布直方图
    ax2 = axes[0, 1]
    corr_values = [g['correlation'] for g in gene_metrics]
    ax2.hist(corr_values, bins=30, edgecolor='black', alpha=0.7, color='#A23B72')
    ax2.set_xlabel('相关系数')
    ax2.set_ylabel('基因数量')
    ax2.set_title('基因相关系数分布', fontsize=12, fontweight='bold')
    ax2.set_xlim(-1, 1)
    ax2.grid(True, alpha=0.3)
    
    mean_corr = np.mean(corr_values)
    median_corr = np.median(corr_values)
    ax2.axvline(mean_corr, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_corr:.4f}')
    ax2.axvline(median_corr, color='green', linestyle='--', linewidth=2, label=f'中位数: {median_corr:.4f}')
    ax2.legend()
    
    # 3. MSE与相关系数散点图
    ax3 = axes[1, 0]
    n_masked = [g['n_masked_spots'] for g in gene_metrics]
    scatter = ax3.scatter(mse_values, corr_values, c=n_masked,
                         cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('MSE')
    ax3.set_ylabel('相关系数')
    ax3.set_title('MSE vs 相关系数', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('被Mask的Spots数量')
    
    # 4. 被Mask Spots数量分布
    ax4 = axes[1, 1]
    ax4.hist(n_masked, bins=30, edgecolor='black', alpha=0.7, color='#44AF69')
    ax4.set_xlabel('被Mask的Spots数量')
    ax4.set_ylabel('基因数量')
    ax4.set_title('被Mask Spots数量分布', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 调整布局
    plt.suptitle(f"基因级别填补性能分析: {metrics_dict.get('experiment_name', '实验')}",
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存或显示图像
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"基因级别可视化图像已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)


