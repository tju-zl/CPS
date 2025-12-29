import numpy as np
import scanpy as sc
import ot
import os
import json
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_attention_score(results):
    spot_attn = results['attn_weights'].mean(dim=2)
    
    scale_attention_stats = {
    'mean': results['attn_weights'].mean(dim=0),  # 每个尺度的平均关注度
    'std': results['attn_weights'].std(dim=0),    # 关注度的变异程度
    'max': results['attn_weights'].max(dim=0).values,  # 最大关注度
    'min': results['attn_weights'].min(dim=0).values,  # 最小关注度
    }
    
    return {
        'spot_attention': spot_attn,
        'scale_stats': scale_attention_stats
    }


def mclust(adata, arg, refine=False, key='z', pca_dim=32):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=pca_dim, random_state=arg.seed)
    embedding = pca.fit_transform(adata.obsm[key].copy())
    adata.obsm['emb_pca'] = embedding
    import rpy2.robjects as r_objects
    r_objects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = r_objects.r['set.seed']
    r_random_seed(arg.seed)
    rmclust = r_objects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm['emb_pca']), arg.clusters, 'EEE')

    mclust_res = np.array(res[-2])
    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')

    if refine:
        new_type = refine_label(adata, radius=15, key='mclust')
        adata.obs['mclust'] = new_type
    return adata

def refine_label(adata, radius=0, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    return new_type


def compute_imputation_metrics(original_data, imputed_data, test_indices=None,
                              mask_pattern=None, output_dir='./results',
                              experiment_name='imputation_experiment',
                              compute_full_data=False):
    """
    计算测试集空间和基因填补的benchmark测试指标
    
    Args:
        original_data: 原始数据，包含真实值
            - 如果是spots填补：包含所有spots的特征和位置
            - 如果是genes填补：包含完整基因表达矩阵
        imputed_data: 填补后的数据
            - 形状应与original_data相同
        test_indices: 测试spots的索引（spots填补时使用）
        mask_pattern: 基因掩码模式（genes填补时使用），布尔矩阵
        output_dir: 结果保存目录
        experiment_name: 实验名称，用于生成文件名
        compute_full_data: 是否计算全数据指标（默认False，只计算测试集）
        
    Returns:
        metrics_dict: 包含所有计算指标的字典
    """

    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为numpy数组
    if hasattr(original_data, 'x'):
        original = original_data.x.cpu().numpy() if hasattr(original_data.x, 'cpu') else original_data.x
        imputed = imputed_data.x.cpu().numpy() if hasattr(imputed_data.x, 'cpu') else imputed_data.x
    else:
        original = np.array(original_data)
        imputed = np.array(imputed_data)
    
    metrics = {
        'experiment_name': experiment_name,
        'timestamp': pd.Timestamp.now().isoformat(),
        'compute_full_data': compute_full_data
    }
    
    # 1. 通用指标（适用于所有填补任务）
    if compute_full_data:
        print("计算全数据填补指标...")
        
        # 全数据的MSE
        mse = mean_squared_error(original.flatten(), imputed.flatten())
        metrics['full_data_mse'] = float(mse)
        
        # 全数据的RMSE
        rmse = np.sqrt(mse)
        metrics['full_data_rmse'] = float(rmse)
        
        # 全数据的MAE
        mae = mean_absolute_error(original.flatten(), imputed.flatten())
        metrics['full_data_mae'] = float(mae)
        
        # 全数据的R²
        r2 = r2_score(original.flatten(), imputed.flatten())
        metrics['full_data_r2'] = float(r2)
        
        # 全数据的皮尔逊相关系数
        pearson_corr, pearson_p = pearsonr(original.flatten(), imputed.flatten())
        metrics['full_data_pearson_correlation'] = float(pearson_corr)
        metrics['full_data_pearson_p_value'] = float(pearson_p)
        
        # 全数据的斯皮尔曼相关系数
        spearman_corr, spearman_p = spearmanr(original.flatten(), imputed.flatten())
        metrics['full_data_spearman_correlation'] = float(spearman_corr)
        metrics['full_data_spearman_p_value'] = float(spearman_p)
    
    # 2. Spots填补特定指标（如果提供了test_indices）
    if test_indices is not None:
        print("计算Spots填补特定指标...")
        
        # 确保test_indices是numpy数组
        test_indices = np.array(test_indices)
        
        # 只计算测试spots的指标
        original_test = original[test_indices]
        imputed_test = imputed[test_indices]
        
        # 测试spots的MSE（存储为通用键）
        test_mse = mean_squared_error(original_test.flatten(), imputed_test.flatten())
        metrics['mse'] = float(test_mse)
        metrics['test_spots_mse'] = float(test_mse)  # 同时保留旧键名以保持兼容性
        
        # 测试spots的RMSE
        test_rmse = np.sqrt(test_mse)
        metrics['rmse'] = float(test_rmse)
        metrics['test_spots_rmse'] = float(test_rmse)
        
        # 测试spots的MAE
        test_mae = mean_absolute_error(original_test.flatten(), imputed_test.flatten())
        metrics['mae'] = float(test_mae)
        metrics['test_spots_mae'] = float(test_mae)
        
        # 测试spots的R²
        test_r2 = r2_score(original_test.flatten(), imputed_test.flatten())
        metrics['r2_score'] = float(test_r2)
        metrics['test_spots_r2'] = float(test_r2)
        
        # 每个测试spot的误差统计
        spot_errors = np.mean((original_test - imputed_test) ** 2, axis=1)
        metrics['test_spots_error_mean'] = float(np.mean(spot_errors))
        metrics['test_spots_error_std'] = float(np.std(spot_errors))
        metrics['test_spots_error_min'] = float(np.min(spot_errors))
        metrics['test_spots_error_max'] = float(np.max(spot_errors))
        
        # 空间连续性指标（如果提供了位置信息）
        if hasattr(original_data, 'pos'):
            positions = original_data.pos.cpu().numpy() if hasattr(original_data.pos, 'cpu') else original_data.pos
            test_positions = positions[test_indices]
            
            # 计算空间距离与填补误差的相关性
            if len(test_indices) > 1:
                # 计算测试spots之间的空间距离
                spatial_dist = cdist(test_positions, test_positions)
                
                # 计算填补误差的差异
                error_diff = np.abs(spot_errors[:, None] - spot_errors[None, :])
                
                # 计算空间自相关性（Moran's I的简化版本）
                # 只考虑最近邻的误差相关性
                n_neighbors = min(5, len(test_indices) - 1)
                spatial_correlations = []
                
                for i in range(len(test_indices)):
                    # 找到最近的邻居
                    dist_to_i = spatial_dist[i]
                    neighbor_indices = np.argsort(dist_to_i)[1:n_neighbors+1]  # 排除自身
                    
                    # 计算与邻居的误差相关性
                    if len(neighbor_indices) > 0:
                        neighbor_errors = spot_errors[neighbor_indices]
                        if np.std(neighbor_errors) > 0:
                            corr = np.corrcoef([spot_errors[i]] * len(neighbor_errors), neighbor_errors)[0, 1]
                            spatial_correlations.append(corr)
                
                if spatial_correlations:
                    metrics['spatial_autocorrelation_mean'] = float(np.mean(spatial_correlations))
                    metrics['spatial_autocorrelation_std'] = float(np.std(spatial_correlations))
    
    # 3. Genes填补特定指标（如果提供了mask_pattern）
    if mask_pattern is not None:
        print("计算Genes填补特定指标...")
        
        # 确保mask_pattern是布尔数组
        mask_pattern = np.array(mask_pattern, dtype=bool)
        
        # 只计算被mask位置的指标
        original_masked = original[mask_pattern]
        imputed_masked = imputed[mask_pattern]
        
        # 被mask基因的MSE（存储为通用键）
        masked_mse = mean_squared_error(original_masked, imputed_masked)
        metrics['mse'] = float(masked_mse)
        metrics['masked_genes_mse'] = float(masked_mse)  # 同时保留旧键名以保持兼容性
        
        # 被mask基因的RMSE
        masked_rmse = np.sqrt(masked_mse)
        metrics['rmse'] = float(masked_rmse)
        metrics['masked_genes_rmse'] = float(masked_rmse)
        
        # 被mask基因的MAE
        masked_mae = mean_absolute_error(original_masked, imputed_masked)
        metrics['mae'] = float(masked_mae)
        metrics['masked_genes_mae'] = float(masked_mae)
        
        # 被mask基因的R²
        masked_r2 = r2_score(original_masked, imputed_masked)
        metrics['r2_score'] = float(masked_r2)
        metrics['masked_genes_r2'] = float(masked_r2)
        
        # 被mask基因的皮尔逊相关系数
        masked_pearson_corr, masked_pearson_p = pearsonr(original_masked, imputed_masked)
        metrics['pearson_correlation'] = float(masked_pearson_corr)
        metrics['pearson_p_value'] = float(masked_pearson_p)
        
        # 被mask基因的斯皮尔曼相关系数
        masked_spearman_corr, masked_spearman_p = spearmanr(original_masked, imputed_masked)
        metrics['spearman_correlation'] = float(masked_spearman_corr)
        metrics['spearman_p_value'] = float(masked_spearman_p)
        
        # 每个基因的填补精度
        n_genes = original.shape[1]
        gene_metrics = []
        
        for gene_idx in range(n_genes):
            # 找到该基因被mask的spots
            gene_mask = mask_pattern[:, gene_idx]
            if np.any(gene_mask):
                gene_original = original[gene_mask, gene_idx]
                gene_imputed = imputed[gene_mask, gene_idx]
                
                gene_mse = mean_squared_error(gene_original, gene_imputed)
                gene_corr, _ = pearsonr(gene_original, gene_imputed)
                
                gene_metrics.append({
                    'gene_index': int(gene_idx),
                    'mse': float(gene_mse),
                    'correlation': float(gene_corr),
                    'n_masked_spots': int(np.sum(gene_mask))
                })
        
        metrics['per_gene_metrics'] = gene_metrics
        
        # 基因填补的总体统计
        if gene_metrics:
            gene_mses = [g['mse'] for g in gene_metrics]
            gene_corrs = [g['correlation'] for g in gene_metrics]
            
            metrics['gene_mse_mean'] = float(np.mean(gene_mses))
            metrics['gene_mse_std'] = float(np.std(gene_mses))
            metrics['gene_correlation_mean'] = float(np.mean(gene_corrs))
            metrics['gene_correlation_std'] = float(np.std(gene_corrs))
    
    # 4. 保存指标到文件
    print("保存指标到文件...")
    
    # JSON格式保存
    json_path = os.path.join(output_dir, f'{experiment_name}_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # CSV格式保存（扁平化）
    csv_metrics = {}
    for key, value in metrics.items():
        if key == 'per_gene_metrics':
            # 单独保存基因级别的指标
            gene_df = pd.DataFrame(value)
            gene_csv_path = os.path.join(output_dir, f'{experiment_name}_gene_metrics.csv')
            gene_df.to_csv(gene_csv_path, index=False)
            csv_metrics['per_gene_metrics_file'] = gene_csv_path
        elif isinstance(value, (list, dict)):
            # 复杂结构不保存到CSV
            continue
        else:
            csv_metrics[key] = value
    
    csv_path = os.path.join(output_dir, f'{experiment_name}_summary.csv')
    pd.DataFrame([csv_metrics]).to_csv(csv_path, index=False)
    
    print(f"指标已保存到: {json_path}")
    print(f"摘要已保存到: {csv_path}")
    
    # 5. 打印关键指标
    print("\n" + "="*60)
    print(f"填补实验: {experiment_name}")
    print("="*60)
    
    # 根据实际存在的指标键来打印
    # 首先检查spots填补指标
    if 'test_spots_mse' in metrics:
        print(f"\nSpots填补指标:")
        print(f"  MSE: {metrics['test_spots_mse']:.6f}")
        print(f"  RMSE: {metrics['test_spots_rmse']:.6f}")
        print(f"  MAE: {metrics['test_spots_mae']:.6f}")
        print(f"  R²: {metrics['test_spots_r2']:.6f}")
        # 如果有空间自相关性指标，也打印
        if 'spatial_autocorrelation_mean' in metrics:
            print(f"  空间自相关性: {metrics['spatial_autocorrelation_mean']:.6f}")
    
    # 然后检查genes填补指标
    elif 'masked_genes_mse' in metrics:
        print(f"\nGenes填补指标:")
        print(f"  MSE: {metrics['masked_genes_mse']:.6f}")
        print(f"  RMSE: {metrics['masked_genes_rmse']:.6f}")
        print(f"  MAE: {metrics['masked_genes_mae']:.6f}")
        print(f"  R²: {metrics['masked_genes_r2']:.6f}")
        if 'gene_correlation_mean' in metrics:
            print(f"  基因平均相关系数: {metrics['gene_correlation_mean']:.6f}")
    
    # 最后检查通用指标（如果存在）
    elif 'mse' in metrics:
        print(f"\n通用指标:")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  R²: {metrics['r2_score']:.6f}")
        if 'pearson_correlation' in metrics:
            print(f"  皮尔逊相关系数: {metrics['pearson_correlation']:.6f}")
    
    # 打印全数据指标（如果计算了）
    if compute_full_data and 'full_data_mse' in metrics:
        print(f"\n全数据指标:")
        print(f"  MSE: {metrics['full_data_mse']:.6f}")
        print(f"  RMSE: {metrics['full_data_rmse']:.6f}")
        print(f"  MAE: {metrics['full_data_mae']:.6f}")
        print(f"  R²: {metrics['full_data_r2']:.6f}")
    
    print("="*60)
    
    return metrics