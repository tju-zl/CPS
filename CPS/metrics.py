"""
全面的填补指标计算函数
包含基本回归指标和SUICA风格的专门指标
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr


def _cosine_similarity(y_true, y_pred, mask=False):
    """
    计算余弦相似度
    """
    if mask:
        # 只计算非零位置
        mask_nonzero = y_true > 0
        y_true = y_true[mask_nonzero]
        y_pred = y_pred[mask_nonzero]
    
    if len(y_true) == 0:
        return np.nan
    
    # 向量化计算余弦相似度
    numerator = np.sum(y_true * y_pred, axis=-1)
    denominator = np.sqrt(np.sum(y_true ** 2, axis=-1)) * np.sqrt(np.sum(y_pred ** 2, axis=-1))
    pixelwise_cosine = numerator / (denominator + 1e-10)
    return pixelwise_cosine.mean()


def _spectral_angle_mapper(y_true, y_pred, mask=False):
    """
    计算光谱角映射器（SAM）
    """
    if mask:
        # 只计算非零位置
        mask_nonzero = y_true > 0
        y_true = y_true[mask_nonzero]
        y_pred = y_pred[mask_nonzero]
    
    if len(y_true) == 0:
        return np.nan
    
    numerator = np.sum(y_true * y_pred, axis=-1)
    denominator = np.sqrt(np.sum(y_true ** 2, axis=-1)) * np.sqrt(np.sum(y_pred ** 2, axis=-1))
    pixelwise_cosine = numerator / (denominator + 1e-10)
    cos_theta = np.clip(pixelwise_cosine, -1.0, 1.0)
    sam_angle = np.rad2deg(np.arccos(cos_theta))
    return sam_angle.mean()


def _spearman_r(y_true, y_pred, mask=False):
    """
    计算Spearman相关系数（逐样本）
    """
    if mask:
        # 只计算非零位置
        corrs = []
        for i in range(y_true.shape[0]):
            mask_i = y_true[i] > 0
            if np.sum(mask_i) > 1:
                corr = spearmanr(y_true[i][mask_i], y_pred[i][mask_i]).statistic
                corrs.append(corr)
    else:
        corrs = []
        for i in range(y_true.shape[0]):
            if len(y_true[i]) > 1:
                corr = spearmanr(y_true[i], y_pred[i]).statistic
                corrs.append(corr)
    
    if len(corrs) == 0:
        return np.nan
    return np.nanmean(corrs)


def _pearson_r(y_true, y_pred, mask=False):
    """
    计算Pearson相关系数（逐样本）
    """
    if mask:
        # 只计算非零位置
        corrs = []
        for i in range(y_true.shape[0]):
            mask_i = y_true[i] > 0
            if np.sum(mask_i) > 1:
                corr = pearsonr(y_true[i][mask_i], y_pred[i][mask_i]).statistic
                corrs.append(corr)
    else:
        corrs = []
        for i in range(y_true.shape[0]):
            if len(y_true[i]) > 1:
                corr = pearsonr(y_true[i], y_pred[i]).statistic
                corrs.append(corr)
    
    if len(corrs) == 0:
        return np.nan
    return np.nanmean(corrs)


def _iou_zero_map(y_true, y_pred):
    """
    计算零值图的IoU
    """
    zero_map_A = (y_true == 0)
    zero_map_B = (y_pred == 0)
    intersection = np.logical_and(zero_map_A, zero_map_B).sum()
    union = np.logical_or(zero_map_A, zero_map_B).sum()
    return intersection / union if union != 0 else 0


def _support_recovery_rate(y_true, y_pred):
    """
    计算非零值图的支持恢复率
    """
    one_map_A = (y_true > 0)
    one_map_B = (y_pred > 0)
    intersection = np.logical_and(one_map_A, one_map_B).sum()
    union = np.logical_or(one_map_A, one_map_B).sum()
    return intersection / union if union != 0 else 0


def _masked_mse(y_true, y_pred):
    """
    计算掩码MSE（只计算非零位置）
    """
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return mean_squared_error(y_true[mask], y_pred[mask])


def _masked_mae(y_true, y_pred):
    """
    计算掩码MAE（只计算非零位置）
    """
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return mean_absolute_error(y_true[mask], y_pred[mask])


def compute_simple_metrics(original, imputed, mask=None):
    """
    计算简洁的填补指标（保持向后兼容）
    
    参数:
        original: numpy array, 原始数据
        imputed: numpy array, 填补数据
        mask: numpy array (bool), 可选，指定计算哪些位置
    
    返回:
        dict: 包含各项指标的字典
    """
    if mask is not None:
        # 只计算mask位置的指标
        original_flat = original[mask].flatten()
        imputed_flat = imputed[mask].flatten()
    else:
        # 计算所有位置的指标
        original_flat = original.flatten()
        imputed_flat = imputed.flatten()
    
    # 基本回归指标
    mse = mean_squared_error(original_flat, imputed_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(original_flat, imputed_flat)
    r2 = r2_score(original_flat, imputed_flat)
    
    # 相关系数
    if len(original_flat) > 1:
        pearson_corr, pearson_p = pearsonr(original_flat, imputed_flat)
        spearman_corr, spearman_p = spearmanr(original_flat, imputed_flat)
    else:
        pearson_corr = pearson_p = spearman_corr = spearman_p = np.nan
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'pearson_corr': float(pearson_corr),
        'spearman_corr': float(spearman_corr),
        'n_samples': len(original_flat)
    }
    
    return metrics


def compute_comprehensive_metrics(original, imputed, mask=None, fast=False):
    """
    计算全面的填补指标（SUICA风格）
    
    参数:
        original: numpy array, 原始数据
        imputed: numpy array, 填补数据
        mask: numpy array (bool), 可选，指定计算哪些位置
        fast: bool, 是否跳过耗时指标
    
    返回:
        dict: 包含各项指标的字典
    """
    # 应用mask
    if mask is not None:
        original_masked = original[mask]
        imputed_masked = imputed[mask]
    else:
        original_masked = original
        imputed_masked = imputed
    
    # 展平版本用于基本指标
    original_flat = original_masked.flatten()
    imputed_flat = imputed_masked.flatten()
    
    # 基本回归指标
    mse = mean_squared_error(original_flat, imputed_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(original_flat, imputed_flat)
    r2 = r2_score(original_flat, imputed_flat)
    
    # 全局相关系数
    if len(original_flat) > 1:
        pearson_global, _ = pearsonr(original_flat, imputed_flat)
        spearman_global, _ = spearmanr(original_flat, imputed_flat)
    else:
        pearson_global = spearman_global = np.nan
    
    # 准备逐样本计算的数据
    if original_masked.ndim == 1:
        original_2d = original_masked.reshape(-1, 1)
        imputed_2d = imputed_masked.reshape(-1, 1)
    else:
        original_2d = original_masked
        imputed_2d = imputed_masked
    
    # 计算专门指标
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'pearson_global': float(pearson_global),
        'spearman_global': float(spearman_global),
        'cosine_similarity': float(_cosine_similarity(original_2d, imputed_2d)),
        'cosine_similarity_mask': float(_cosine_similarity(original_2d, imputed_2d, mask=True)),
        'sam': float(_spectral_angle_mapper(original_2d, imputed_2d)),
        'sam_mask': float(_spectral_angle_mapper(original_2d, imputed_2d, mask=True)),
        'iou_zero': float(_iou_zero_map(original_masked, imputed_masked)),
        'support_recovery': float(_support_recovery_rate(original_masked, imputed_masked)),
        'mse_mask': float(_masked_mse(original_masked, imputed_masked)),
        'mae_mask': float(_masked_mae(original_masked, imputed_masked)),
        'n_samples': len(original_flat)
    }
    
    # 耗时指标（可跳过）
    if not fast:
        metrics.update({
            'pearson_per_sample': float(_pearson_r(original_2d, imputed_2d)),
            'spearman_per_sample': float(_spearman_r(original_2d, imputed_2d)),
            'pearson_per_sample_mask': float(_pearson_r(original_2d, imputed_2d, mask=True)),
            'spearman_per_sample_mask': float(_spearman_r(original_2d, imputed_2d, mask=True)),
        })
    
    return metrics


def compute_spots_imputation_metrics(original_data, imputed_data, test_indices, comprehensive=True, fast=False):
    """
    计算spots填补指标
    
    参数:
        original_data: Data object 或 numpy array 或 torch.Tensor, 原始数据
        imputed_data: Data object 或 numpy array 或 torch.Tensor, 填补数据
        test_indices: list/array, 测试spots索引
        comprehensive: bool, 是否使用全面指标（SUICA风格）
        fast: bool, 是否跳过耗时指标
    
    返回:
        dict: 填补指标
    """
    # 提取原始数据
    if hasattr(original_data, 'y'):
        # Data对象
        original = original_data.y.cpu().numpy() if hasattr(original_data.x, 'cpu') else original_data.x
    elif hasattr(original_data, 'cpu'):
        # torch.Tensor
        original = original_data.cpu().numpy()
    else:
        # numpy array
        original = np.array(original_data)
    
    # 提取填补数据
    if hasattr(imputed_data, 'y'):
        # Data对象
        imputed = imputed_data.y.cpu().numpy() if hasattr(imputed_data.x, 'cpu') else imputed_data.x
    elif hasattr(imputed_data, 'cpu'):
        # torch.Tensor
        imputed = imputed_data.cpu().numpy()
    else:
        # numpy array
        imputed = np.array(imputed_data)
    
    # 只计算测试spots
    original_test = original[test_indices]
    imputed_test = imputed[test_indices]
    
    if comprehensive:
        return compute_comprehensive_metrics(original_test, imputed_test, fast=fast)
    else:
        return compute_simple_metrics(original_test, imputed_test)


def compute_genes_imputation_metrics(original_data, imputed_data, mask_pattern, comprehensive=False, fast=False):
    """
    计算genes填补指标
    
    参数:
        original_data: Data object 或 numpy array, 原始数据
        imputed_data: Data object 或 numpy array, 填补数据
        mask_pattern: numpy array (bool), 基因掩码模式
        comprehensive: bool, 是否使用全面指标（SUICA风格）
        fast: bool, 是否跳过耗时指标
    
    返回:
        dict: 填补指标
    """
    # 提取数据
    if hasattr(original_data, 'x'):
        original = original_data.x.cpu().numpy() if hasattr(original_data.x, 'cpu') else original_data.x
        imputed = imputed_data.x.cpu().numpy() if hasattr(imputed_data.x, 'cpu') else imputed_data.x
    else:
        original = np.array(original_data)
        imputed = np.array(imputed_data)
    
    # 确保mask_pattern是布尔数组
    mask_pattern = np.array(mask_pattern, dtype=bool)
    
    if comprehensive:
        return compute_comprehensive_metrics(original, imputed, mask=mask_pattern, fast=fast)
    else:
        return compute_simple_metrics(original, imputed, mask_pattern)


def print_metrics(metrics, title="填补指标", comprehensive=True):
    """
    打印指标
    
    参数:
        metrics: dict, 指标字典
        title: str, 标题
        comprehensive: bool, 是否打印全面指标
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # 基本指标
    print("\n[基本回归指标]")
    print(f"MSE:           {metrics.get('mse', 'N/A'):.6f}")
    print(f"RMSE:          {metrics.get('rmse', 'N/A'):.6f}")
    print(f"MAE:           {metrics.get('mae', 'N/A'):.6f}")
    print(f"R²:            {metrics.get('r2', 'N/A'):.6f}")
    
    # 全局相关系数
    print("\n[全局相关系数]")
    print(f"Pearson:       {metrics.get('pearson_global', metrics.get('pearson_corr', 'N/A')):.6f}")
    print(f"Spearman:      {metrics.get('spearman_global', metrics.get('spearman_corr', 'N/A')):.6f}")
    
    if comprehensive:
        # 专门指标
        print("\n[专门指标]")
        print(f"Cosine相似度:  {metrics.get('cosine_similarity', 'N/A'):.6f}")
        print(f"Cosine(mask): {metrics.get('cosine_similarity_mask', 'N/A'):.6f}")
        print(f"SAM角度:      {metrics.get('sam', 'N/A'):.6f}°")
        print(f"SAM(mask):    {metrics.get('sam_mask', 'N/A'):.6f}°")
        print(f"零值IoU:      {metrics.get('iou_zero', 'N/A'):.6f}")
        print(f"支持恢复率:    {metrics.get('support_recovery', 'N/A'):.6f}")
        print(f"MSE(mask):    {metrics.get('mse_mask', 'N/A'):.6f}")
        print(f"MAE(mask):    {metrics.get('mae_mask', 'N/A'):.6f}")
        
        # 逐样本相关系数（如果存在）
        if 'pearson_per_sample' in metrics:
            print("\n[逐样本相关系数]")
            print(f"Pearson/样本:  {metrics.get('pearson_per_sample', 'N/A'):.6f}")
            print(f"Spearman/样本: {metrics.get('spearman_per_sample', 'N/A'):.6f}")
            print(f"Pearson(mask): {metrics.get('pearson_per_sample_mask', 'N/A'):.6f}")
            print(f"Spearman(mask):{metrics.get('spearman_per_sample_mask', 'N/A'):.6f}")
    
    print(f"\n样本数:        {metrics.get('n_samples', 'N/A')}")
    print(f"{'='*60}")