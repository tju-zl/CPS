import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr


def _cosine_similarity(y_true, y_pred, mask=False):
    if mask:
        # non-zero position
        mask_nonzero = y_true > 0
        y_true = y_true[mask_nonzero]
        y_pred = y_pred[mask_nonzero]
    
    if len(y_true) == 0:
        return np.nan
    
    numerator = np.sum(y_true * y_pred, axis=-1)
    denominator = np.sqrt(np.sum(y_true ** 2, axis=-1)) * np.sqrt(np.sum(y_pred ** 2, axis=-1))
    pixelwise_cosine = numerator / (denominator + 1e-10)
    return pixelwise_cosine.mean()


def _spectral_angle_mapper(y_true, y_pred, mask=False):
    if mask:
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

    if mask:
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


def _support_recovery_rate(y_true, y_pred):
    one_map_A = (y_true > 0)
    one_map_B = (y_pred > 0)
    intersection = np.logical_and(one_map_A, one_map_B).sum()
    union = np.logical_or(one_map_A, one_map_B).sum()
    return intersection / union if union != 0 else 0


def _masked_mse(y_true, y_pred, mask=False):
    if mask:
        mask = y_true > 0
        if mask.sum() == 0:
            return np.nan
        return mean_squared_error(y_true[mask], y_pred[mask])
    else:
        return mean_squared_error(y_true, y_pred)


def _masked_mae(y_true, y_pred, mask=False):
    if mask:
        mask = y_true > 0
        if mask.sum() == 0:
            return np.nan
        return mean_absolute_error(y_true[mask], y_pred[mask])
    else:
        return mean_absolute_error(y_true, y_pred)


def compute_spots_imputation_metrics(test_expr, imputed_expr, verbose=True):
    test_expr = test_expr.cpu().numpy()
    imputed_expr = imputed_expr.cpu().numpy()
    
    # global metrics
    test_expr_flat = test_expr.flatten()
    imputed_expr_flat = imputed_expr.flatten()
    mse_global = mean_squared_error(test_expr_flat, imputed_expr_flat)
    rmse_global = np.sqrt(mse_global)
    mae_global = mean_absolute_error(test_expr_flat, imputed_expr_flat)
    r2_global = r2_score(test_expr_flat, imputed_expr_flat)
    pearson_global, _ = pearsonr(test_expr_flat, imputed_expr_flat)
    spearman_global, _ = spearmanr(test_expr_flat, imputed_expr_flat)
    
    # sample metrics
    cosine_sample = _cosine_similarity(test_expr, imputed_expr)
    cosine_nonzero_sample = _cosine_similarity(test_expr, imputed_expr, mask=True)
    pearson_sample = _pearson_r(test_expr, imputed_expr)
    pearson_nonzero_sample = _pearson_r(test_expr, imputed_expr, mask=True)
    spearman_sample = _spearman_r(test_expr, imputed_expr)
    spearman_nonzero_sample = _spearman_r(test_expr, imputed_expr, mask=True)
    sam_sample = _spectral_angle_mapper(test_expr, imputed_expr)
    sam_nonzero_sample = _spectral_angle_mapper(test_expr, imputed_expr, mask=True)
    mse_sample = _masked_mse(test_expr, imputed_expr)
    mse_nonzero_sample = _masked_mse(test_expr, imputed_expr, mask=True)
    mae_sample = _masked_mae(test_expr, imputed_expr)
    mae_nonzero_sample = _masked_mae(test_expr, imputed_expr, mask=True)
    
    metrics = {
        'mse_global': float(mse_global),
        'rmse_global': float(rmse_global),
        'mae_global': float(mae_global),
        'r2_global': float(r2_global),
        'pearson_global': float(pearson_global),
        'spearman_global': float(spearman_global),
        'cosine_sample': float(cosine_sample),
        'cosine_nonzero_sample': float(cosine_nonzero_sample),
        'pearson_sample': float(pearson_sample),         
        'pearson_nonzero_sample': float(pearson_nonzero_sample),
        'spearman_sample': float(spearman_sample),
        'spearman_nonzero_sample': float(spearman_nonzero_sample),
        'sam_sample': float(sam_sample),
        'sam_nonzero_sample': float(sam_nonzero_sample),
        'mse_sample': float(mse_sample),
        'mse_nonzero_sample': float(mse_nonzero_sample),
        'mae_sample': float(mae_sample),               
        'mae_nonzero_sample': float(mae_nonzero_sample),
        'n_sample': float(len(test_expr))
    }
    
    if verbose:
        print_metrics(metrics)
    
    return metrics


def print_metrics(metrics):
    print(f"\n{'='*60}")
    print("Spatial imputation metrics")
    print(f"{'='*60}")
    
    print("\n[global flatten metrics]")
    print(f"MSE:           {metrics.get('mse_global', 'N/A'):.6f}")
    print(f"RMSE:          {metrics.get('rmse_global', 'N/A'):.6f}")
    print(f"MAE:           {metrics.get('mae_global', 'N/A'):.6f}")
    print(f"R²:            {metrics.get('r2_global', 'N/A'):.6f}")
    print(f"Pearson:       {metrics.get('pearson_global', 'N/A'):.6f}")
    print(f"Spearman:      {metrics.get('spearman_global', 'N/A'):.6f}")
    
    print("\n[sample mean metrics]")
    print(f"CS:               {metrics.get('cosine_sample', 'N/A'):.6f}")
    print(f"CS nonzero:       {metrics.get('cosine_nonzero_sample', 'N/A'):.6f}")
    print(f"Pearson:          {metrics.get('pearson_sample', 'N/A'):.6f}")
    print(f"Pearson nonzero:  {metrics.get('pearson_nonzero_sample', 'N/A'):.6f}")
    print(f"Spearman:         {metrics.get('spearman_sample', 'N/A'):.6f}")
    print(f"Spearman nonzero: {metrics.get('spearman_nonzero_sample', 'N/A'):.6f}")
    print(f"SAM:              {metrics.get('sam_sample', 'N/A'):.6f}°")
    print(f"SAM nonzero:      {metrics.get('sam_nonzero_sample', 'N/A'):.6f}°")
    print(f"MSE:              {metrics.get('mse_sample', 'N/A'):.6f}")
    print(f"MSE nonzero:      {metrics.get('mse_nonzero_sample', 'N/A'):.6f}")
    print(f"MAE:              {metrics.get('mae_sample', 'N/A'):.6f}")
    print(f"MAE nonzero:      {metrics.get('mae_nonzero_sample', 'N/A'):.6f}")
    
    print(f"\n num of sample: {metrics.get('n_sample', 'N/A')}")
    print(f"{'='*60}")