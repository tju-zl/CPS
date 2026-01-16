import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr


def _cosine_similarity(y_true, y_pred, mask=False):
    if mask:
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


def compute_spots_imputation_metrics(test_expr, imputed_expr, verbose=True, use_log1p=False):
    """
    compute spatial imutation metrics
    Args:
        test_expr: ground truth (Tensor or array)
        imputed_expr: imputed expression (Tensor or array)
        verbose: show results
        use_log1p: Bool. if True log1p in count data
    """
    
    # 1. to Numpy
    if torch.is_tensor(test_expr):
        test_expr = test_expr.detach().cpu().numpy()
    if torch.is_tensor(imputed_expr):
        imputed_expr = imputed_expr.detach().cpu().numpy()
    
    # 2. data transform (Switch)
    data_label = "Raw Count"
    if use_log1p:
        # safe 0
        imputed_expr = np.clip(imputed_expr, 0, None) 
        
        # Log1p transform: log(x + 1)
        test_expr = np.log1p(test_expr)
        imputed_expr = np.log1p(imputed_expr)
        data_label = "Log1p Transformed"

    if verbose:
        print(f"\nComputing metrics on **{data_label}** data...")
    
    # global flatten metrics
    test_expr_flat = test_expr.flatten()
    imputed_expr_flat = imputed_expr.flatten()
    
    mse_global = mean_squared_error(test_expr_flat, imputed_expr_flat)
    rmse_global = np.sqrt(mse_global)
    mae_global = mean_absolute_error(test_expr_flat, imputed_expr_flat)
    r2_global = r2_score(test_expr_flat, imputed_expr_flat)
    pearson_global, _ = pearsonr(test_expr_flat, imputed_expr_flat)
    spearman_global, _ = spearmanr(test_expr_flat, imputed_expr_flat)
    
    # sample metrics (spot-wise)
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
        'data_type': data_label, 
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


def compute_genes_imputation_metrics(test_expr, imputed_expr, mask_pattern, verbose=True, use_log1p=False):
    """
    Gene Imputation Metrics
    Args:
        test_expr: Ground Truth (Full Matrix)
        imputed_expr: Prediction (Full Matrix)
        mask_pattern: Boolean Matrix (True = Validation Set/Masked positions)
    """

    if torch.is_tensor(test_expr):
        test_expr = test_expr.detach().cpu().numpy()
    if torch.is_tensor(imputed_expr):
        imputed_expr = imputed_expr.detach().cpu().numpy()
    if torch.is_tensor(mask_pattern):
        mask_pattern = mask_pattern.detach().cpu().numpy().astype(bool)
        
    data_label = "Raw Count"
    if use_log1p:
        imputed_expr = np.clip(imputed_expr, 0, None)
        test_expr = np.log1p(test_expr)
        imputed_expr = np.log1p(imputed_expr)
        data_label = "Log1p Transformed"

    if verbose:
        print(f"\nComputing GENE metrics on **{data_label}** data (Masked only)...")

    y_true_flat = test_expr[mask_pattern]
    y_pred_flat = imputed_expr[mask_pattern]
    
    mse_global = mean_squared_error(y_true_flat, y_pred_flat)
    rmse_global = np.sqrt(mse_global)
    mae_global = mean_absolute_error(y_true_flat, y_pred_flat)
    r2_global = r2_score(y_true_flat, y_pred_flat)

    if len(y_true_flat) > 1 and np.std(y_true_flat) > 1e-9:
        pearson_global, _ = pearsonr(y_true_flat, y_pred_flat)
        spearman_global, _ = spearmanr(y_true_flat, y_pred_flat)
    else:
        pearson_global, spearman_global = np.nan, np.nan
    
    n_spots = test_expr.shape[0]

    res = {
        'pearson': [], 'spearman': [], 'cosine': [], 'sam': [], 'mse': [], 'mae': [],
        'pearson_nz': [], 'spearman_nz': [], 'cosine_nz': [], 'sam_nz': [], 'mse_nz': [], 'mae_nz': []
    }
    
    for i in range(n_spots):
        m = mask_pattern[i]
        if np.sum(m) < 2: 
            continue

        yt = test_expr[i][m]
        yp = imputed_expr[i][m]
        
        res['mse'].append(mean_squared_error(yt, yp))
        res['mae'].append(mean_absolute_error(yt, yp))
        
        if np.std(yt) > 1e-9 and np.std(yp) > 1e-9:
            res['pearson'].append(pearsonr(yt, yp)[0])
            res['spearman'].append(spearmanr(yt, yp)[0])
        
        num = np.dot(yt, yp)
        denom = np.linalg.norm(yt) * np.linalg.norm(yp)
        if denom > 0:
            cos_val = num / denom
            res['cosine'].append(cos_val)
            # SAM (Spectral Angle Mapper)
            cos_clip = np.clip(cos_val, -1.0, 1.0)
            res['sam'].append(np.rad2deg(np.arccos(cos_clip)))

        nz = yt > 0
        if np.sum(nz) > 1:
            yt_nz, yp_nz = yt[nz], yp[nz]
            res['mse_nz'].append(mean_squared_error(yt_nz, yp_nz))
            res['mae_nz'].append(mean_absolute_error(yt_nz, yp_nz))
            
            if np.std(yt_nz) > 1e-9 and np.std(yp_nz) > 1e-9:
                res['pearson_nz'].append(pearsonr(yt_nz, yp_nz)[0])
                res['spearman_nz'].append(spearmanr(yt_nz, yp_nz)[0])
            
            num_nz = np.dot(yt_nz, yp_nz)
            denom_nz = np.linalg.norm(yt_nz) * np.linalg.norm(yp_nz)
            if denom_nz > 0:
                cos_val_nz = num_nz / denom_nz
                res['cosine_nz'].append(cos_val_nz)
                res['sam_nz'].append(np.rad2deg(np.arccos(np.clip(cos_val_nz, -1.0, 1.0))))

    metrics = {
        'data_type': data_label,
        # Global
        'mse_global': float(mse_global),
        'rmse_global': float(rmse_global),
        'mae_global': float(mae_global),
        'r2_global': float(r2_global),
        'pearson_global': float(pearson_global),
        'spearman_global': float(spearman_global),
        
        # Sample (Spot-wise) Mean
        'mse_sample': float(np.nanmean(res['mse'])),
        'mae_sample': float(np.nanmean(res['mae'])),
        'pearson_sample': float(np.nanmean(res['pearson'])),
        'spearman_sample': float(np.nanmean(res['spearman'])),
        'cosine_sample': float(np.nanmean(res['cosine'])),
        'sam_sample': float(np.nanmean(res['sam'])),
        
        # Sample (Non-zero) Mean
        'mse_nonzero_sample': float(np.nanmean(res['mse_nz'])),
        'mae_nonzero_sample': float(np.nanmean(res['mae_nz'])),
        'pearson_nonzero_sample': float(np.nanmean(res['pearson_nz'])),
        'spearman_nonzero_sample': float(np.nanmean(res['spearman_nz'])),
        'cosine_nonzero_sample': float(np.nanmean(res['cosine_nz'])),
        'sam_nonzero_sample': float(np.nanmean(res['sam_nz'])),
        
        'n_sample': float(n_spots)
    }

    if verbose:
        print_metrics(metrics) 
        
    return metrics


def print_metrics(metrics):
    print(f"\n{'='*60}")
    print(f"Spatial imputation metrics [{metrics.get('data_type', 'Unknown')}]")
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
    
