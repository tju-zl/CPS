import numpy as np
import scanpy as sc
import ot
import os
import json
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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


def calc_morans_i(adata_input, genes=None):

    if 'neighbors' not in adata_input.uns:
        sc.pp.neighbors(adata_input, use_rep='spatial', n_neighbors=15)

    if genes is None:
        genes = adata_input.var_names

    m_i = sc.metrics.morans_i(adata_input, vals=adata_input[:, genes].X.T)
    
    return m_i

def calc_cnr(adata, gene_name, roi_mask, bg_mask):

    expr = adata[:, gene_name].X.toarray().flatten()

    signal_roi = expr[roi_mask]
    signal_bg = expr[bg_mask]

    mu_roi, std_roi = np.mean(signal_roi), np.std(signal_roi)
    mu_bg, std_bg = np.mean(signal_bg), np.std(signal_bg)

    cnr = np.abs(mu_roi - mu_bg) / np.sqrt(std_roi**2 + std_bg**2)
    
    return cnr, mu_roi, std_roi, mu_bg, std_bg