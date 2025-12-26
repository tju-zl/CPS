import numpy as np
import scanpy as sc
import ot


def compute_attention_score(results):
    spot_attn = results['attn_weights'].mean(dim=2)
    
    scale_attention_stats = {
    'mean': results['attn_weights'].mean(dim=0),  # 每个尺度的平均关注度
    'std': results['attn_weights'].std(dim=0),    # 关注度的变异程度
    'max': results['attn_weights'].max(dim=0).values,  # 最大关注度
    'min': results['attn_weights'].min(dim=0).values,  # 最小关注度
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