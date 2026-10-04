"""Разреженные графы экономического сходства, без географических ограничений."""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors


def build_graph(x, neighbors=20, mutual=False, metric='euclidean'):
    """Симметричный kNN с локальным масштабом (Zelnik-Manor, Perona, 2004).

    w_ij=exp(-d_ij²/(sigma_i*sigma_j)), sigma_i — расстояние до k-го
    соседа. Ребро существует при i∈N_k(j) или j∈N_k(i); mutual=True
    заменяет объединение пересечением. Вес всегда в [0,1].
    """
    n = len(x)
    k = min(neighbors,n-1)
    if k<1:
        raise ValueError('Для графа нужны хотя бы два муниципалитета')
    model = NearestNeighbors(n_neighbors=k+1,metric=metric).fit(x)
    ds, ids = model.kneighbors(x)
    # Исключаем себя по индексу, а не по положению: возможны дубликаты.
    pairs = [(d[i!=row][:k],i[i!=row][:k]) for row,(d,i) in enumerate(zip(ds,ids))]
    dist = np.stack([p[0] for p in pairs]); idx=np.stack([p[1] for p in pairs])
    sigma = np.maximum(dist[:,-1],1e-8)
    weights=np.exp(-dist**2/(sigma[:,None]*sigma[idx]))
    w=sparse.csr_matrix((weights.ravel(),(np.repeat(np.arange(n),k),idx.ravel())),shape=(n,n))
    w=w.minimum(w.T) if mutual else w.maximum(w.T)
    w.setdiag(0); w.eliminate_zeros()
    return w


def graph_stats(w):
    nc, comp=connected_components(w,directed=False)
    degree=np.asarray(w.sum(axis=1)).ravel()
    return dict(edges=int(w.nnz//2),components=int(nc),isolated=int((degree==0).sum()),
                largest_component=int(np.bincount(comp).max()),mean_degree=float(np.diff(w.indptr).mean()))


def normalized_graph(w):
    d=np.asarray(w.sum(axis=1)).ravel()
    inv=np.divide(1,np.sqrt(d),out=np.zeros_like(d),where=d>0)
    return sparse.diags(inv) @ w @ sparse.diags(inv)
