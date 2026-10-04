"""Базовые методы и совместная оптимизация атрибутов, сети и истории."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from .graph import normalized_graph


def align_labels(current, previous, k):
    """Венгерское сопоставление: максимум пересечения соседних разбиений."""
    counts=np.zeros((k,k),dtype=int)
    np.add.at(counts,(current,previous),1)
    rows,cols=linear_sum_assignment(-counts)
    mapping=np.arange(k); mapping[rows]=cols
    return mapping[current]


def graph_regularized(x,w,k,seed=42,graph_weight=0.5,time_weight=0.15,
                      previous=None,max_iter=30):
    """Минимизирует сумму квадратов + взвешенный разрез + смену меток.

    J = sum_i ||x_i-c_z_i||²/p
      + graph_weight * sum_ij Wnorm_ij [z_i != z_j]
      + time_weight * sum_i [z_i != previous_i].
    Перебор узлов последовательный, центры обновляются после прохода.
    Перенос последней точки кластера запрещен, чтобы сохранять заданное K.
    """
    y=KMeans(k,n_init=20,random_state=seed).fit_predict(x)
    if previous is not None:
        y=align_labels(y,previous,k)
    a=normalized_graph(w).tocsr()
    rng=np.random.default_rng(seed)
    history=[]
    for iteration in range(max_iter):
        centers=np.stack([x[y==j].mean(0) for j in range(k)])
        sizes=np.bincount(y,minlength=k)
        changes=0
        for i in rng.permutation(len(x)):
            if sizes[y[i]]<=1:
                continue
            start,end=a.indptr[i:i+2]
            votes=np.bincount(y[a.indices[start:end]],weights=a.data[start:end],minlength=k)
            cost=((centers-x[i])**2).mean(1)-2*graph_weight*votes
            if previous is not None:
                cost+=time_weight*(np.arange(k)!=previous[i])
            candidate=int(np.argmin(cost))
            if candidate!=y[i] and cost[candidate]<cost[y[i]]-1e-12:
                sizes[y[i]]-=1; sizes[candidate]+=1; y[i]=candidate; changes+=1
        centers=np.stack([x[y==j].mean(0) for j in range(k)])
        objective=(((x-centers[y])**2).sum()/x.shape[1])
        coo=a.tocoo()
        objective+=graph_weight*np.sum(coo.data*(y[coo.row]!=y[coo.col]))
        if previous is not None:
            objective+=time_weight*np.sum(y!=previous)
        history.append(float(objective))
        if changes==0: break
    return y,history


def fit(method,x,w,k,seed=42,previous=None,graph_weight=0.5,time_weight=0.15,max_iter=30):
    if method=='graph_temporal':
        return graph_regularized(x,w,k,seed,graph_weight,time_weight,previous,max_iter)[0]
    if method=='kmeans':
        y=KMeans(k,n_init=20,random_state=seed).fit_predict(x)
    elif method=='ward':
        y=AgglomerativeClustering(n_clusters=k,linkage='ward').fit_predict(x)
    elif method=='spectral':
        y=SpectralClustering(n_clusters=k,affinity='precomputed',assign_labels='cluster_qr',
            random_state=seed,eigen_solver='arpack').fit_predict(w)
    else:
        raise ValueError(f'Неизвестный метод: {method}')
    return align_labels(y,previous,k) if previous is not None else y
