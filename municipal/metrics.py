"""Внутренние индексы в пространстве признаков и на взвешенном графе.

Соглашения и первоисточники приведены в reports/methodology.md.
Диагональ матрицы смежности должна быть нулевой.
"""
import numpy as np
from scipy import sparse
from sklearn.metrics import silhouette_score, calinski_harabasz_score


def network_scores(adjacency, labels):
    """AVI, AVU, BasicMQ и модульность Q для неориентированной сети.

    Внутренний вес B[k,k] учитывает каждое ребро дважды.
    Изолированный кластер получает Isolability=0, нулевой знаменатель
    Unifiability дает 0. BasicMQ использует направленное представление W.
    """
    _, y = np.unique(labels, return_inverse=True)
    n, k = len(y), len(np.unique(y))
    s = sparse.csr_matrix((np.ones(n), (np.arange(n), y)), shape=(n, k))
    b = (s.T @ adjacency @ s).toarray()
    size = np.bincount(y)
    internal = np.diag(b)
    volume = b.sum(axis=1)
    external = volume - internal
    isolation = np.divide(internal, volume, out=np.zeros(k), where=volume>0)
    denom = external[:, None] + external[None, :] - b
    unification = np.divide(b, denom, out=np.zeros_like(b), where=denom>0)
    np.fill_diagonal(unification, 0)
    avi, avu = isolation.mean(), unification.sum()/k
    within = internal/(size**2)
    between = b / (size[:, None]*size[None, :])
    np.fill_diagonal(between, 0)
    mq = within.mean() - (between.sum()/(k*(k-1)) if k>1 else 0)
    total = b.sum()
    q = (internal/total - (volume/total)**2).sum() if total>0 else 0.0
    return dict(AVI=float(avi), AVU=float(avu), MQ=float(mq), Q=float(q))


def s_dbw(x, labels):
    """S_Dbw: Scat + Dens_bw, радиус sqrt(sum ||variance_k||)/K.

    Плотность означает число точек ВНУТРИ шара. Плотность в середине
    считается на объединении двух кластеров, в центрах — на своих кластерах.
    При пустых шарах и нулевом знаменателе возвращается NaN, не ложный ноль.
    """
    groups = [x[labels==c] for c in np.unique(labels)]
    k = len(groups)
    if k < 2:
        return np.nan
    centers = [g.mean(axis=0) for g in groups]
    variances = np.array([np.linalg.norm(g.var(axis=0)) for g in groups])
    global_var = np.linalg.norm(x.var(axis=0))
    if global_var <= 0:
        return np.nan
    radius = np.sqrt(variances.sum())/k
    density = [np.sum(np.linalg.norm(g-c,axis=1)<=radius) for g,c in zip(groups,centers)]
    ratios = []
    for i in range(k):
        for j in range(i+1,k):
            den = max(density[i],density[j])
            if den == 0:
                return np.nan
            points = np.vstack([groups[i],groups[j]])
            middle = (centers[i]+centers[j])/2
            ratios.append(np.sum(np.linalg.norm(points-middle,axis=1)<=radius)/den)
    return float(variances.mean()/global_var + np.mean(ratios))


def evaluate(x, adjacency, labels, seed=42, sample_size=1200):
    k = len(np.unique(labels))
    sizes = np.bincount(np.unique(labels, return_inverse=True)[1])
    scores = network_scores(adjacency, labels)
    scores.update(k=k, min_cluster=int(sizes.min()), max_cluster=int(sizes.max()))
    if 1 < k < len(x):
        scores['SW'] = float(silhouette_score(x,labels,sample_size=min(len(x),sample_size),random_state=seed))
        scores['CH'] = float(calinski_harabasz_score(x,labels))
        scores['CH_per_node'] = scores['CH']/len(x)
        scores['S_Dbw'] = s_dbw(x,labels)
    else:
        scores.update(SW=np.nan,CH=np.nan,CH_per_node=np.nan,S_Dbw=np.nan)
    return scores
