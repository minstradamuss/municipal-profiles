import numpy as np
from scipy import sparse
from sklearn.metrics import adjusted_rand_score
from municipal.clustering import align_labels,graph_regularized
from municipal.graph import build_graph,normalized_graph
from municipal.metrics import network_scores,s_dbw


def test_indices_on_two_disconnected_pairs():
    w=sparse.csr_matrix([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]],dtype=float)
    result=network_scores(w,np.array([0,0,1,1]))
    assert result=={'AVI':1.0,'AVU':0.0,'MQ':0.5,'Q':0.5}


def test_external_edge_and_label_permutation():
    w=sparse.csr_matrix([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]],dtype=float)
    a=network_scores(w,np.array([0,0,1,1]))
    b=network_scores(w,np.array([8,8,3,3]))
    assert a==b
    assert np.isclose(a['AVI'],2/3)
    assert np.isclose(a['AVU'],1)
    assert np.isclose(a['MQ'],0.25)


def test_matching_recovers_permuted_partition():
    old=np.array([0,0,1,1,2,2]); new=np.array([2,2,0,0,1,1])
    assert np.array_equal(align_labels(new,old,3),old)


def test_graph_handles_duplicates_and_mutual_is_subset():
    x=np.array([[0,0],[0,0],[1,0],[2,0],[10,0]],dtype=float)
    w=build_graph(x,neighbors=2); mutual=build_graph(x,neighbors=2,mutual=True)
    assert (w-w.T).nnz==0
    assert np.all(w.diagonal()==0)
    assert np.all((w.data>0)&(w.data<=1))
    assert np.all(mutual.toarray()<=w.toarray())
    assert np.isfinite(normalized_graph(mutual).data).all()


def test_sdbw_perfect_separation():
    x=np.array([[0,0]]*5+[[10,10]]*5,dtype=float)
    assert s_dbw(x,np.array([0]*5+[1]*5))==0


def test_objective_descends_and_signal_is_recovered():
    rng=np.random.default_rng(9)
    truth=np.repeat(np.arange(3),30)
    x=rng.normal(scale=.3,size=(90,3))+np.eye(3)[truth]*5
    w=build_graph(x,neighbors=10)
    y,history=graph_regularized(x,w,3,seed=42,graph_weight=.5,time_weight=.2,previous=truth)
    assert np.all(np.diff(history)<=1e-8)
    assert adjusted_rand_score(truth,y)>.95
    assert np.isfinite(history).all()


def test_zero_edge_graph_stays_finite():
    w=sparse.csr_matrix((4,4),dtype=float)
    assert all(np.isfinite(v) for v in network_scores(w,np.array([0,0,1,1])).values())
