"""Полный воспроизводимый расчет: python run.py --config configs/default.yaml."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
import json
import platform
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.stats import kruskal
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from municipal.data import prepare,COMPOSITION
from municipal.graph import build_graph,graph_stats
from municipal.clustering import fit,align_labels
from municipal.metrics import evaluate,network_scores


def log(msg): print(msg,flush=True)


def run(cfg):
    out=Path(cfg['output_dir']); out.mkdir(exist_ok=True,parents=True)
    (out/'graphs').mkdir(exist_ok=True)
    seed=cfg['seed']; cl=cfg['clustering']; ev=cfg['evaluation']
    panel,xs,info,excluded,quality=prepare(cfg)
    months=list(xs); ids=info.index.to_numpy(); n=len(ids)
    log(f'Панель: {n} МО, {len(months)} месяцев')
    info.to_csv(out/'municipalities.csv',encoding='utf-8-sig')
    excluded.to_csv(out/'excluded_municipalities.csv',index=False,encoding='utf-8-sig')
    panel.to_parquet(out/'panel.parquet')
    (out/'data_quality.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2),encoding='utf-8')
    graphs={m:build_graph(x,**cfg['graph']) for m,x in xs.items()}
    pd.DataFrame([dict(month=m,**graph_stats(w)) for m,w in graphs.items()]).to_csv(out/'graph_diagnostics.csv',index=False)
    for m,w in graphs.items(): sparse.save_npz(out/'graphs'/f'{m}.npz',w)
    rows=[]
    for k in cl['k_candidates']:
        for method in cl['methods']:
            for m in cfg['calibration_months']:
                y=fit(method,xs[m],graphs[m],k,seed,graph_weight=cl['graph_weight'],time_weight=0,max_iter=cl['max_iter'])
                rows.append(dict(month=m,method=method,k_candidate=k,**evaluate(xs[m],graphs[m],y,seed,ev['silhouette_sample'])))
        log(f'Сравнение методов: K={k}')
    tuning=pd.DataFrame(rows); tuning.to_csv(out/'calibration_metrics.csv',index=False)
    rank=tuning.groupby(['method','k_candidate'])[['SW','CH','S_Dbw','Q','min_cluster']].mean().reset_index()
    # K основной модели выбирается только по калибровочным месяцам.
    own=rank[rank.method==cl['main_method']].copy()
    own['rank_score']=sum(own[c].rank(ascending=asc,na_option='bottom') for c,asc in [('SW',False),('CH',False),('S_Dbw',True),('Q',False)])/4
    eligible=own[own.min_cluster>=cl['minimum_cluster_size']]
    if eligible.empty: raise ValueError('Ни одно K не удовлетворяет минимальному размеру кластера')
    chosen=eligible.sort_values(['rank_score','k_candidate']).iloc[0]; k=int(chosen.k_candidate)
    own.to_csv(out/'selection.csv',index=False)
    log(f'Выбрано K={k} по данным до {cfg["calibration_end"]}')
    labels={}; metrics=[]; assignment=[]; previous={}; dynamics=[]
    for m in months:
        x,w=xs[m],graphs[m]
        for method in cl['methods']:
            prev=previous.get(method)
            y=fit(method,x,w,k,seed,prev,cl['graph_weight'],cl['time_weight'],cl['max_iter'])
            if prev is None:
                medians=[panel.loc[m].loc[ids[y==c],'Все категории'].median() for c in range(k)]
                order=np.argsort(medians); mapping=np.argsort(order); y=mapping[y]
            score=evaluate(x,w,y,seed,ev['silhouette_sample'])
            score.update(month=m,method=method,phase='calibration' if m<=cfg['calibration_end'] else 'evaluation')
            if prev is not None:
                score['ARI_previous']=adjusted_rand_score(prev,y)
                score['switch_rate']=float(np.mean(prev!=y))
            metrics.append(score); labels[(m,method)]=y
            if method==cl['main_method']:
                assignment.extend(dict(month=m,territory_id=int(t),cluster=int(c)+1) for t,c in zip(ids,y))
                if prev is not None:
                    for a in range(k):
                        for b in range(k):
                            dynamics.append(dict(month=m,source=a+1,target=b+1,count=int(((prev==a)&(y==b)).sum())))
            previous[method]=y
        log(f'Готов месяц {m}')
    metrics=pd.DataFrame(metrics); metrics.to_csv(out/'monthly_metrics.csv',index=False)
    summary=metrics[metrics.phase=='evaluation'].groupby('method').mean(numeric_only=True)
    summary.to_csv(out/'method_comparison.csv')
    assignments=pd.DataFrame(assignment)
    assignments.to_csv(out/'assignments.csv',index=False)
    transitions=pd.DataFrame(dynamics); transitions.to_csv(out/'transitions.csv',index=False)
    hist=assignments.pivot(index='territory_id',columns='month',values='cluster').reindex(ids)
    switches=(hist.iloc[:,1:].to_numpy()!=hist.iloc[:,:-1].to_numpy()).sum(1)
    churn=info[['municipal_district_name_short','region_name']].copy(); churn['switches']=switches
    churn.to_csv(out/'municipality_switches.csv',encoding='utf-8-sig')
    m=months[-1]; x=xs[m]; w=graphs[m]; y=labels[(m,cl['main_method'])]
    final=panel.loc[m].join(info); final['cluster']=y+1
    final.to_csv(out/'final_profiles.csv',encoding='utf-8-sig')
    profile_columns=['Все категории']+['share_'+c for c in COMPOSITION]+['salary','population','market_access']
    medians=final.groupby('cluster')[profile_columns].median()
    medians.insert(0,'count',final.groupby('cluster').size())
    medians.to_csv(out/'cluster_profiles.csv',encoding='utf-8-sig')
    # Внешняя проверка: перестановочный тест различий log1p-индикаторов.
    rng=np.random.default_rng(seed)
    external=[]
    for column in ['salary','population','market_access']:
        v=final[column].to_numpy(); valid=np.isfinite(v)&(v>0); z=np.log1p(v[valid]); yy=y[valid]
        mean=z.mean(); total=((z-mean)**2).sum()
        def eta(lab):
            return sum(np.sum(lab==c)*(z[lab==c].mean()-mean)**2 for c in np.unique(lab))/total
        value=eta(yy); null=[eta(rng.permutation(yy)) for _ in range(999)]
        external.append(dict(indicator=column,n=int(valid.sum()),eta_squared=float(value),permutation_p=(1+sum(v>=value for v in null))/1000))
    pd.DataFrame(external).to_csv(out/'external_validation.csv',index=False)
    # Контрольный случайный результат сохраняет размеры кластеров.
    null=[]
    for r in range(ev['random_repeats']):
        yy=rng.permutation(y)
        null.append(dict(repeat=r,**evaluate(x,w,yy,seed,ev['silhouette_sample'])))
    pd.DataFrame(null).to_csv(out/'random_baseline.csv',index=False)
    # Subsampling: и граф, и модель строятся заново на 80% МО, без временного якоря.
    stability=[]
    for method in cl['methods']:
        base=fit(method,x,w,k,seed,graph_weight=cl['graph_weight'],time_weight=0,max_iter=cl['max_iter'])
        for r in range(ev['stability_repeats']):
            chosen_ids=np.sort(rng.choice(n,int(n*ev['subsample_fraction']),replace=False))
            xx=x[chosen_ids]; ww=build_graph(xx,**cfg['graph'])
            yy=fit(method,xx,ww,k,seed+r,graph_weight=cl['graph_weight'],time_weight=0,max_iter=cl['max_iter'])
            stability.append(dict(method=method,repeat=r,ARI=adjusted_rand_score(base[chosen_ids],yy)))
        log(f'Устойчивость: {method}')
    pd.DataFrame(stability).to_csv(out/'stability.csv',index=False)
    # Разрежение, правило расстояния и вклад сети на одном и том же срезе.
    sensitivity=[]
    for neighbor in [10,20,40]:
        for mutual,metric in [(False,'euclidean'),(True,'euclidean'),(False,'cosine')]:
            ww=build_graph(x,neighbors=neighbor,mutual=mutual,metric=metric)
            yy=fit(cl['main_method'],x,ww,k,seed,graph_weight=cl['graph_weight'],time_weight=0,max_iter=cl['max_iter'])
            sensitivity.append(dict(neighbors=neighbor,mutual=mutual,metric=metric,graph_weight=cl['graph_weight'],ARI_main=adjusted_rand_score(y,yy),**graph_stats(ww),**evaluate(x,ww,yy,seed,ev['silhouette_sample'])))
    for weight in [0,0.25,1.0]:
        yy=fit(cl['main_method'],x,w,k,seed,graph_weight=weight,time_weight=0,max_iter=cl['max_iter'])
        sensitivity.append(dict(neighbors=cfg['graph']['neighbors'],mutual=False,metric='euclidean',graph_weight=weight,ARI_main=adjusted_rand_score(y,yy),**graph_stats(w),**evaluate(x,w,yy,seed,ev['silhouette_sample'])))
    pd.DataFrame(sensitivity).to_csv(out/'sensitivity.csv',index=False)
    # Временная абляция: вся последовательность без штрафа за смену группы.
    ablation=[]; prev=None
    for m in months:
        yy=fit(cl['main_method'],xs[m],graphs[m],k,seed,prev,cl['graph_weight'],0,cl['max_iter'])
        row=dict(month=m,ARI_main=adjusted_rand_score(labels[(m,cl['main_method'])],yy),SW=evaluate(xs[m],graphs[m],yy,seed,ev['silhouette_sample'])['SW'])
        if prev is not None: row['switch_rate']=float(np.mean(prev!=yy))
        ablation.append(row); prev=yy
    pd.DataFrame(ablation).to_csv(out/'temporal_ablation.csv',index=False)
    (out/'config_used.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    (out/'run.json').write_text(json.dumps(dict(k=k,main_method=cl['main_method'],seed=seed,
        python=platform.python_version(),packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','matplotlib','pyarrow']},
        months=months,municipalities=n),indent=2),encoding='utf-8')
    log('Расчет завершен. Таблицы и сети сохранены в '+str(out))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='configs/default.yaml'); args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    with threadpool_limits(limits=1): run(cfg)
