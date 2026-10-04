"""Портреты, примеры и дополнительные проверки уже рассчитанной типологии."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score
from municipal.data import prepare,COMPOSITION


def main():
    cfg=yaml.safe_load(Path('results/config_used.yaml').read_text(encoding='utf-8'))
    panel,xs,info,_,_=prepare(cfg)
    out=Path(cfg['output_dir']); run=json.loads((out/'run.json').read_text())
    final=pd.read_csv(out/'final_profiles.csv').set_index('territory_id').reindex(info.index)
    x=xs[run['months'][-1]]; y=final.cluster.to_numpy()
    examples=[]
    for c in sorted(np.unique(y)):
        ids=np.flatnonzero(y==c); center=x[ids].mean(0)
        dist=np.linalg.norm(x[ids]-center,axis=1)
        for position in np.argsort(dist)[:5]:
            row=final.iloc[ids[position]]
            examples.append(dict(cluster=int(c),territory_id=int(row.name),name=row.municipal_district_name_short,
                region=row.region_name,distance_to_center=float(dist[position]),spending=float(row['Все категории'])))
    pd.DataFrame(examples).to_csv(out/'representative_municipalities.csv',index=False,encoding='utf-8-sig')
    a=pd.read_csv(out/'assignments.csv').pivot(index='territory_id',columns='month',values='cluster').reindex(info.index)
    stable=(info.dictionary_versions==1).to_numpy()
    rows=[]
    for mask,name in [(np.ones(len(info),bool),'all'),(stable,'single_dictionary_version')]:
        z=a.to_numpy()[mask]
        for t in range(1,z.shape[1]):
            rows.append(dict(sample=name,n=int(mask.sum()),month=a.columns[t],switch_rate=float(np.mean(z[:,t]!=z[:,t-1])),ARI_previous=adjusted_rand_score(z[:,t-1],z[:,t])))
    pd.DataFrame(rows).to_csv(out/'dictionary_sensitivity.csv',index=False)
    yearly=[]
    for category in COMPOSITION:
        s=panel['share_'+category].unstack('territory_id')
        for month in range(1,13):
            before=f'2023-{month:02d}'; after=f'2024-{month:02d}'
            delta=s.loc[after]-s.loc[before]
            yearly.append(dict(category=category,month_of_year=month,median_2023=float(s.loc[before].median()),median_2024=float(s.loc[after].median()),median_change_pp=float(delta.median()*100)))
    pd.DataFrame(yearly).to_csv(out/'same_month_changes.csv',index=False,encoding='utf-8-sig')
    # Изменения отдельных МО: ранжируем по расстоянию между декабрями,
    # чтобы не сравнивать сезонно разные месяцы.
    before=xs['2023-12']; after=xs['2024-12']
    change=np.linalg.norm(after-before,axis=1)
    cases=info[['municipal_district_name_short','region_name','dictionary_versions']].copy()
    cases['distance_december']=change
    cases['cluster_2023_12']=a['2023-12']; cases['cluster_2024_12']=a['2024-12']
    cases['marketplace_change_pp']=(panel.loc['2024-12']['share_Маркетплейсы']-panel.loc['2023-12']['share_Маркетплейсы'])*100
    cases.sort_values('distance_december',ascending=False).to_csv(out/'change_cases.csv',encoding='utf-8-sig')
    print('Портреты и дополнительные проверки сохранены.')


if __name__=='__main__': main()
