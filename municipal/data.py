"""Проверки исходных данных, панель и экономически интерпретируемые атрибуты."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

CATEGORIES=['Продовольствие','Здоровье','Маркетплейсы','Общественное питание','Транспорт']
COMPOSITION=CATEGORIES+['Прочие расходы']


def check_snapshot(raw_dir):
    manifest=json.loads((Path(raw_dir).parent/'sources.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path=Path(raw_dir)/item['file']
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=item['sha256']:
            raise ValueError(f'Контрольная сумма не совпадает: {path}')


def prepare(cfg):
    rawdir=Path(cfg['data_dir'])
    check_snapshot(rawdir)
    raw=pd.read_parquet(rawdir/'8_consumption.parquet')
    required={'date','territory_id','category','value'}
    if not required.issubset(raw): raise ValueError(f'Нужны колонки {required}')
    if raw.duplicated(['date','territory_id','category']).any():
        raise ValueError('Неоднозначные дубликаты МО × месяц × категория')
    raw['date']=pd.to_datetime(raw.date,format='%Y-%m').dt.strftime('%Y-%m')
    raw=raw[raw.date.between(cfg['start'],cfg['end'])]
    if raw['value'].isna().any() or (raw['value']<0).any():
        raise ValueError('Пропуски или отрицательные расходы в исходных строках')
    months=pd.period_range(cfg['start'],cfg['end'],freq='M').astype(str).tolist()
    p=raw.pivot(index=['date','territory_id'],columns='category',values='value')
    p=p.reindex(columns=CATEGORIES+['Все категории'])
    p['Прочие расходы']=p['Все категории']-p[CATEGORIES].sum(1,min_count=len(CATEGORIES))
    valid=p.notna().all(1)&(p['Все категории']>0)&(p['Прочие расходы']>=0)
    counts=valid.groupby('territory_id').sum()
    ids=np.sort(counts[counts==len(months)].index.to_numpy())
    if len(ids)<20: raise ValueError('Недостаточно полных рядов для кластеризации')
    excluded=pd.DataFrame({'territory_id':counts.index,'valid_months':counts.values})
    excluded=excluded[~excluded.territory_id.isin(ids)]
    index=pd.MultiIndex.from_product([months,ids],names=['date','territory_id'])
    panel=p.reindex(index).copy()
    if panel.isna().any().any(): raise ValueError('Панель содержит отсутствующие месяцы')
    shares=panel[COMPOSITION].div(panel['Все категории'],axis=0)
    for c in COMPOSITION: panel['share_'+c]=shares[c]
    # CLR после аддитивного сглаживания. Остаточная категория закрывает композицию.
    amounts=panel[COMPOSITION].to_numpy()+cfg['features']['pseudocount']
    logparts=np.log(amounts/amounts.sum(1,keepdims=True))
    clr=logparts-logparts.mean(1,keepdims=True)
    loglevel=np.log(panel['Все категории'])
    relative=loglevel-loglevel.groupby(level='date').transform('median')
    features=np.column_stack([clr,relative])
    cal=np.asarray(panel.index.get_level_values('date')<=cfg['calibration_end'])
    scaler=RobustScaler(quantile_range=(25,75)).fit(features[cal])
    scaled=scaler.transform(features)
    clip=cfg['features']['clip']; scaled=np.clip(scaled,-clip,clip)
    scaled[:,:len(COMPOSITION)]/=np.sqrt(len(COMPOSITION))
    scaled[:,-1]*=cfg['features']['level_weight']
    attributes={m:scaled[i*len(ids):(i+1)*len(ids)] for i,m in enumerate(months)}
    names=pd.read_excel(rawdir/'t_dict_municipal_districts.xlsx')
    # year_to — год окончания версии; последняя начавшаяся версия имеет приоритет.
    end_year=int(cfg['end'][:4])
    active=names[(names.year_from<=end_year)&(names.year_to>=end_year)]
    active=active.sort_values(['territory_id','year_from','year_to']).drop_duplicates('territory_id',keep='last')
    info=active.set_index('territory_id').reindex(ids)
    info['municipal_district_name_short']=info.municipal_district_name_short.fillna(pd.Series(ids,index=ids).map(lambda i:f'МО {i}'))
    info['region_name']=info.region_name.fillna('Регион не указан')
    versions=names[names.territory_id.isin(ids)&(names.year_from<=end_year)&(names.year_to>=int(cfg['start'][:4]))]
    info['dictionary_versions']=versions.groupby('territory_id').size().reindex(ids).fillna(0).astype(int)
    # Внешние признаки используются только после кластеризации.
    salary=pd.read_parquet(rawdir/'4_bdmo_salary.parquet')
    salary=salary[(salary.year==end_year)&(salary.period=='январь-декабрь')&(salary.okved_letter=='0')]
    salary_duplicates=int(salary.duplicated().sum()); salary=salary.drop_duplicates()
    if salary.duplicated('territory_id').any(): raise ValueError('Дубликаты годовой зарплаты')
    info['salary']=salary.set_index('territory_id').value.reindex(ids)
    population=pd.read_parquet(rawdir/'2_bdmo_population.parquet')
    population=population[(population.year==end_year)&(population.age=='Всего')]
    population_duplicates=int(population.duplicated().sum()); population=population.drop_duplicates()
    if population.duplicated(['territory_id','gender']).any(): raise ValueError('Дубликаты населения')
    info['population']=population.groupby('territory_id').value.sum(min_count=2).reindex(ids)
    access=pd.read_parquet(rawdir/'1_market_access.parquet').set_index('territory_id')
    info['market_access']=access.market_access.reindex(ids)
    quality=dict(raw_rows=int(len(raw)),raw_municipalities=int(raw.territory_id.nunique()),
        municipalities=int(len(ids)),months=len(months),panel_rows=int(len(panel)),
        excluded_municipalities=int(len(excluded)),missing_cells=int(p.isna().sum().sum()),
        negative_residuals=int((p['Прочие расходы']<0).sum()),
        dictionary_multi_version=int((info.dictionary_versions>1).sum()),
        salary_exact_duplicates_removed=salary_duplicates,population_exact_duplicates_removed=population_duplicates,
        calibration_end=cfg['calibration_end'],start=months[0],end=months[-1],
        feature_center=scaler.center_.tolist(),feature_scale=scaler.scale_.tolist(),
        feature_names=['clr_'+c for c in COMPOSITION]+['relative_log_spending'])
    return panel,attributes,info,excluded,quality
