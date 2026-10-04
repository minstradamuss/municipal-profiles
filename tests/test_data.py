from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from municipal.data import prepare,check_snapshot,COMPOSITION


def test_input_snapshot_and_panel_invariants():
    cfg=yaml.safe_load(Path('configs/default.yaml').read_text(encoding='utf-8'))
    check_snapshot(cfg['data_dir'])
    p,x,info,excluded,q=prepare(cfg)
    assert q['raw_rows']==303126
    assert q['municipalities']==2016
    assert q['months']==24
    assert p.shape[0]==2016*24
    assert np.allclose(p[['share_'+c for c in COMPOSITION]].sum(axis=1),1)
    assert all(np.isfinite(v).all() for v in x.values())
    assert info.index.is_unique
    assert not set(excluded.territory_id)&set(info.index)


def test_saved_outputs_are_complete():
    path=Path('results/assignments.csv')
    if not path.exists():
        return  # Интеграционный контроль активируется после полного запуска.
    a=pd.read_csv(path)
    assert not a.duplicated(['month','territory_id']).any()
    assert a.groupby('month').size().eq(2016).all()
    assert a.groupby('territory_id').size().eq(24).all()
    assert a.cluster.min()==1
