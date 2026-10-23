"""Read-only verification of published V3 scores and last-vintage coefficients.

Uses frozen V2 feature construction, but does not call V3 prediction, calibration,
selection or metric helpers. No fitting or artifact mutation occurs.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vol_surface.predictive import load_anchor_data
from vol_surface.predictive_v2 import build_features


def main():
    folder=ROOT/'data_out/predictive_v3'
    report=json.loads((folder/'report.json').read_text())
    artifact=json.loads((folder/'model.json').read_text())
    assert hashlib.sha256((folder/'model.json').read_bytes()).hexdigest()==report['model_sha256']
    for name,expected in report['algorithm_source_sha256'].items():
        assert hashlib.sha256((ROOT/'vol_surface'/name).read_bytes()).hexdigest()==expected,name
    for name,expected in [('scripts/train_predictive_v3.py',report['training_script_sha256']),
                          ('docs/V3实验设计.md',report['design_sha256']),
                          ('data_out/predictive_v2/test_predictions.parquet',report['comparison_v2']['v2_predictions_sha256'])]:
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    p=pd.read_parquet(folder/'test_predictions.parquet')
    old=pd.read_parquet(ROOT/'data_out/predictive_v2/test_predictions.parquet')
    keys=['fold','symbol','_dt','horizon_days']
    paired=p.merge(old[keys+['label','probability']],on=keys,suffixes=('','_v2'),validate='one_to_one')
    assert len(paired)==len(p)==len(old) and np.array_equal(paired.label,paired.label_v2)
    for fold in report['folds']:
        for h,metrics in fold['horizons'].items():
            rows=p[(p.fold==fold['fold'])&(p.horizon_days==int(h))]
            y,prob,base=rows.label.to_numpy(),rows.probability.to_numpy(),rows.baseline_probability.to_numpy()
            brier,base_brier=np.mean((y-prob)**2),np.mean((y-base)**2)
            np.testing.assert_allclose([brier,base_brier,1-brier/base_brier],
                [metrics['test']['brier'],metrics['test']['baseline_brier'],metrics['test']['brier_skill']],rtol=1e-12)
            tp=int((rows.operational_alert & rows.label.eq(1)).sum())
            fp=int((rows.operational_alert & rows.label.eq(0)).sum())
            op=metrics['operational']
            assert fp==op['false_alarms'] and int(rows.operational_alert.sum())==op['alerts']
            assert abs(tp/rows.label.sum()-op['recall_all_origins'])<1e-12
            assert abs(rows.gate_allowed.mean()-op['probability_coverage'])<1e-12
        wide=p[p.fold==fold['fold']].pivot(index=['symbol','_dt'],columns='horizon_days',values='probability')
        assert wide.notna().all().all() and (np.diff(wide.to_numpy(),axis=1)>=-1e-14).all()
    history=load_anchor_data(ROOT/'data_out/anchor')
    features,_=build_features(history,artifact['symbols'])
    latest=p[p.fold==max(p.fold)]
    origins=latest[['symbol','_dt','prediction_task']].drop_duplicates().merge(features,on=['symbol','_dt'],validate='one_to_one')
    names,models=artifact['feature_names'],artifact['models']
    prep=models['preprocessing']
    raw=origins[names].to_numpy(float)
    filled=np.where(np.isfinite(raw),raw,np.asarray(prep['median']))
    x=np.clip((filled-np.asarray(prep['median']))/np.asarray(prep['scale']),-prep['clip'],prep['clip'])
    reconstructed=np.zeros((len(origins),5))
    for task,model in models['tasks'].items():
        index=np.flatnonzero(origins.prediction_task.to_numpy()==task)
        symbols=origins.iloc[index].symbol
        def hazards(c):
            if c['mode']=='hazard_baseline':
                hz=np.array([c['hazards_by_symbol'].get(s,c['pooled_hazards']) for s in symbols])
                z=np.log(hz/(1-hz))
            else:
                co=np.asarray(c['coefficients'])
                z=c['intercept']+x[index]@co[:len(names)]
                z=z[:,None]+np.r_[0.,co[len(names):]][None,:]
            cal=c['calibration']
            return expit(np.clip(cal['intercept']+cal['slope']*z+symbols.map(cal['symbol_offsets']).to_numpy()[:,None],-40,40))
        base=hazards(model['candidate_models']['hazard_baseline'])
        feat=hazards(model['candidate_models'][model['selected_candidate']])
        alpha=model['blend_weight']
        reconstructed[index]=1-np.cumprod(1-((1-alpha)*base+alpha*feat),axis=1)
    actual=latest.pivot(index=['symbol','_dt'],columns='horizon_days',values='probability').reindex(pd.MultiIndex.from_frame(origins[['symbol','_dt']])).to_numpy()
    np.testing.assert_allclose(reconstructed,actual,atol=1e-12,rtol=1e-12)
    # Recompute all last-vintage frozen-origin labels directly from daily maxima.
    daily=features.sort_values(['symbol','_dt']).copy()
    for h in range(1,6):
        event=np.zeros(len(daily),bool)
        for step in range(1,h+1):
            event |= ((daily.groupby('symbol').atm_iv_daily_max.shift(-step)>daily._atm_iv_q95)|
                      (daily.groupby('symbol').convexity_violation_daily_max.shift(-step)>daily._convexity_violation_q95)).to_numpy()
        expected=daily[['symbol','_dt']].assign(manual_label=event.astype(int)).merge(latest[latest.horizon_days==h][['symbol','_dt','label']],on=['symbol','_dt'],validate='one_to_one')
        assert np.array_equal(expected.manual_label,expected.label)
    print(f"V3 source/model/design checksums, {len(p):,} paired rows, all fold scores/operational counts, {len(latest):,} final labels verified")
    print(f"Manual coefficient/calibration/mixture reconstruction max error: {np.max(np.abs(reconstructed-actual)):.3g}")
    return 0


if __name__=='__main__':
    raise SystemExit(main())
