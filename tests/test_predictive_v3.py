"""V3 causal fitting, shrinkage arithmetic, calibrated inference and safe loading."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from tests.test_predictive_v2_independent import small_daily_frame, complete_history
from vol_surface import predictive_v3 as v3
from vol_surface import predictive_v2 as v2


@pytest.fixture(scope='module')
def fitted():
    frame, names, dates = small_daily_frame()
    parts = [frame.loc[frame._calendar_index < 240],
             frame.loc[frame._calendar_index.between(246,325)],
             frame.loc[frame._calendar_index.between(332,411)]]
    models = v3.fit_models(*parts, names)
    return frame, names, parts, models


def test_affine_calibration_arithmetic_and_mixture_survival_product(fitted):
    frame,names,_,models = fitted
    sub=frame.loc[(frame.prediction_task=='onset') & (frame._calendar_index>=418)]
    task=models['tasks']['onset']
    for mode,c in task['candidate_models'].items():
        cal=c['calibration']
        raw=v2._candidate_logits(sub,names,models['preprocessing'],c)
        hz=1/(1+np.exp(-(cal['intercept']+cal['slope']*raw+sub.symbol.map(cal['symbol_offsets']).to_numpy()[:,None])))
        np.testing.assert_allclose(hz,v3._hazards(sub,names,models['preprocessing'],c),rtol=1e-14)
        if mode!='hazard_baseline':
            base=v3._hazards(sub,names,models['preprocessing'],task['candidate_models']['hazard_baseline'])
            for weight in v3.WEIGHTS:
                pred=v3.predict_curves(sub,names,models,mode=mode,weight=weight)
                expected=1-np.cumprod(1-((1-weight)*base+weight*hz),axis=1)
                np.testing.assert_allclose(expected,pred['probabilities'],rtol=1e-14)
                assert (np.diff(expected,axis=1)>=0).all()


def test_calibration_is_bounded_and_same_policy_for_baseline(fitted):
    for task in fitted[-1]['tasks'].values():
        for c in task['candidate_models'].values():
            cal=c['calibration']
            assert cal['applied'] and 0<=cal['slope']<=1
            assert abs(cal['intercept'])<=4
            assert all(abs(x)<=4 for x in cal['symbol_offsets'].values())
        assert task['blend_weight'] in (0.,.25,.5)
        assert len(set(task['mode_by_symbol'].values()))==1
        assert all(task['thresholds_by_symbol']['ag'][i]==task['thresholds_by_symbol']['cu'][i] for i in range(5))


def test_reserved_test_labels_and_features_cannot_alter_fit(fitted):
    frame,names,parts,models=fitted
    changed=frame.copy()
    changed.loc[changed._calendar_index>=418,names]=1e6
    changed.loc[changed._calendar_index>=418,[f'label_{h}' for h in range(1,6)]]=0
    refit=v3.fit_models(changed.loc[parts[0].index],changed.loc[parts[1].index],changed.loc[parts[2].index],names)
    assert refit==models


def test_fit_rejects_overlap_and_unknown_task(fitted):
    _,names,parts,_=fitted
    bad=parts[0].copy();bad.iloc[-1,bad.columns.get_loc('label_end_5')]=parts[1]._dt.min()
    with pytest.raises(ValueError,match='overlap'):
        v3.fit_models(bad,parts[1],parts[2],names)
    bad=parts[0].copy();bad['prediction_task']='unknown'
    with pytest.raises(ValueError,match='unknown'):
        v3.fit_models(bad,parts[1],parts[2],names)


def test_selection_margin_requires_support_and_baseline_wins_ties():
    n=100
    f=pd.DataFrame({'_dt':pd.bdate_range('2023-01-02',periods=n), '_first_event':np.tile([0,1],n//2)})
    for h in range(1,6): f[f'label_{h}']=np.tile([0,1],n//2)
    base=np.full((n,5),.5)
    best=np.repeat(f.label_1.to_numpy()[:,None],5,axis=1)*.8+.1
    choices={('hazard_baseline',0.):base,('joint_logistic',.25):best}
    selected,record=v3._selection(f,choices)
    assert selected==('joint_logistic',.25) and record['supported']
    assert v3._selection(f,{('hazard_baseline',0.):base,('joint_logistic',.25):base})[0]==('hazard_baseline',0.)
    short=f.iloc[:20]
    assert v3._selection(short,{k:x[:20] for k,x in choices.items()})[0]==('hazard_baseline',0.)


def test_same_date_symbols_share_validation_blocks():
    dates=pd.bdate_range('2023-01-02',periods=40)
    f=pd.DataFrame({'_dt':np.repeat(dates,2),'_first_event':np.tile([0,1],40)})
    for h in range(1,6):f[f'label_{h}']=np.tile([0,1],40)
    base=np.full((80,5),.5)
    _,r=v3._selection(f,{('hazard_baseline',0.):base},dates)
    assert r['nonoverlapping_date_blocks']==4


def test_no_calibration_event_support_uses_identity(fitted):
    frame,names,parts,models=fitted
    empty=parts[1].iloc[:0]
    candidate=models['tasks']['onset']['candidate_models']['hazard_baseline']
    cal=v3._fit_calibration(empty,names,models['preprocessing'],candidate,models['symbols'])
    assert not cal['applied'] and cal['slope']==1 and cal['intercept']==0


def test_quality_gate_covers_whole_curve_and_baseline_emits_no_alert(fitted):
    frame,names,_,models=fitted
    row=frame.iloc[[-1]].copy()
    row[['atm_iv','convexity_violation','arb_score','liquidity_ratio']]=1e8
    assert not v3.operational_gates(row,names,models)['allowed'].any()
    normal=v3.predict_curves(frame.iloc[-40:],names,models,mode='hazard_baseline',weight=0.)
    assert not normal['alerts'].any()
    with pytest.raises(ValueError):v3.predict_curves(row,names,models,weight=.9)


@pytest.mark.parametrize('problem',['slope','weight','coefficients','scale','schema','missing_field','null_calibration','negative_threshold','mode_disagreement'])
def test_loader_rejects_invalid_numeric_artifacts(fitted,tmp_path,problem):
    _,names,_,models=fitted
    a={'schema_version':3,'feature_names':names,'symbols':models['symbols'],'models':copy.deepcopy(models)}
    if problem=='slope':a['models']['tasks']['onset']['candidate_models']['hazard_baseline']['calibration']['slope']=1.5
    elif problem=='weight':a['models']['tasks']['onset']['blend_weight']=.7
    elif problem=='coefficients':a['models']['tasks']['onset']['candidate_models']['joint_logistic']['coefficients']=[]
    elif problem=='scale':a['models']['preprocessing']['scale'][0]=0
    elif problem=='missing_field':del a['models']['tasks']['onset']['candidate_models']['hazard_baseline']['calibration']['slope']
    elif problem=='null_calibration':a['models']['tasks']['onset']['candidate_models']['hazard_baseline']['calibration']=None
    elif problem=='negative_threshold':a['models']['tasks']['onset']['thresholds_by_symbol']['ag'][0]['value']=-.5
    elif problem=='mode_disagreement':a['models']['tasks']['onset']['mode_by_symbol']['ag']='unknown'
    else:a['schema_version']=True
    path=tmp_path/'model.json';path.write_text(json.dumps(a))
    with pytest.raises(ValueError):v3.load_artifact(path)


def test_forecast_prefix_invariance_runtime_arithmetic_and_safety(fitted,tmp_path):
    _,names,_,models=fitted
    a={'schema_version':3,'feature_names':names,'symbols':models['symbols'],'models':models,
       'model_available_from':'2023-01-01','target_description':v3.TARGET_DESCRIPTION}
    path=tmp_path/'model.json';path.write_text(json.dumps(a))
    a=v3.load_artifact(path)
    raw=complete_history(140)
    cutoff=pd.bdate_range('2023-01-02',periods=140)[119]+pd.Timedelta(hours=23)
    first=v3.forecast_latest(a,raw,'ag',as_of=cutoff)
    changed=raw.copy();future=pd.to_datetime(changed._timestamp,format='%Y%m%d%H%M%S')>cutoff
    changed.loc[future,'atm_iv']=100
    second=v3.forecast_latest(a,changed,'ag',as_of=cutoff)
    assert first==second and len(first['horizons'])==5
    assert first['model_version']=='v3'
    assert first['prediction_task']=='onset'
    assert first['horizons'][0]['top_contributions']==[]
    features,_=v3.build_features(raw.loc[~future],models['symbols'])
    row=features[features.symbol=='ag'].iloc[[-1]]
    expected=v3.predict_curves(row,names,models)
    np.testing.assert_allclose([h['probability'] for h in first['horizons']],expected['probabilities'][0])
    assert not v3.forecast_latest(a,raw,'ag',as_of='2026-01-01')['horizons']
    assert not v3.forecast_latest(a,raw,'ag',as_of='2022-01-01')['horizons']
    assert not v3.forecast_latest(a,raw,'unknown',as_of=cutoff)['horizons']
