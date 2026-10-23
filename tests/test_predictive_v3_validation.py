import numpy as np
import pandas as pd
import pytest
from vol_surface.predictive_v3_validation import reliability, _decisions, compare_frozen_v2


def predictions():
    rows=[]
    for i,day in enumerate(pd.bdate_range('2025-01-02',periods=4)):
        for h in range(1,6):
            rows.append(dict(fold=1,symbol='ag',_dt=day,horizon_days=h,label=i%2,
                             probability=.3+.3*(i%2),baseline_probability=.5,
                             operational_alert=bool(i%2),raw_alert=bool(i%2),
                             gate_allowed=i!=1,gate_reason='out_of_distribution' if i==1 else None,
                             model_status='joint_logistic_shrunk',calendar_index=i))
    frame=pd.DataFrame(rows)
    frame.operational_alert &= frame.gate_allowed
    return frame


def test_fixed_reliability_bins_include_endpoints_and_manual_log_loss():
    f=pd.DataFrame({'label':[0,1,0,1],'probability':[0.,1.,.2,.8]})
    r=reliability(f)
    assert sum(b['n'] for b in r['reliability_bins'])==4
    assert r['mean_probability']==.5
    assert abs(r['calibration_bias'])<1e-14
    np.testing.assert_allclose(r['log_loss'],-np.log(.8)/2,atol=1e-10)
    assert reliability(f.iloc[:0])['mean_probability'] is None


def test_shrunk_model_operational_coverage_and_abstention_denominator():
    f=predictions().query('horizon_days==1')
    r=_decisions(f,'operational_alert',True)
    assert r['predictive_signal_coverage']==.75
    assert r['recall_all_origins']==.5 and r['abstentions']==1
    f.loc[f.index[-1],'model_status']='hazard_baseline'
    assert _decisions(f,'operational_alert',True)['predictive_signal_coverage']==.5


@pytest.mark.parametrize('corruption',[None,'missing_origin','changed_label','duplicate'])
def test_comparison_requires_exact_frozen_pairs(tmp_path,corruption):
    old=predictions();new=old.copy()
    if corruption=='missing_origin':new=new.iloc[:-1]
    elif corruption=='changed_label':new.loc[0,'label']=1
    elif corruption=='duplicate':new=pd.concat([new,new.iloc[:1]])
    path=tmp_path/'v2.parquet';old.to_parquet(path,index=False)
    if corruption:
        with pytest.raises(ValueError):compare_frozen_v2(new,path,n_boot=10)
    else:
        report=compare_frozen_v2(new,path,n_boot=10)
        assert report['status']=='paired_all_origins'
        assert report['folds']['1']['1']['paired_v3_vs_v2']['skill']==0
