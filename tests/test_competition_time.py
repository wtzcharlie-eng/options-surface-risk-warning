import numpy as np
import pandas as pd
import pytest
from vol_surface.time_utils import epoch_seconds,epoch_minutes
from drl.metrics import evaluate_alerts
from drl.metrics_v2 import _epoch as epoch_v2
from drl.dataset import _lead_minutes,session_dead_zone


def test_wall_clock_window_does_not_compress_days_into_minutes():
    ts=['20260102100000','20260102101500','20260105100000']
    assert np.diff(epoch_seconds(ts)).tolist()==[900,258300]
    assert np.diff(epoch_minutes(ts)).tolist()==[15,4305]
    assert np.array_equal(epoch_seconds(ts),epoch_v2(ts))
    assert evaluate_alerts(ts,np.array([2,0,0]),[2]).precision==0
    m=evaluate_alerts(ts,np.array([2,0,0]),[1])
    assert m.precision==1 and m.recall==1 and m.avg_lead_min==15


@pytest.mark.parametrize('unit',['s','ms','us','ns'])
def test_datetime_resolution_cannot_change_lead_minutes(unit):
    dt=pd.Series(np.array(['2026-01-02T10:00','2026-01-02T10:15','2026-01-02T12:00'],dtype=f'datetime64[{unit}]'))
    np.testing.assert_allclose(_lead_minutes(dt,[2]),[120,105,0])


def test_dead_zone_keeps_lunch_and_detects_overnight():
    ts=['20260102113000','20260102134500','20260102150000','20260105090000']
    assert session_dead_zone(ts,k_bars=1).tolist()==[False,False,True,False]
    assert np.array_equal(epoch_seconds([s.encode() for s in ts]),epoch_seconds(ts))
    with pytest.raises(ValueError):epoch_seconds(['not-a-date'])
