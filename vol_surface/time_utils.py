"""Explicit epoch units across pandas datetime resolutions.

Pandas 3 can infer microseconds where older pandas inferred nanoseconds. Casting
to int64 before fixing the unit silently changes every wall-clock window.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def epoch_seconds(values) -> np.ndarray:
    data=[v.decode('utf-8') if isinstance(v,(bytes,np.bytes_)) else v for v in values]
    dt=pd.to_datetime(pd.Series(data),format='%Y%m%d%H%M%S',errors='raise')
    if dt.isna().any():
        raise ValueError('Missing timestamp cannot enter a wall-clock risk window')
    return dt.to_numpy(dtype='datetime64[s]').astype(np.int64)


def epoch_minutes(values) -> np.ndarray:
    return epoch_seconds(values)//60
