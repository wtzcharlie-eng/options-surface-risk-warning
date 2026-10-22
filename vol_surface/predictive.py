"""Causal 1--5 observed-trading-day option-surface risk-proxy forecasts.

The target is *not* a portfolio loss probability: it is whether ATM IV or the
surface convexity-violation daily maximum exceeds its strictly-past 60-session
95th percentile on any of the next H complete observed trading days. The
threshold is frozen at the forecast origin. Missing sessions are excluded.

Persisted models contain JSON arrays only. No pickle/joblib file is loaded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HORIZONS = (1, 2, 3, 4, 5)
CADENCE_MINUTES = 15
HISTORY_WINDOW = 60
MIN_HISTORY = 20
SCHEMA_VERSION = 1
RAW_FEATURES = (
    "atm_iv", "convexity_violation", "term_slope", "skew_val",
    "gamma_concentration", "vega_concentration", "arb_score",
    "arb_calendar_n", "arb_butterfly_n", "arb_parity_n", "liquidity_ratio",
    "iv_near", "iv_far", "n_contracts",
)
LAG_FEATURES = (
    "atm_iv", "convexity_violation", "term_slope", "skew_val",
    "liquidity_ratio", "arb_score",
)
TARGET_DESCRIPTION = (
    "在已完成交易日日盘15:00收盘后，预测未来1–5个观测交易日内，"
    "任一日的ATM隐波或凸性违反代理最大值超过预测日前最近60个完整交易日"
    "（至少20日）的95%分位阈值。阈值在预测时点冻结；包括异常状态延续。"
    "交易日依据实测日盘日期，夜盘归下一观测日盘；缺日和超过7自然日断档窗口排除，"
    "不将自然日当作交易日，也不冒充官方交易所日历。"
    "这是市场曲面异常代理概率，不是持仓损失、VaR或CVaR。"
)


def _local_timestamp(value: Any = None) -> pd.Timestamp:
    ts = pd.Timestamp.now(tz="Asia/Shanghai") if value is None else pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("Asia/Shanghai").tz_localize(None)
    return ts


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    if "symbol" not in frame or "_timestamp" not in frame:
        raise ValueError("History requires symbol and _timestamp columns")
    out = frame.copy()
    # Provenance is retained by train_models separately. Carrying its large
    # manifest through thousands of pandas groups causes repeated deep copies.
    out.attrs = {}
    out["symbol"] = out["symbol"].astype(str).str.lower()
    out["_dt"] = pd.to_datetime(out["_timestamp"].astype(str),
                                format="%Y%m%d%H%M%S", errors="coerce")
    if out["_dt"].isna().any():
        raise ValueError("Unparseable timestamps; expected local YYYYMMDDHHMMSS")
    if out.duplicated(["symbol", "_dt"]).any():
        raise ValueError("Duplicate symbol/timestamp observations are ambiguous")
    for name in RAW_FEATURES:
        out[name] = pd.to_numeric(out[name], errors="coerce") if name in out else np.nan
    out[list(RAW_FEATURES)] = out[list(RAW_FEATURES)].replace([np.inf, -np.inf], np.nan)
    return out.sort_values(["symbol", "_dt"]).reset_index(drop=True)


def load_anchor_data(anchor_dir: str | Path = "data_out/anchor",
                     symbols: list[str] | None = None) -> pd.DataFrame:
    """Load raw numeric snapshots only; legacy y and fitted artifacts are ignored."""
    root = Path(anchor_dir)
    wanted = None if symbols is None else {s.lower() for s in symbols}
    rows, manifest = [], []
    for path in sorted(root.glob("*/*/features.parquet")):
        symbol = path.parent.parent.name.lower()
        if wanted is not None and symbol not in wanted:
            continue
        part = pd.read_parquet(path)
        keep = [c for c in ("_timestamp", *RAW_FEATURES) if c in part]
        part = part[keep].copy()
        part["symbol"] = symbol
        rows.append(part)
        manifest.append({"path": str(path.relative_to(root)), "rows": len(part),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "has_meta": path.with_name("meta.parquet").exists()})
    if not rows:
        raise FileNotFoundError(f"No anchor snapshots under {root}")
    out = _normalise(pd.concat(rows, ignore_index=True))
    out.attrs["source_manifest"] = manifest
    return out


def build_features(history: pd.DataFrame,
                   symbols: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate only completed sessions, then make strictly causal daily inputs."""
    out = aggregate_sessions(history)
    if out.empty:
        return out, list(RAW_FEATURES)
    syms = sorted(out.symbol.unique()) if symbols is None else list(symbols)
    grouped = out.groupby("symbol", sort=False)
    day_gap = grouped["_trading_date"].diff().dt.total_seconds().div(86400)
    calendar_gap = grouped["_calendar_index"].diff()
    out["_segment"] = ((calendar_gap != 1) | (day_gap > 7) | day_gap.isna()).cumsum()
    features = list(RAW_FEATURES)
    for name in ("atm_iv", "convexity_violation"):
        features.extend([f"{name}_daily_max", f"{name}_daily_min", f"{name}_daily_std"])
    features.extend(["liquidity_ratio_daily_min", "night_snapshots", "daily_snapshots"])
    for name in ("atm_iv", "convexity_violation"):
        daily = f"{name}_daily_max"
        out[f"_{name}_q95"] = grouped[daily].transform(
            lambda s: s.shift(1).rolling(HISTORY_WINDOW, min_periods=MIN_HISTORY).quantile(.95))
        prior_mean = grouped[daily].transform(
            lambda s: s.shift(1).rolling(HISTORY_WINDOW, min_periods=MIN_HISTORY).mean())
        prior_std = grouped[daily].transform(
            lambda s: s.shift(1).rolling(HISTORY_WINDOW, min_periods=MIN_HISTORY).std())
        out[f"{name}_past_z"] = (out[daily] - prior_mean) / prior_std.clip(lower=1e-8)
        out[f"{name}_q95_distance"] = (out[daily] - out[f"_{name}_q95"]) / prior_std.clip(lower=1e-8)
        features.extend([f"{name}_past_z", f"{name}_q95_distance"])
    out["_history_count"] = grouped["atm_iv"].transform(
        lambda s: s.shift(1).rolling(HISTORY_WINDOW, min_periods=1).count()).fillna(0)
    out["_history_ready"] = out[["_atm_iv_q95", "_convexity_violation_q95"]].notna().all(axis=1)
    out["current_exceedance"] = (
        (out.atm_iv_daily_max > out._atm_iv_q95) |
        (out.convexity_violation_daily_max > out._convexity_violation_q95)
    ).astype(float)
    features.append("current_exceedance")
    segment = out.groupby("_segment", sort=False)
    for name in LAG_FEATURES:
        lag1, lag5 = segment[name].shift(1), segment[name].shift(5)
        out[f"{name}_lag1d"] = lag1
        out[f"{name}_change1d"] = out[name] - lag1
        out[f"{name}_change5d"] = out[name] - lag5
        out[f"{name}_mean5d"] = segment[name].transform(
            lambda s: s.rolling(5, min_periods=2).mean())
        out[f"{name}_std5d"] = segment[name].transform(
            lambda s: s.rolling(5, min_periods=2).std())
        features.extend([f"{name}_{suffix}" for suffix in
                         ("lag1d", "change1d", "change5d", "mean5d", "std5d")])
    weekday = out._dt.dt.dayofweek
    out["weekday_sin"], out["weekday_cos"] = np.sin(weekday * 2*np.pi/7), np.cos(weekday * 2*np.pi/7)
    features.extend(["weekday_sin", "weekday_cos"])
    for symbol in syms:
        name = f"symbol_{symbol}"
        out[name] = (out.symbol == symbol).astype(float)
        features.append(name)
    out[features] = out[features].replace([np.inf, -np.inf], np.nan)
    return out, features


# These are observed source bar-end times in the supplied data (all seven
# commodities), rather than an invented continuous 09:00--15:00 grid.
DAYTIME_SLOTS = ("09:15", "09:30", "09:45", "10:00", "10:15", "10:45",
                 "11:00", "11:15", "11:30", "13:45", "14:00", "14:15",
                 "14:30", "14:45", "15:00")


def aggregate_sessions(history: pd.DataFrame) -> pd.DataFrame:
    """Map nights to the next observed daytime session, retaining complete days.

    A night without a subsequently observed daytime date is pending and is not
    a daily forecast origin. Appending future rows never changes an already
    completed origin. Global observed daytime dates expose missing symbol days;
    a future label may not leap over them. A >7 calendar-day gap is conservatively
    a break even when it may actually have been a long exchange holiday.
    """
    raw = _normalise(history)
    hour = raw._dt.dt.hour
    daytime = (hour >= 6) & (hour < 18)
    observed_dates = np.array(sorted(raw.loc[daytime, "_dt"].dt.normalize().unique()), dtype="datetime64[ns]")
    if not len(observed_dates):
        return pd.DataFrame()
    dates = raw._dt.dt.normalize().to_numpy(dtype="datetime64[ns]")
    query = dates.copy()
    query[hour.to_numpy() >= 18] += np.timedelta64(1, "D")
    index = np.searchsorted(observed_dates, query, side="left")
    valid_index = index < len(observed_dates)
    assigned = np.full(len(raw), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    assigned[valid_index] = observed_dates[index[valid_index]]
    mapping_gap = (assigned-dates)/np.timedelta64(1, "D")
    valid_mapping = valid_index & (mapping_gap <= 7)
    raw["_trading_date"] = assigned
    raw["_calendar_index"] = index
    raw["_daytime"] = daytime.to_numpy()
    raw["_clock"] = raw._dt.dt.strftime("%H:%M")
    raw = raw.loc[valid_mapping]
    rows, incomplete = [], {}
    for (symbol, day), sub in raw.groupby(["symbol", "_trading_date"], sort=True):
        day_rows = sub.loc[sub._daytime]
        slots = set(day_rows._clock)
        core_finite = day_rows[["atm_iv", "convexity_violation"]].notna().all(axis=1).all()
        if not set(DAYTIME_SLOTS).issubset(slots) or not core_finite:
            incomplete[symbol] = incomplete.get(symbol, 0) + 1
            continue
        sub = sub.sort_values("_dt")
        close = day_rows.sort_values("_dt").iloc[-1]
        # All observations attributed to this session are at/before its close.
        sub = sub.loc[sub._dt <= close._dt]
        if not sub[["atm_iv", "convexity_violation"]].notna().all(axis=1).all():
            incomplete[symbol] = incomplete.get(symbol, 0) + 1
            continue
        row = {name: close[name] for name in RAW_FEATURES}
        row.update({"symbol": symbol, "_dt": close._dt,
                    "_timestamp": close._dt.strftime("%Y%m%d%H%M%S"),
                    "_trading_date": day, "_calendar_index": int(close._calendar_index),
                    "daily_snapshots": len(sub), "night_snapshots": int((~sub._daytime).sum()),
                    "liquidity_ratio_daily_min": sub.liquidity_ratio.min()})
        for name in ("atm_iv", "convexity_violation"):
            row[f"{name}_daily_max"] = sub[name].max()
            row[f"{name}_daily_min"] = sub[name].min()
            row[f"{name}_daily_std"] = sub[name].std()
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values(["symbol", "_dt"]).reset_index(drop=True)
    result.attrs["calendar"] = [pd.Timestamp(d).isoformat() for d in observed_dates]
    result.attrs["incomplete_sessions_by_symbol"] = incomplete
    result.attrs["unassigned_night_snapshots"] = int((~valid_mapping).sum())
    return result


def build_targets(features: pd.DataFrame,
                  horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Attach explicit future labels and exact label_end timestamps; never use y."""
    out = features.copy()
    group = out.groupby("symbol", sort=False)
    for horizon in horizons:
        if horizon <= 0 or not isinstance(horizon, int):
            raise ValueError("Horizons must be positive integer observed trading days")
        steps = horizon
        future_end = group["_dt"].shift(-steps)
        contiguous = (group["_segment"].shift(-steps) == out._segment)
        finite = out[["atm_iv_daily_max", "convexity_violation_daily_max"]].notna().all(axis=1)
        exceeded = np.zeros(len(out), dtype=bool)
        for step in range(1, steps + 1):
            iv = group["atm_iv_daily_max"].shift(-step)
            convexity = group["convexity_violation_daily_max"].shift(-step)
            finite &= iv.notna() & convexity.notna()
            exceeded |= ((iv > out._atm_iv_q95) |
                         (convexity > out._convexity_violation_q95)).to_numpy()
        valid = contiguous & finite & out._history_ready
        out[f"label_{horizon}"] = np.where(valid, exceeded.astype(float), np.nan)
        out[f"label_end_{horizon}"] = future_end.where(valid)
        out[f"complete_window_{horizon}"] = contiguous & finite
    return out


def chronological_splits(frame: pd.DataFrame, horizon: int,
                         boundaries: dict[str, str] | None = None) -> tuple[dict, dict]:
    """Global calendar split across symbols, purging windows touching next block."""
    if boundaries is None:
        dates = np.array(sorted(frame._dt.dt.normalize().unique()))
        if len(dates) < 20:
            raise ValueError("At least 20 observed dates are required for temporal evaluation")
        boundaries = {name: pd.Timestamp(dates[min(int(len(dates)*fraction), len(dates)-1)]).isoformat()
                      for name, fraction in (("calibration", .60), ("validation", .75), ("test", .85))}
    c, v, t = (pd.Timestamp(boundaries[name]) for name in ("calibration", "validation", "test"))
    if not c < v < t:
        raise ValueError("Split boundaries must be strictly increasing")
    dt, end = frame._dt, frame[f"label_end_{horizon}"]
    valid = frame[f"label_{horizon}"].notna()
    masks = {
        "train": valid & (dt < c) & (end < c),
        "calibration": valid & (dt >= c) & (dt < v) & (end < v),
        "validation": valid & (dt >= v) & (dt < t) & (end < t),
        "test": valid & (dt >= t),
    }
    return masks, boundaries


def _preprocessor_fit(frame: pd.DataFrame, names: list[str]) -> dict:
    x = frame[names].to_numpy(dtype=float)
    median, scale, low, high = [], [], [], []
    for j in range(x.shape[1]):
        finite = x[:, j][np.isfinite(x[:, j])]
        if not len(finite):
            median.append(0.); scale.append(1.); low.append(0.); high.append(0.)
            continue
        q = np.quantile(finite, [.005, .25, .5, .75, .995])
        median.append(float(q[2])); scale.append(float(max(q[3]-q[1], 1e-6)))
        low.append(float(q[0])); high.append(float(q[4]))
    return {"median": median, "scale": scale, "low": low, "high": high, "clip": 8.0}


def _transform(frame: pd.DataFrame, names: list[str], preprocessing: dict) -> np.ndarray:
    x = frame[names].to_numpy(dtype=float)
    med = np.array(preprocessing["median"])
    x = np.where(np.isfinite(x), x, med)
    return np.clip((x-med)/np.array(preprocessing["scale"]),
                   -preprocessing["clip"], preprocessing["clip"])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -40., 40.)))


def _fit_logistic(x: np.ndarray, y: np.ndarray, penalty: float = 1e-4) -> dict:
    """Fixed ridge logistic regression; optimization uses no validation outcomes."""
    from scipy.optimize import minimize
    n = len(y)
    initial = np.zeros(x.shape[1] + 1)
    initial[0] = np.log((y.sum()+.5)/(n-y.sum()+.5))

    def objective(w):
        z = w[0] + x @ w[1:]
        loss = np.mean(np.logaddexp(0, z)-y*z) + penalty * np.sum(w[1:]**2)/2
        residual = _sigmoid(z) - y
        gradient = np.r_[residual.mean(), x.T @ residual/n + penalty*w[1:]]
        return loss, gradient

    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-6})
    return {"intercept": float(result.x[0]), "coefficients": result.x[1:].tolist(),
            "converged": bool(result.success), "iterations": int(result.nit),
            "optimizer_message": str(result.message)}


def _raw_logit(model: dict, x: np.ndarray) -> np.ndarray:
    return float(model["intercept"]) + x @ np.array(model["coefficients"])


def _calibrated_probability(model: dict, x: np.ndarray) -> np.ndarray:
    z = _raw_logit(model, x)
    calibration = model["calibration"]
    return _sigmoid(calibration["slope"]*z + calibration["intercept"])


def _constant_probabilities(frame: pd.DataFrame, model: dict) -> np.ndarray:
    return np.array([model["baseline_by_symbol"].get(s, model["baseline_probability"])
                     for s in frame.symbol], dtype=float)


def _state_probabilities(frame: pd.DataFrame, model: dict) -> np.ndarray:
    return np.array([model["state_baseline"].get(f"{s}:{int(state)}",
                     model["baseline_by_symbol"].get(s, model["baseline_probability"]))
                     for s, state in zip(frame.symbol, frame.current_exceedance)])


def _threshold(y: np.ndarray, p: np.ndarray) -> dict:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"value": 1.000001, "objective": "F1 on validation only", "f1": 0.}
    # Quantile candidates are fixed in count; ties choose the higher threshold.
    candidates = np.unique(np.r_[np.quantile(p, np.linspace(0, 1, 201)), 1.000001])
    best = (0., 1.000001)
    for candidate in candidates:
        alert = p >= candidate
        tp = int(np.sum(alert & (y == 1)))
        score = 2*tp/max(int(alert.sum())+int(y.sum()), 1)
        if (score, float(candidate)) > best:
            best = score, float(candidate)
    return {"value": best[1], "objective": "F1 on validation only", "f1": best[0]}


def probability_metrics(y: np.ndarray, p: np.ndarray, baseline: np.ndarray,
                        alerts: np.ndarray | None = None) -> dict:
    """Metrics return null for undefined quantities, never perfect-looking zeros."""
    from scipy.stats import rankdata
    y, p, baseline = np.asarray(y, int), np.asarray(p, float), np.asarray(baseline, float)
    if not len(y):
        return {"n": 0, "positives": 0, "prevalence": None, "brier": None,
                "baseline_brier": None, "brier_skill": None, "roc_auc": None,
                "average_precision": None, "precision": None, "recall": None,
                "alert_rate": None, "alerts": 0}
    positives = int(y.sum()); negatives = len(y)-positives
    brier, base_brier = float(np.mean((y-p)**2)), float(np.mean((y-baseline)**2))
    auc, ap = None, None
    if positives and negatives:
        auc = float((rankdata(p)[y == 1].sum()-positives*(positives+1)/2)/(positives*negatives))
    if positives:
        order = np.argsort(-p, kind="stable")
        sorted_p, sorted_y = p[order], y[order]
        ends = np.r_[np.flatnonzero(np.diff(sorted_p)), len(y)-1]
        tp = np.cumsum(sorted_y)[ends]
        ap = float(np.sum(np.diff(np.r_[0, tp/positives])*tp/(ends+1)))
    alerts = np.zeros(len(y), bool) if alerts is None else np.asarray(alerts, bool)
    tp = int(np.sum(alerts & (y == 1)))
    return {"n": len(y), "positives": positives, "prevalence": float(y.mean()),
            "brier": brier, "baseline_brier": base_brier,
            "brier_skill": float(1-brier/base_brier) if base_brier > 0 else None,
            "roc_auc": auc, "average_precision": ap,
            "precision": tp/int(alerts.sum()) if alerts.sum() else None,
            "recall": tp/positives if positives else None,
            "alert_rate": float(alerts.mean()), "alerts": int(alerts.sum())}


def _model_predictions(frame: pd.DataFrame, names: list[str], model: dict) -> tuple[np.ndarray, np.ndarray]:
    baseline = _constant_probabilities(frame, model)
    modes = np.array([model["mode_by_symbol"].get(s, "baseline_only") for s in frame.symbol])
    state = _state_probabilities(frame, model)
    p = np.where(modes == "state_baseline", state, baseline)
    if model.get("coefficients") is not None:
        raw = _calibrated_probability(model, _transform(frame, names, model["preprocessing"]))
        p = np.where(modes == "logistic", raw, p)
    active = (modes == "logistic") | (modes == "state_baseline")
    alert = active & (p >= model["threshold"]["value"])
    return p, alert


def train_models(history: pd.DataFrame, boundaries: dict[str, str] | None = None,
                 progress=None) -> tuple[dict, dict, pd.DataFrame]:
    """Train/calibrate/select/test once with chronological, globally purged blocks."""
    source_manifest = history.attrs.get("source_manifest", [])
    features, names = build_features(history)
    data = build_targets(features)
    symbols = sorted(data.symbol.unique())
    artifact = {"schema_version": SCHEMA_VERSION, "model_type": "ridge_logistic_platt_json",
                "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "target_description": TARGET_DESCRIPTION,
                "cadence_minutes": CADENCE_MINUTES, "history_window": HISTORY_WINDOW,
                "minimum_history": MIN_HISTORY, "horizons_days": list(HORIZONS),
                "feature_names": names, "symbols": symbols, "models": {},
                "source_data_start": data._dt.min().isoformat(),
                "source_data_end": data._dt.max().isoformat(),
                "fixed_parameters": {"ridge_penalty": 1e-4, "max_iterations": 2000,
                    "random_seed": 42, "calibration": "Platt on a separate temporal block",
                    "selection": "per-symbol minimum validation Brier among training-fit constant, state baseline and calibrated logistic; ties prefer simpler candidate",
                    "threshold": "validation-only maximum F1, 201 probability quantiles"}}
    report = {"target_description": TARGET_DESCRIPTION, "source_manifest": source_manifest,
              "legacy_labels_used": False, "future_session_features_used": False,
              "evaluation_design": "Development chronological holdout, inspected during development; not a blinded external test. No holdout observations enter fitting, calibration, candidate selection or threshold selection.",
              "operational_gate": {"rule": "suppress all alerts if cumulative probabilities decrease across horizons",
                  "reason_code": "cross_horizon_inconsistency", "fitted_probabilities_unchanged": True,
                  "reported_test_metrics": "raw model probabilities and validation thresholds, before the runtime operational gate"},
              "development_changes": ["Raised numerical optimizer iteration limit from 400 to 2000 after training convergence failures; same loss and regularization.",
                  "Corrected candidate selection to retain a validation-winning current-state baseline, rather than always falling back to a constant when logistic lost. Selection still uses validation outcomes only; holdout metrics had already been inspected during development."],
              "limitations": ["Target measures a surface-risk proxy, not losses or portfolio risk.",
                "Future exceedance includes persistence of an already elevated state; onset-subset metrics are reported separately.",
                "Targets use observed session maxima. All 15 daytime snapshots are required, but source night-session coverage may be partial.",
                "Overlapping forecast windows are correlated; metrics are completed-session forecast origins, not independent event counts.",
                "Observed daytime dates are not an official exchange calendar. Missing symbol dates and calendar gaps over seven days break labels; long holidays are conservatively excluded.",
                "No live market feed is bundled. Stale observations cannot produce a current forecast.",
                "Per-symbol model selection uses validation; test metrics are never used to choose the deployed model.",
                "A calibrated probability may lose accuracy under market regime changes.",
                "Horizons are estimated independently; occasional nonmonotonic cumulative probabilities are disclosed rather than hidden with an unvalidated adjustment.",
                "No claim of predictive value is made for baseline-only or negative held-out skill cases."],
              "horizons": {}, "symbols": symbols, "fixed_parameters": artifact["fixed_parameters"]}
    predictions = []
    for horizon in HORIZONS:
        if progress:
            progress(f"Preparing {horizon}-trading-day temporal split")
        masks, boundaries = chronological_splits(data, horizon, boundaries)
        split = {name: data.loc[mask].copy() for name, mask in masks.items()}
        train, calibration, validation, test = (split[n] for n in
                                               ("train", "calibration", "validation", "test"))
        if len(train) < 200 or len(calibration) < 30 or len(validation) < 30 or len(test) < 30:
            raise ValueError(f"Insufficient complete observations for {horizon}-day temporal split")
        label = f"label_{horizon}"
        ytrain = train[label].to_numpy(int)
        model = {"baseline_probability": float((ytrain.sum()+1)/(len(ytrain)+2)),
                 "baseline_by_symbol": {}, "state_baseline": {}, "state_baseline_support": {},
                 "mode_by_symbol": {}, "mode_reason_by_symbol": {},
                 "threshold": {"value": 1.000001},
                 "coefficients": None}
        for symbol, sub in train.groupby("symbol"):
            model["baseline_by_symbol"][symbol] = float((sub[label].sum()+1)/(len(sub)+2))
            for state, state_sub in sub.groupby("current_exceedance"):
                model["state_baseline"][f"{symbol}:{int(state)}"] = float(
                    (state_sub[label].sum()+1)/(len(state_sub)+2))
                model["state_baseline_support"][f"{symbol}:{int(state)}"] = len(state_sub)
        if len(np.unique(ytrain)) == 2 and ytrain.sum() >= 10 and (1-ytrain).sum() >= 10:
            if progress:
                progress(f"Fitting {horizon}-day fixed ridge logistic model on {len(train):,} rows")
            model["preprocessing"] = _preprocessor_fit(train, names)
            fitted = _fit_logistic(_transform(train, names, model["preprocessing"]), ytrain)
            model.update(fitted)
            zcal = _raw_logit(model, _transform(calibration, names, model["preprocessing"]))
            ycal = calibration[label].to_numpy(int)
            if len(np.unique(ycal)) == 2 and model["converged"]:
                platt = _fit_logistic(zcal.reshape(-1, 1), ycal, penalty=1e-3)
                model["calibration"] = {"slope": platt["coefficients"][0],
                                        "intercept": platt["intercept"],
                                        "converged": platt["converged"]}
                if not platt["converged"]:
                    model["coefficients"] = None
            else:
                model["coefficients"] = None
        validation_selection = {}
        for symbol in symbols:
            sub = validation.loc[validation.symbol == symbol]
            reason = "insufficient validation support"
            mode = "baseline_only"
            if len(sub) >= 50:
                yv = sub[label].to_numpy(int)
                bc = _constant_probabilities(sub, model)
                bs = _state_probabilities(sub, model)
                scores = {"model_brier": None,
                          "constant_brier": float(np.mean((yv-bc)**2)),
                          "state_brier": float(np.mean((yv-bs)**2)), "n": len(sub)}
                candidates = [(scores["constant_brier"], 0, "baseline_only"),
                              (scores["state_brier"], 1, "state_baseline")]
                if sub[label].nunique() == 2 and model["coefficients"] is not None:
                    pv = _calibrated_probability(model, _transform(sub, names, model["preprocessing"]))
                    scores["model_brier"] = float(np.mean((yv-pv)**2))
                    candidates.append((scores["model_brier"], 2, "logistic"))
                validation_selection[symbol] = scores
                mode = min(candidates)[2]
                reason = "minimum validation Brier among constant/state/logistic; ties prefer simpler candidate"
            model["mode_by_symbol"][symbol] = mode
            model["mode_reason_by_symbol"][symbol] = reason
        active_val = validation.loc[validation.symbol.map(model["mode_by_symbol"]) != "baseline_only"]
        if len(active_val):
            pval, _ = _model_predictions(active_val, names, model)
            model["threshold"] = _threshold(active_val[label].to_numpy(int), pval)
        else:
            model["threshold"] = {"value": 1.000001, "objective": "disabled for baseline-only models", "f1": 0.}
        artifact["models"][str(horizon)] = model
        ptest, alerts = _model_predictions(test, names, model)
        baseline = _constant_probabilities(test, model)
        state_baseline = _state_probabilities(test, model)
        ytest = test[label].to_numpy(int)
        result = probability_metrics(ytest, ptest, baseline, alerts)
        state_brier = float(np.mean((ytest-state_baseline)**2))
        result["current_state_baseline_brier"] = state_brier
        result["skill_vs_current_state_baseline"] = float(1-result["brier"]/state_brier) if state_brier else None
        result["baseline_average_precision"] = probability_metrics(ytest, baseline, baseline)["average_precision"]
        result["baseline_roc_auc"] = probability_metrics(ytest, baseline, baseline)["roc_auc"]
        per_symbol = {}
        for symbol in symbols:
            indices = np.flatnonzero((test.symbol == symbol).to_numpy())
            sub = probability_metrics(ytest[indices], ptest[indices], baseline[indices], alerts[indices])
            sub["mode"] = model["mode_by_symbol"][symbol]
            sub["mode_reason"] = model["mode_reason_by_symbol"][symbol]
            if len(indices):
                state_bs = float(np.mean((ytest[indices]-state_baseline[indices])**2))
                sub["current_state_baseline_brier"] = state_bs
                sub["skill_vs_current_state_baseline"] = float(1-sub["brier"]/state_bs) if state_bs else None
            per_symbol[symbol] = sub
        onset = test.current_exceedance.to_numpy() == 0
        onset_metrics = probability_metrics(ytest[onset], ptest[onset], baseline[onset], alerts[onset])
        coverage = {}
        for symbol, sub in data.groupby("symbol"):
            valid_n = int(sub[label].notna().sum())
            coverage[symbol] = {"completed_daily_origins": len(sub), "eligible_daily_origins": valid_n,
                "eligible_fraction": valid_n/len(sub),
                "incomplete_or_missing_trading_day_windows": int((~sub[f"complete_window_{horizon}"]).sum()),
                "insufficient_past_history": int((~sub._history_ready).sum()),
                "purged_at_split_boundaries": int(sub[label].notna().sum() -
                    sum(int((mask & (data.symbol == symbol)).sum()) for mask in masks.values()))}
        split_report = {}
        for name, sub in split.items():
            split_report[name] = {"n": len(sub), "start": sub._dt.min().isoformat(),
                                 "last_origin": sub._dt.max().isoformat(),
                                 "last_label_end": sub[f"label_end_{horizon}"].max().isoformat(),
                                 "prevalence": float(sub[label].mean())}
        report["horizons"][str(horizon)] = {"test": result, "per_symbol": per_symbol,
            "new_exceedance_subset_test": onset_metrics,
            "new_exceedance_subset_definition": "Origins at which neither proxy currently exceeds its past-only threshold",
            "validation_selection": validation_selection, "threshold": model["threshold"],
            "coverage": coverage, "splits": split_report}
        part = test[["symbol", "_timestamp", "_dt", f"label_end_{horizon}",
                     "current_exceedance"]].copy()
        part = part.rename(columns={f"label_end_{horizon}": "label_end"})
        part["horizon_days"], part["label"] = horizon, ytest
        part["probability"], part["constant_baseline_probability"] = ptest, baseline
        part["current_state_baseline_probability"] = state_baseline
        part["alert"], part["threshold"] = alerts, model["threshold"]["value"]
        part["model_status"] = part.symbol.map(model["mode_by_symbol"])
        predictions.append(part)
        if progress:
            progress(f"{horizon} day test: n={result['n']:,}, Brier skill={result['brier_skill']:.3f}, "
                     f"active symbols={sum(m == 'logistic' for m in model['mode_by_symbol'].values())}/{len(symbols)}")
    artifact["split_boundaries"] = boundaries
    artifact["model_available_from"] = boundaries["test"]
    artifact["fitted_until"] = boundaries["test"]
    report["daily_session_quality"] = {"incomplete_sessions_by_symbol": features.attrs.get("incomplete_sessions_by_symbol", {}),
        "unassigned_night_snapshots": features.attrs.get("unassigned_night_snapshots", 0),
        "required_daytime_slots": list(DAYTIME_SLOTS), "night_coverage": "observed source snapshots only; may be partial"}
    report["split_boundaries"] = boundaries
    artifact["test_metrics"] = {h: r["test"] for h, r in report["horizons"].items()}
    artifact["per_symbol_test_metrics"] = {h: r["per_symbol"] for h, r in report["horizons"].items()}
    artifact["latest_snapshot_by_symbol"] = {s: sub._dt.max().isoformat() for s, sub in data.groupby("symbol")}
    return artifact, report, pd.concat(predictions, ignore_index=True)


def load_artifact(path: str | Path = "data_out/predictive/model.json") -> dict:
    """Read only the JSON format produced by this module; never deserialize code."""
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported predictive model schema")
    if not isinstance(artifact.get("feature_names"), list) or not artifact.get("models"):
        raise ValueError("Invalid predictive model artifact")
    width = len(artifact["feature_names"])
    for model in artifact["models"].values():
        if model.get("coefficients") is not None:
            if len(model["coefficients"]) != width:
                raise ValueError("Coefficient/feature mismatch")
            for key in ("median", "scale", "low", "high"):
                if len(model["preprocessing"][key]) != width:
                    raise ValueError("Invalid preprocessing dimensions")
            if not np.isfinite(np.array(model["coefficients"], float)).all():
                raise ValueError("Non-finite coefficients")
    return artifact


def forecast_latest(artifact: dict, history: pd.DataFrame, symbol: str,
                    as_of: Any = None, max_staleness_minutes: float = 4*24*60) -> dict:
    """Forecast the latest observation no later than as_of (Shanghai local time).

    A caller replaying history must pass an explicit as_of after model selection ended. By default the real
    current clock is used and stale historical data cannot masquerade as live.
    Severe missingness/OOD suppresses model probabilities; baseline-only outputs
    are identified as historical rates and never trigger predictive alerts.
    """
    symbol = symbol.lower()
    when = _local_timestamp(as_of)
    result = {"status": "unavailable", "symbol": symbol, "as_of": when.isoformat(),
              "data_timestamp": None, "age_minutes": None,
              "target_description": artifact["target_description"],
              "horizons": [], "signals": {}, "warnings": [], "quality": {},
              "evaluation": {h: artifact.get("per_symbol_test_metrics", {}).get(h, {}).get(symbol, {})
                             for h in artifact["models"]}}
    if symbol not in artifact["symbols"]:
        result["warnings"].append("该品种没有经过训练和验证的预测模型。")
        return result
    if max_staleness_minutes < 0:
        raise ValueError("max_staleness_minutes must be non-negative")
    normal = _normalise(history)
    normal = normal.loc[normal._dt <= when]
    if not (normal.symbol == symbol).any():
        result["warnings"].append("指定时点之前该品种没有可用观测。")
        return result
    if normal.empty:
        result["warnings"].append("指定时点之前没有可用观测。")
        return result
    # Keep the global observed daytime calendar so missing symbol days are visible.
    available_from = artifact.get("model_available_from")
    if available_from and when < pd.Timestamp(available_from):
        result["warnings"].append("该时点早于模型训练、校准和阈值选择完成时间；禁止把样本内回看当成预测。")
        result["quality"]["model_available_from"] = available_from
        return result
    feat, _ = build_features(normal, artifact["symbols"])
    if feat.empty:
        result["warnings"].append("还没有包含全部15个日盘截面的已完成交易日；夜盘或日盘进行中不生成日级预测。")
        return result
    completed_calendar = sorted(feat._trading_date.unique())
    feat = feat.loc[feat.symbol == symbol]
    if feat.empty:
        result["warnings"].append("该品种尚无完整日盘交易日，暂停预测。")
        return result
    latest = feat.iloc[-1]
    age = (when-latest._dt).total_seconds()/60
    result["data_timestamp"], result["age_minutes"] = latest._dt.isoformat(), float(age)
    result["age_calendar_days"] = float(age/(24*60))
    result["trading_date"] = latest._trading_date.date().isoformat()
    result["quality"]["last_completed_session_snapshots"] = int(latest.daily_snapshots)
    missed_dates = [pd.Timestamp(d).date().isoformat() for d in completed_calendar
                    if pd.Timestamp(d) > latest._trading_date]
    result["quality"]["newer_completed_dates_without_symbol"] = missed_dates
    pending = normal.loc[(normal.symbol == symbol) & (normal._dt > latest._dt)]
    result["quality"]["pending_incomplete_session_snapshots"] = len(pending)
    if len(pending):
        result["warnings"].append("最新夜盘或日盘尚未构成完整交易日；预测基于上一已完成日盘收盘。")
    result["quality"]["past_observations"] = int(latest._history_count)
    result["signals"] = {name: (float(latest[name]) if np.isfinite(latest[name]) else None)
                         for name in RAW_FEATURES}
    result["signals"].update({"atm_iv_past_q95": float(latest._atm_iv_q95) if pd.notna(latest._atm_iv_q95) else None,
        "convexity_violation_past_q95": float(latest._convexity_violation_q95) if pd.notna(latest._convexity_violation_q95) else None,
        "currently_exceeding": bool(latest.current_exceedance),
        "atm_iv_daily_max": float(latest.atm_iv_daily_max),
        "convexity_violation_daily_max": float(latest.convexity_violation_daily_max),
        "signal_description": "观测值和历史阈值；模型贡献另以校准后的log-odds列示，均不代表因果作用。"})
    if age > max_staleness_minutes or missed_dates:
        result["status"] = "stale"
        result["warnings"].append("数据已过期或其他品种已有更新完整交易日而本品种缺失；暂停当前预测。选择该品种明确的历史回放时点或接入新行情。")
        return result
    if not latest._history_ready or latest[["atm_iv", "convexity_violation"]].isna().any():
        result["warnings"].append("有效历史不足20个完整交易日或当前核心曲面特征缺失，暂停预测。")
        return result
    result["status"] = "ready"
    names = artifact["feature_names"]
    row = feat.iloc[[-1]]
    missing_fraction = float(row[list(RAW_FEATURES)].isna().mean(axis=1).iloc[0])
    result["quality"]["raw_feature_missing_fraction"] = missing_fraction
    if missing_fraction > .35:
        result["status"] = "unavailable"
        result["warnings"].append("当前特征缺失过多，暂停预测。")
        return result
    ood_fractions, nonpositive_constant_days, nonpositive_state_days = [], [], []
    for horizon in artifact["horizons_days"]:
        model = artifact["models"][str(horizon)]
        baseline = float(model["baseline_by_symbol"].get(symbol, model["baseline_probability"]))
        mode = model["mode_by_symbol"].get(symbol, "baseline_only")
        item = {"days": horizon,
                "evaluation": artifact.get("per_symbol_test_metrics", {}).get(str(horizon), {}).get(symbol, {}), "probability": baseline, "threshold": model["threshold"]["value"],
                "alert": False, "model_status": mode, "baseline_probability": baseline,
                "top_contributions": [], "intercept_log_odds": None,
                "total_log_odds": None,
                "forecast_origin": latest._dt.isoformat(),
                "horizon_basis": "以数据截至的已完成收盘为原点，之后的观测交易日；未来确切日期未知",
                "interpretation": "未来观测交易日窗口曲面代理超阈值概率" if mode == "logistic" else "训练期历史基础发生率；无经验证的增量预测信号"}
        if mode == "state_baseline":
            state_key = f"{symbol}:{int(latest.current_exceedance)}"
            state_missing = state_key not in model["state_baseline"]
            item["probability"] = float(model["state_baseline"].get(state_key, baseline))
            item["alert"] = item["probability"] >= item["threshold"]
            item["interpretation"] = "简单状态统计基准：训练期在当前是否超阈值这一状态下的未来发生率；不宣称复杂模型增量"
            item["state_condition"] = {"currently_exceeding": bool(latest.current_exceedance),
                "training_observations_in_state": model.get("state_baseline_support", {}).get(state_key, 0),
                "fallback_to_symbol_baseline": state_missing,
                "historical_conditional_probability": item["probability"],
                "explanation": ("训练期未出现该状态，回退到该品种整体历史基础率，没有逻辑回归特征贡献" if state_missing else
                    "训练期该品种同状态的(未来事件数+1)/(样本数+2)，没有逻辑回归特征贡献")}
        heldout_skill = item["evaluation"].get("brier_skill")
        item["heldout_skill_status"] = ("unavailable" if heldout_skill is None else
                                        "positive" if heldout_skill > 0 else "nonpositive")
        incremental_skill = item["evaluation"].get("skill_vs_current_state_baseline")
        item["incremental_skill_vs_state_status"] = ("unavailable" if incremental_skill is None else
            "positive" if incremental_skill > 0 else "nonpositive")
        if mode in ("logistic", "state_baseline") and heldout_skill is not None and heldout_skill <= 0:
            nonpositive_constant_days.append(horizon)
        if mode == "logistic" and incremental_skill is not None and incremental_skill <= 0:
            nonpositive_state_days.append(horizon)
        if mode in ("logistic", "state_baseline") and "preprocessing" in model:
            preprocessing = model["preprocessing"]
            raw = row[names].to_numpy(dtype=float)[0]
            low, high = np.array(preprocessing["low"]), np.array(preprocessing["high"])
            monitored = np.array([n in RAW_FEATURES for n in names]) & np.isfinite(raw) & (high > low)
            ood = monitored & ((raw < low) | (raw > high))
            ood_fraction = float(ood.sum()/max(monitored.sum(), 1))
            ood_fractions.append(ood_fraction)
            item["ood_fraction"] = ood_fraction
            item["ood_features"] = [n for n, outside in zip(names, ood) if outside]
            if ood_fraction > .35:
                item["probability"] = None
                item["alert"] = False
                item["model_status"] = "out_of_distribution"
            elif mode == "logistic" and model.get("coefficients") is not None:
                x = _transform(row, names, preprocessing)[0]
                cal = model["calibration"]
                contribution = cal["slope"] * np.array(model["coefficients"]) * x
                intercept = cal["slope"]*model["intercept"] + cal["intercept"]
                logit = float(intercept+contribution.sum())
                p = float(_sigmoid(np.array([logit]))[0])
                item.update({"probability": p, "alert": p >= item["threshold"],
                             "intercept_log_odds": float(intercept), "total_log_odds": logit,
                             "contribution_sum": float(contribution.sum()),
                             "top_contributions": [{"feature": names[i],
                                 "value": float(raw[i]) if np.isfinite(raw[i]) else None,
                                 "imputed": bool(not np.isfinite(raw[i])),
                                 "log_odds_contribution": float(contribution[i])}
                                for i in np.argsort(-np.abs(contribution))[:6]]})
        result["horizons"].append(item)
    if nonpositive_constant_days:
        result["warnings"].append(f"{nonpositive_constant_days}日预测在保留测试段未超过历史基础率；该监测结论未用于回头选择模型。")
    if nonpositive_state_days:
        result["warnings"].append(f"{nonpositive_state_days}日复杂模型在保留测试段未超过简单状态基准，不能宣称增量预测能力。")
    result["quality"]["maximum_ood_fraction"] = max(ood_fractions, default=0.)
    if result["quality"]["maximum_ood_fraction"] > .35:
        result["status"] = "out_of_distribution"
        result["warnings"].append("核心特征明显超出训练分布；对应预测已暂停。")
    elif any(item["model_status"] in ("baseline_only", "state_baseline") for item in result["horizons"]):
        result["status"] = "degraded"
        result["warnings"].append("部分期限使用简单统计基准：constant仅展示历史基础率且不预警；state按当前状态的历史条件率并使用验证集阈值，不能宣称复杂模型增量。")
    elif max(ood_fractions, default=0.) > 0:
        result["status"] = "degraded"
        result["warnings"].append("部分观测超出训练期99%范围，需谨慎解读概率。")
    valid_probabilities = [item["probability"] for item in result["horizons"]
                           if item["probability"] is not None]
    consistent = not any(right < left for left, right in zip(valid_probabilities, valid_probabilities[1:]))
    result["quality"]["horizon_probabilities_consistent"] = consistent
    if not consistent:
        for item in result["horizons"]:
            item["model_alert_before_operational_gate"] = item["alert"]
            item["alert"] = False
            item["alert_suppressed_reason"] = "cross_horizon_inconsistency"
        result["warnings"].append("各期限累计概率出现非单调，违反嵌套事件的一致性；保留原始估计供研究查看，全部预警已被运行时一致性检查暂停。")
        result["status"] = "degraded"
    return result
