"""Leakage, observed trading-date, label coverage and inference invariants."""
import json
import unittest

import numpy as np
import pandas as pd

from vol_surface.predictive import (
    DAYTIME_SLOTS, RAW_FEATURES, aggregate_sessions, build_features, build_targets,
    chronological_splits, forecast_latest, probability_metrics,
    _model_predictions,
)


def synthetic(n_days=90, symbol="ag", start="2025-01-01"):
    rows = []
    for i, day in enumerate(pd.bdate_range(start, periods=n_days)):
        for j, clock in enumerate(DAYTIME_SLOTS):
            ts = pd.Timestamp(f"{day.date()} {clock}")
            row = {name: .2 + .03*np.sin(i/9) + j*.0001 for name in RAW_FEATURES}
            row["convexity_violation"] = .1 + .01*np.cos(i/7)
            row.update(symbol=symbol, _timestamp=ts.strftime("%Y%m%d%H%M%S"))
            rows.append(row)
    return pd.DataFrame(rows)


class TestPredictive(unittest.TestCase):
    def test_daily_features_do_not_change_when_future_changes(self):
        history = synthetic()
        initial, names = build_features(history)
        changed = history.copy()
        changed.loc[60*15:, list(RAW_FEATURES)] = 99999
        mutated, _ = build_features(changed)
        pd.testing.assert_frame_equal(initial.loc[:59, names], mutated.loc[:59, names])

    def test_threshold_excludes_current_daily_max(self):
        history = synthetic(40)
        history.loc[39*15:, "atm_iv"] = 99999
        frame, _ = build_features(history)
        expected = frame.atm_iv_daily_max.iloc[:39].quantile(.95)
        self.assertAlmostEqual(frame._atm_iv_q95.iloc[-1], expected)

    def test_label_next_trading_days_and_actual_label_end(self):
        history = synthetic(45)
        history["atm_iv"], history["convexity_violation"] = .2, .1
        history.loc[30*15:31*15-1, "atm_iv"] = 1.
        frame, _ = build_features(history)
        result = build_targets(frame)
        self.assertEqual(result.loc[29, "label_1"], 1.)
        self.assertEqual(result.loc[28, "label_1"], 0.)
        self.assertEqual(result.loc[30, "label_1"], 0.)  # current is not future
        self.assertEqual(result.loc[27, "label_3"], 1.)
        self.assertEqual(result.loc[27, "label_end_3"], result.loc[30, "_dt"])
        self.assertTrue(pd.isna(result.label_5.iloc[-1]))
        self.assertTrue((result.label_end_1.dropna().dt.hour == 15).all())

    def test_missing_symbol_day_breaks_label_and_lag(self):
        ag, au = synthetic(60), synthetic(60, "au")
        missing_date = ag._timestamp.iloc[35*15][:8]
        ag = ag.loc[~ag._timestamp.str.startswith(missing_date)]
        frame, _ = build_features(pd.concat([ag, au], ignore_index=True))
        result = build_targets(frame)
        rows = result.loc[result.symbol == "ag"].reset_index(drop=True)
        self.assertTrue(pd.isna(rows.loc[34, "label_1"]))
        self.assertTrue(pd.isna(rows.loc[35, "atm_iv_change1d"]))
        self.assertTrue(pd.isna(rows.loc[35, "atm_iv_change5d"]))

    def test_long_calendar_gap_is_not_five_adjacent_trading_days(self):
        first, second = synthetic(35), synthetic(35, start="2025-08-01")
        frame, _ = build_features(pd.concat([first, second], ignore_index=True))
        result = build_targets(frame)
        self.assertTrue(pd.isna(result.loc[34, "label_1"]))
        self.assertTrue(pd.isna(result.loc[32, "label_5"]))

    def test_partial_day_is_excluded_and_breaks_future_window(self):
        history = synthetic(45)
        history = history.drop(index=range(31*15, 31*15+10))
        result = build_targets(build_features(history)[0])
        self.assertEqual(len(result), 44)
        self.assertTrue(pd.isna(result.loc[30, "label_1"]))

    def test_friday_night_maps_to_monday_and_pending_night_is_excluded(self):
        history = synthetic(30, start="2025-01-01")
        friday = pd.Timestamp("2025-01-31 21:15")
        night = {name: 10. for name in RAW_FEATURES}
        night.update(symbol="ag", _timestamp=friday.strftime("%Y%m%d%H%M%S"))
        history = pd.concat([history, pd.DataFrame([night])], ignore_index=True)
        sessions = aggregate_sessions(history)
        monday = sessions.loc[sessions._trading_date == pd.Timestamp("2025-02-03")].iloc[0]
        self.assertEqual(monday.atm_iv_daily_max, 10.)
        self.assertEqual(monday.night_snapshots, 1)
        friday_close = sessions.loc[sessions._trading_date == pd.Timestamp("2025-01-31")].iloc[0]
        self.assertLess(friday_close.atm_iv_daily_max, 1.)
        cutoff = history.loc[history._timestamp <= "20250131235959"]
        prefix = aggregate_sessions(cutoff)
        self.assertEqual(prefix._trading_date.max(), pd.Timestamp("2025-01-31"))
        prefix_friday = prefix.iloc[-1]
        self.assertEqual(prefix_friday.atm_iv_daily_max, friday_close.atm_iv_daily_max)

    def test_global_split_purges_full_daily_horizon(self):
        history = pd.concat([synthetic(100), synthetic(100, "au")], ignore_index=True)
        result = build_targets(build_features(history)[0])
        masks, boundaries = chronological_splits(result, 5)
        for name, following in (("train", "calibration"), ("calibration", "validation"), ("validation", "test")):
            self.assertTrue((result.loc[masks[name], "label_end_5"] < pd.Timestamp(boundaries[following])).all())
        for i, (left_name, left) in enumerate(masks.items()):
            for right_name, right in list(masks.items())[i+1:]:
                self.assertFalse((left & right).any(), f"{left_name}/{right_name}")

    def test_constant_metrics_are_honest_and_json_safe(self):
        result = probability_metrics(np.array([0, 1, 0, 1]), np.ones(4)*.5, np.ones(4)*.5)
        self.assertEqual(result["roc_auc"], .5)
        self.assertEqual(result["average_precision"], .5)
        self.assertEqual(result["brier_skill"], 0.)
        self.assertIsNone(result["precision"])
        json.dumps(result, allow_nan=False)

    def test_stale_unknown_and_model_hindsight_refuse_predictions(self):
        artifact = {"symbols": ["ag"], "target_description": "proxy", "models": {},
                    "horizons_days": [1, 3, 5], "model_available_from": "2025-01-01"}
        history = synthetic(45)
        stale = forecast_latest(artifact, history, "ag", as_of="2026-01-01")
        self.assertEqual(stale["status"], "stale")
        self.assertEqual(stale["horizons"], [])
        unknown = forecast_latest(artifact, history, "cu", as_of="2025-01-01")
        self.assertEqual(unknown["status"], "unavailable")
        hindsight = forecast_latest(artifact, history, "ag", as_of="2024-12-31")
        self.assertEqual(hindsight["status"], "unavailable")

    def test_partial_today_does_not_enter_daily_forecast(self):
        history = synthetic(45)
        when = pd.to_datetime(history._timestamp.iloc[-8], format="%Y%m%d%H%M%S")
        artifact = {"symbols": ["ag"], "target_description": "proxy", "models": {},
                    "horizons_days": [], "feature_names": list(RAW_FEATURES)}
        result = forecast_latest(artifact, history, "ag", as_of=when)
        self.assertLess(pd.Timestamp(result["data_timestamp"]).normalize(), when.normalize())
        self.assertGreater(result["quality"]["pending_incomplete_session_snapshots"], 0)

    def test_symbol_missing_newer_observed_session_is_stale_within_four_days(self):
        history = pd.concat([synthetic(44), synthetic(45, "au")], ignore_index=True)
        artifact = {"symbols": ["ag", "au"], "target_description": "proxy",
                    "models": {}, "horizons_days": [], "feature_names": list(RAW_FEATURES)}
        latest = pd.to_datetime(history._timestamp.max(), format="%Y%m%d%H%M%S")
        result = forecast_latest(artifact, history, "ag", as_of=latest)
        self.assertEqual(result["status"], "stale")
        self.assertLess(result["age_calendar_days"], 4)
        self.assertEqual(result["horizons"], [])

    def test_contributions_reconstruct_calibrated_probability(self):
        history = synthetic(45)
        frame, names = build_features(history)
        width = len(names)
        preprocessing = {"median": [0.]*width, "scale": [1.]*width,
                         "low": [-1e10]*width, "high": [1e10]*width, "clip": 8.}
        model = {"baseline_probability": .2, "baseline_by_symbol": {"ag": .2},
                 "mode_by_symbol": {"ag": "logistic"}, "coefficients": [.01]*width,
                 "intercept": -.5, "calibration": {"slope": .8, "intercept": .1},
                 "threshold": {"value": .5}, "preprocessing": preprocessing}
        artifact = {"symbols": ["ag"], "target_description": "proxy", "models": {"1": model},
                    "horizons_days": [1], "feature_names": names}
        result = forecast_latest(artifact, history, "ag", as_of=frame._dt.max())
        item = result["horizons"][0]
        self.assertAlmostEqual(item["total_log_odds"], item["intercept_log_odds"]+item["contribution_sum"])
        self.assertAlmostEqual(item["probability"], 1/(1+np.exp(-item["total_log_odds"])))
        json.dumps(result, allow_nan=False)

    def test_state_baseline_works_without_logistic_coefficients(self):
        history = synthetic(45)
        frame, names = build_features(history)
        model = {"baseline_probability": .2, "baseline_by_symbol": {"ag": .2},
                 "mode_by_symbol": {"ag": "state_baseline"}, "coefficients": None,
                 "state_baseline": {"ag:0": .7, "ag:1": .9},
                 "state_baseline_support": {"ag:0": 42, "ag:1": 12},
                 "threshold": {"value": .6}}
        artifact = {"symbols": ["ag"], "target_description": "proxy", "models": {"1": model},
                    "horizons_days": [1], "feature_names": names}
        result = forecast_latest(artifact, history, "ag", as_of=frame._dt.max())
        item = result["horizons"][0]
        expected = .9 if frame.current_exceedance.iloc[-1] else .7
        self.assertEqual(item["probability"], expected)
        self.assertEqual(item["model_status"], "state_baseline")
        self.assertEqual(item["top_contributions"], [])
        self.assertTrue(item["alert"])
        p, alert = _model_predictions(frame.iloc[[-1]], names, model)
        self.assertEqual(p[0], expected)
        self.assertTrue(alert[0])

    def test_inconsistent_nested_horizons_suppress_alerts_without_altering_estimates(self):
        history = synthetic(45)
        frame, names = build_features(history)
        models = {}
        for days, probability in ((1, .8), (3, .2), (5, .9)):
            models[str(days)] = {"baseline_probability": .1, "baseline_by_symbol": {"ag": .1},
                "mode_by_symbol": {"ag": "state_baseline"}, "coefficients": None,
                "state_baseline": {"ag:0": probability, "ag:1": probability},
                "threshold": {"value": .5}}
        artifact = {"symbols": ["ag"], "target_description": "proxy", "models": models,
                    "horizons_days": [1, 3, 5], "feature_names": names}
        result = forecast_latest(artifact, history, "ag", as_of=frame._dt.max())
        self.assertFalse(result["quality"]["horizon_probabilities_consistent"])
        self.assertEqual(result["status"], "degraded")
        self.assertEqual([item["probability"] for item in result["horizons"]], [.8, .2, .9])
        self.assertEqual([item["model_alert_before_operational_gate"] for item in result["horizons"]], [True, False, True])
        for item in result["horizons"]:
            self.assertFalse(item["alert"])
            self.assertEqual(item["alert_suppressed_reason"], "cross_horizon_inconsistency")
        for days, probability in ((1, .2), (3, .8), (5, .9)):
            models[str(days)]["state_baseline"] = {"ag:0": probability, "ag:1": probability}
        coherent = forecast_latest(artifact, history, "ag", as_of=frame._dt.max())
        self.assertTrue(coherent["quality"]["horizon_probabilities_consistent"])
        self.assertEqual([item["alert"] for item in coherent["horizons"]], [False, True, True])
        self.assertTrue(all("alert_suppressed_reason" not in item for item in coherent["horizons"]))


if __name__ == "__main__":
    unittest.main()
