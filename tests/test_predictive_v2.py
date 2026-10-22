"""V2 invariants: event semantics, temporal fitting and operational coherence."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from vol_surface import predictive as v1
from vol_surface import predictive_v2 as v2


def history(n=180, symbols=("ag", "au")):
    rows = []
    for symbol_index, symbol in enumerate(symbols):
        for i, day in enumerate(pd.bdate_range("2024-01-01", periods=n)):
            for j, clock in enumerate(v1.DAYTIME_SLOTS):
                row = {name: .25+.03*np.sin(i/6+symbol_index)+j*.0001 for name in v1.RAW_FEATURES}
                row["convexity_violation"] = .1+.02*np.cos(i/8+symbol_index)
                row.update(symbol=symbol, _timestamp=pd.Timestamp(f"{day.date()} {clock}").strftime("%Y%m%d%H%M%S"))
                rows.append(row)
    return pd.DataFrame(rows)


class TestPredictiveV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = history()
        cls.features, cls.names = v2.build_features(cls.raw)
        cls.data = v2.build_targets(cls.features)
        dates = sorted(cls.data._dt.dt.normalize().unique())
        boundaries = {key: pd.Timestamp(dates[index]).isoformat() for key, index in
                      (("calibration", 95), ("validation", 130), ("test", 160))}
        masks, _ = v1.chronological_splits(cls.data, 5, boundaries)
        cls.splits = {key: cls.data.loc[mask].copy() for key, mask in masks.items()}
        cls.models = v2.fit_models(cls.splits["train"], cls.splits["calibration"],
                                   cls.splits["validation"], cls.names, ["ag", "au"])
        cls.artifact = {"schema_version": 2, "feature_names": cls.names, "models": cls.models,
                        "symbols": ["ag", "au"], "horizons_days": list(v2.HORIZONS),
                        "model_available_from": boundaries["test"],
                        "target_description": v2.TARGET_DESCRIPTION}

    def test_multiscale_features_cannot_see_future(self):
        changed = self.raw.copy()
        cutoff = sorted(changed._timestamp.str[:8].unique())[120]
        changed.loc[changed._timestamp.str[:8] >= cutoff, list(v1.RAW_FEATURES)] = 1e5
        mutated, _ = v2.build_features(changed)
        before = self.features._timestamp.str[:8] < cutoff
        pd.testing.assert_frame_equal(self.features.loc[before, self.names], mutated.loc[before, self.names])
        self.assertLess(len(self.names), 65)

    def test_first_future_event_is_not_current_or_continuous_duration(self):
        raw = history(45, ("ag",))
        raw["atm_iv"], raw["convexity_violation"] = .2, .1
        # Origin day30 is abnormal, day31 normal, day32 abnormal again.
        for day in (30, 32):
            raw.loc[day*15:(day+1)*15-1, "atm_iv"] = 1.
        data = v2.build_targets(v2.build_features(raw)[0])
        self.assertEqual(data.loc[30, "prediction_task"], "persistence")
        self.assertEqual(data.loc[30, "_first_event"], 2.)
        self.assertEqual(data.loc[30, "label_1"], 0.)
        self.assertEqual(data.loc[30, "label_2"], 1.)
        self.assertEqual(data.loc[29, "prediction_task"], "onset")
        self.assertEqual(data.loc[29, "_first_event"], 1.)
        self.assertIn("不代表连续异常", v2.TASK_DEFINITIONS["persistence"])

    def test_full_five_day_censoring_applies_to_all_horizons(self):
        for _, sub in self.data.groupby("symbol"):
            self.assertTrue(sub.iloc[-5:][[f"label_{h}" for h in v2.HORIZONS]].isna().all().all())
        eligible = self.data.label_5.notna()
        for h in v2.HORIZONS:
            self.assertTrue((self.data[f"label_{h}"].notna() == eligible).all())

    def test_at_risk_expansion_stops_after_first_event(self):
        frame = pd.DataFrame({"_first_event": [0., 1., 3.]})
        origins, steps, outcome = v2._risk_rows(frame)
        self.assertEqual(set(steps[origins == 0]), {1, 2, 3, 4, 5})
        self.assertEqual(set(steps[origins == 1]), {1})
        self.assertEqual(set(steps[origins == 2]), {1, 2, 3})
        self.assertEqual(outcome[origins == 2].sum(), 1.)

    def test_fit_rejects_future_window_overlap(self):
        overlap = pd.concat([self.splits["train"].tail(1), self.splits["calibration"]])
        with self.assertRaisesRegex(ValueError, "purge"):
            v2.fit_models(self.splits["train"], overlap, self.splits["validation"], self.names)

    def test_all_candidates_calibrate_hazards_and_preserve_nested_curves(self):
        for mode in v2.MODES:
            prediction = v2.predict_curves(self.splits["test"], self.names, self.models, mode=mode)
            p = prediction["probabilities"]
            self.assertTrue(np.isfinite(p).all())
            self.assertTrue(((p > 0) & (p < 1)).all())
            self.assertTrue((np.diff(p, axis=1) >= 0).all())
            np.testing.assert_allclose(p, 1-np.cumprod(1-prediction["hazards"], axis=1))
        for task in v2.TASKS:
            for candidate in self.models["tasks"][task]["candidate_models"].values():
                self.assertIn("intercept", candidate["calibration"])
                self.assertNotIn("slope_by_horizon", candidate["calibration"])

    def test_baseline_has_coherent_rates_and_never_predictive_alerts(self):
        pred = v2.predict_curves(self.splits["test"], self.names, self.models, mode="hazard_baseline")
        np.testing.assert_allclose(pred["probabilities"], pred["baseline_probabilities"])
        self.assertFalse(pred["alerts"].any())
        self.assertTrue((np.diff(pred["probabilities"], axis=1) >= 0).all())

    def test_preprocessing_is_frozen_to_training_block(self):
        changed = self.splits["validation"].copy()
        changed[self.names] = changed[self.names] * 1000
        fit = v2.fit_models(self.splits["train"], self.splits["calibration"], changed,
                            self.names, ["ag", "au"])
        self.assertEqual(fit["preprocessing"], self.models["preprocessing"])
        for task in v2.TASKS:
            self.assertEqual(fit["tasks"][task]["candidate_models"], self.models["tasks"][task]["candidate_models"])

    def test_one_quality_gate_covers_whole_curve_and_baseline(self):
        row = self.splits["test"].iloc[[0]].copy()
        for name in v1.RAW_FEATURES:
            row[name] = 1e9
        gate = v2.operational_gates(row, self.names, self.models)
        self.assertFalse(gate["allowed"][0])
        self.assertEqual(gate["reasons"][0], "out_of_distribution")
        self.assertGreater(gate["ood_fraction"][0], .35)

    def test_constant_raw_feature_departure_abstains_without_diluting_dynamic_ood(self):
        models = copy.deepcopy(self.models)
        row = self.splits["test"].iloc[[0]].copy()
        prep = models["preprocessing"]
        # Keep all varying raw inputs at their training medians to isolate the
        # collapsed-range case, including tolerance for harmless float noise.
        for i, name in enumerate(self.names):
            if name in v1.RAW_FEATURES:
                row[name] = prep["median"][i]
        index = self.names.index("arb_score")
        prep["low"][index] = prep["high"][index] = .2
        row["arb_score"] = .2+1e-9
        near = v2.operational_gates(row, self.names, models)
        self.assertTrue(near["allowed"][0])
        self.assertEqual(near["constant_features_violated"][0], [])
        row["arb_score"] = 100.
        changed = v2.operational_gates(row, self.names, models)
        self.assertFalse(changed["allowed"][0])
        self.assertEqual(changed["reasons"][0], "constant_feature_violation")
        self.assertEqual(changed["constant_features_violated"][0], ["arb_score"])
        self.assertEqual(changed["ood_fraction"][0], near["ood_fraction"][0])

    def test_numeric_json_roundtrip_and_invalid_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.json"
            path.write_text(json.dumps(self.artifact, allow_nan=False))
            loaded = v2.load_artifact(path)
            self.assertEqual(loaded["models"], self.models)
            broken = copy.deepcopy(self.artifact)
            broken["models"]["preprocessing"]["scale"][0] = 0.
            path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "scale"):
                v2.load_artifact(path)

    def test_latest_staleness_and_future_model_gates(self):
        stale = v2.forecast_latest(self.artifact, self.raw, "ag", "2026-01-01")
        self.assertEqual(stale["status"], "stale")
        self.assertEqual(stale["horizons"], [])
        hindsight = v2.forecast_latest(self.artifact, self.raw, "ag", "2024-03-01")
        self.assertEqual(hindsight["status"], "unavailable")
        self.assertIn("model_available_from", hindsight["quality"])

    def test_unsupported_state_is_an_abstention_and_unknown_task_rejected(self):
        models = copy.deepcopy(self.models)
        rows = self.splits["test"].iloc[[0]].copy()
        task = rows.prediction_task.iloc[0]
        models["tasks"][task]["training_origins"] = 0
        pred = v2.predict_curves(rows, self.names, models)
        self.assertEqual(pred["modes"][0], "unsupported_task")
        self.assertFalse(pred["alerts"].any())
        gate = v2.operational_gates(rows, self.names, models)
        self.assertFalse(gate["allowed"][0])
        self.assertEqual(gate["reasons"][0], "unsupported_task")
        rows["prediction_task"] = "recovery"
        with self.assertRaisesRegex(ValueError, "Unknown prediction task"):
            v2.predict_curves(rows, self.names, models)

    def test_task_evaluation_is_flattened_for_runtime_skill_disclosure(self):
        artifact = copy.deepcopy(self.artifact)
        latest = self.features.loc[self.features.symbol == "ag"].iloc[-1]
        task = "onset" if latest.current_exceedance == 0 else "persistence"
        artifact["per_symbol_test_metrics"] = {
            str(h): {"ag": {"task_metrics": {task: {"test": {"brier_skill": -.2, "n": 18},
                "operational": {"coverage": .5}, "raw": {"n": 36}}}}} for h in v2.HORIZONS}
        out = v2.forecast_latest(artifact, self.raw, "ag", latest._dt)
        self.assertEqual(out["evaluation"]["1"]["brier_skill"], -.2)
        self.assertEqual(out["evaluation_details"]["1"]["operational"]["coverage"], .5)

    def test_forecast_contributions_explain_hazard_not_cumulative_logit(self):
        artifact = copy.deepcopy(self.artifact)
        # Widen quality monitor solely to isolate the explanation invariant.
        prep = artifact["models"]["preprocessing"]
        prep["low"], prep["high"] = [-1e10]*len(self.names), [1e10]*len(self.names)
        latest = self.features.loc[self.features.symbol == "ag"].iloc[-1]
        task = "onset" if latest.current_exceedance == 0 else "persistence"
        candidates = artifact["models"]["tasks"][task]["candidate_models"]
        if "joint_logistic" not in candidates:
            self.skipTest("synthetic task has insufficient event support for logistic")
        artifact["models"]["tasks"][task]["mode_by_symbol"]["ag"] = "joint_logistic"
        out = v2.forecast_latest(artifact, self.raw, "ag", latest._dt)
        self.assertEqual(len(out["horizons"]), 5)
        self.assertTrue(out["quality"]["horizon_probabilities_consistent"])
        for item in out["horizons"]:
            self.assertIsNone(item["total_log_odds"])
            self.assertAlmostEqual(item["hazard_total_log_odds"], item["hazard_intercept_log_odds"]+item["hazard_contribution_sum"])
            self.assertAlmostEqual(item["hazard_probability"], float(v1._sigmoid(np.array([item["hazard_total_log_odds"]]))[0]))
            self.assertTrue(all(c["hazard_step"] == item["days"] for c in item["top_contributions"]))
        json.dumps(out, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
