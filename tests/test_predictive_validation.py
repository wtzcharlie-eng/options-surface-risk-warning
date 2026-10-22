"""Temporal evaluator tests: boundaries, paired clustering, and abstentions."""
import unittest

import numpy as np
import pandas as pd

from vol_surface.predictive_validation import (
    default_test_windows, evaluate_predictions, global_purged_splits,
    paired_block_bootstrap,
)


class PredictiveValidationTests(unittest.TestCase):
    def test_global_split_purges_actual_five_day_end(self):
        dates = pd.to_datetime(["2024-01-01 15:00", "2024-01-02 15:00", "2024-01-05 15:00",
                                "2024-01-08 15:00", "2024-01-12 15:00"])
        frame = pd.DataFrame({"_dt": dates,
                              "label_end_5": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-06", "2024-01-10", "2024-01-16"])})
        for h in range(1, 6): frame[f"label_{h}"] = 1.
        masks = global_purged_splits(frame, {"calibration": "2024-01-04", "validation": "2024-01-07", "test": "2024-01-11"}, "2024-01-16")
        self.assertEqual(np.flatnonzero(masks["train"]).tolist(), [0])
        self.assertEqual(np.flatnonzero(masks["calibration"]).tolist(), [2])
        self.assertEqual(np.flatnonzero(masks["validation"]).tolist(), [3])
        # Equality at label boundary is purged, even though the origin is earlier.
        self.assertFalse(masks["test"].any())
        self.assertTrue(np.all(np.stack(list(masks.values())).sum(axis=0) <= 1))

    def test_any_missing_horizon_excludes_origin_for_all(self):
        frame = pd.DataFrame({"_dt": pd.to_datetime(["2024-02-01"]),
                              "label_end_5": pd.to_datetime(["2024-02-08"])})
        for h in range(1, 6): frame[f"label_{h}"] = 0.
        frame["label_2"] = np.nan
        masks = global_purged_splits(frame, {"calibration": "2024-01-01", "validation": "2024-01-10", "test": "2024-01-20"})
        self.assertFalse(any(mask.any() for mask in masks.values()))

    def test_bootstrap_retains_paired_symbols_in_same_dates(self):
        dates = pd.date_range("2024-01-01", periods=30)
        frame = pd.DataFrame({"_dt": np.repeat(dates, 2), "label": 0.,
                              "probability": np.tile([0., 1.], 30),
                              "baseline_probability": np.tile([1., 0.], 30)})
        result = paired_block_bootstrap(frame, n_boot=50)
        self.assertEqual(result["brier_delta"], 0.)
        self.assertEqual(result["brier_delta_ci95"], [0., 0.])
        self.assertEqual(result["dates"], 30)
        self.assertEqual(result, paired_block_bootstrap(frame, n_boot=50))

    def test_abstentions_remain_in_operational_recall_denominator(self):
        rows = []
        for h in range(1, 6):
            for i, (label, raw, allowed) in enumerate(zip([1, 1, 0, 0], [True, True, True, False], [False, True, True, True])):
                rows.append({"symbol": "a", "_dt": pd.Timestamp("2024-01-01")+pd.Timedelta(days=i),
                             "horizon_days": h, "calendar_index": i, "label": label,
                             "probability": .6, "baseline_probability": .5,
                             "raw_alert": raw, "gate_allowed": allowed,
                             "gate_reason": None if allowed else "out_of_distribution",
                             "operational_alert": raw and allowed, "prediction_task": "onset",
                             "model_status": "joint_logistic"})
        item = evaluate_predictions(pd.DataFrame(rows), n_boot=20)["1"]
        self.assertEqual(item["test"]["n"], 4)
        self.assertEqual(item["raw"]["recall_all_origins"], 1.)
        self.assertEqual(item["operational"]["recall_all_origins"], .5)
        self.assertEqual(item["operational"]["probability_coverage"], .75)
        self.assertEqual(item["operational"]["abstentions"], 1)
        self.assertEqual(item["operational"]["false_alarms"], 1)
        self.assertEqual(item["operational"]["abstention_reasons"], {"out_of_distribution": 1})
        self.assertEqual(item["operational"]["probability_metrics_when_available"]["n"], 3)
        self.assertIsNone(item["tasks"]["persistence"]["test"]["brier"])

    def test_fixed_fold_dates_do_not_depend_on_test_outcomes(self):
        frame = pd.DataFrame({"_dt": pd.bdate_range("2023-01-01", "2026-01-01")})
        windows = default_test_windows(frame)
        self.assertEqual([w["boundaries"]["test"][:10] for w in windows],
                         ["2024-12-19", "2025-06-20", "2025-10-17"])
        for window in windows:
            b = window["boundaries"]
            self.assertLess(pd.Timestamp(b["calibration"]), pd.Timestamp(b["validation"]))
            self.assertLess(pd.Timestamp(b["validation"]), pd.Timestamp(b["test"]))


if __name__ == "__main__":
    unittest.main()
