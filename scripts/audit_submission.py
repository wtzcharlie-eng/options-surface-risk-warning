"""Read frozen checkpoints and report explicit protocols. No training or selection."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import sklearn
from drl.api import AlertAgentAPI
from drl.dataset import load_episodes, apply_continuous_risk, split_episodes, feature_names, suppress_dead_zone
from drl.env import AlertEnv, RewardSpec
from drl.metrics import evaluate_alerts
from scripts.rule_ml import rule_levels, fuse
from vol_surface.submission_metrics import strict_alert_metrics, combine_metrics
from vol_surface.portable_rule_ml import PortableRuleML
from vol_surface.time_utils import epoch_seconds


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = ROOT / "data_out/submission"
    out.mkdir(exist_ok=True)
    eps = load_episodes(str(ROOT / "data_out/anchor"), underlying=True)
    apply_continuous_risk(eps)
    tr, va, te = split_episodes(eps)
    api = AlertAgentAPI.from_dir(str(ROOT / "data_out/drl"))
    wp = json.loads((ROOT / "data_out/drl/workpoint.json").read_text())
    cfg = json.loads((ROOT / "data_out/rule_ml.json").read_text())["picked"]
    api.set_confidence_threshold(wp["lean"])
    model = PortableRuleML(ROOT / "data_out/rule_ml_portable.json")
    sp = json.loads((ROOT / "data_out/symbol_params.json").read_text())
    rules = rule_levels(te, feature_names(True), sp, cfg["zmult"])
    groups = {"rule_ml": [], "drl": []}
    rows = []
    cases = []
    for e, lv in zip(te, rules):
        ml = fuse(lv, model.predict_proba(e.X)[:, 1], cfg["thr"])
        ro = AlertEnv(e, api.norm, RewardSpec()).rollout(api.policy())
        drl = suppress_dead_zone(e.ts, ro["actions"])
        for name, levels in (("rule_ml", ml), ("drl", drl)):
            record = dict(symbol=e.symbol, month=e.ym, start=e.ts[0], end=e.ts[-1],
                          n_observations=len(e),
                          finite_strict=strict_alert_metrics(e.ts, levels, e.risk_idx),
                          unlimited_strict=strict_alert_metrics(e.ts, levels, e.risk_idx, None),
                          legacy_closed=evaluate_alerts(e.ts, levels, e.risk_idx).as_dict())
            groups[name].append(record)
            # Deterministic successful/false-alert examples, not cherry-picked
            # extreme-event narratives: first eligible alert of each kind.
            if sum(c["model"] == name for c in cases) < 2:
                t = epoch_seconds(e.ts)
                risks = t[e.risk_idx]
                for i in np.flatnonzero(np.asarray(levels) >= 2):
                    match = risks[(risks > t[i]) & (risks <= t[i] + 7200)]
                    kind = "future_start_match" if len(match) else "no_future_start_in_120min"
                    if any(c["model"] == name and c["kind"] == kind for c in cases):
                        continue
                    r_index = int(np.searchsorted(t, match[0])) if len(match) else None
                    cases.append(dict(model=name, symbol=e.symbol, month=e.ym, kind=kind,
                                      alert_time=e.ts[int(i)], level=int(levels[i]),
                                      risk_start=e.ts[r_index] if r_index is not None else None,
                                      lead_min=float((match[0]-t[i])/60) if len(match) else None,
                                      evidence_at_alert={k: float(e.X[i,j]) for j,k in
                                          enumerate(feature_names(True)) if k in
                                          ("atm_iv","atm_iv_z","convexity_violation","gamma_concentration","vega_concentration")},
                                      source=f"data_out/anchor/{e.symbol}/{e.ym}/features.parquet"))
                    if sum(c["model"] == name for c in cases) >= 2:
                        break
        rows.append(dict(symbol=e.symbol, month=e.ym, n=len(e), start=e.ts[0], end=e.ts[-1]))
    metrics = {name: {protocol: combine_metrics([r[protocol] for r in values])
                       for protocol in ("finite_strict", "unlimited_strict", "legacy_closed")}
               for name, values in groups.items()}
    checks = []
    for name, args in (("drl", ["-m", "drl.tests", "--underlying"]),
                       ("kg", ["-m", "kg.tests"])):
        run = subprocess.run([sys.executable, "-W", "ignore", *args], cwd=ROOT,
                             capture_output=True, text=True, timeout=900)
        log = run.stdout + run.stderr
        (out / f"{name}_validation.txt").write_text(log, encoding="utf-8")
        import re
        found = re.findall(r"(\d+)/(\d+) 通过", log)
        n, total = map(int, found[-1]) if found else (0, 0)
        checks.append(dict(name=name, passed=run.returncode == 0 and n == total and total > 0,
                           n_passed=n, n_total=total, returncode=run.returncode,
                           log=f"data_out/submission/{name}_validation.txt"))
    paths = sorted({*ROOT.glob("drl/*.py"), *ROOT.glob("kg/*.py"),
                    *ROOT.glob("vol_surface/*.py"), *ROOT.glob("data_out/drl/agent_seed*.npz"),
                    *ROOT.glob("data_out/anchor/*/*/features.parquet"),
                    *ROOT.glob("data_out/anchor/*/*/underlying.parquet"),
                    ROOT / "scripts/rule_ml.py", ROOT / "scripts/audit_submission.py",
                    ROOT / "data_out/rule_ml_model.joblib", ROOT / "data_out/rule_ml.json",
                    ROOT / "data_out/rule_ml_portable.json", ROOT / "scripts/export_rule_ml.py",
                    ROOT / "data_out/symbol_params.json", ROOT / "data_out/drl/workpoint.json",
                    ROOT / "requirements-predictive.lock.txt"})
    evidence = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    report = dict(schema_version=1, purpose="competition-delivery-review",
                  environment=dict(python=platform.python_version(), pandas=pd.__version__,
                                   numpy=np.__version__, sklearn=sklearn.__version__),
                  frozen_configuration=dict(rule_ml={"zmult": cfg["zmult"], "thr": cfg["thr"]},
                                            drl={"lean": wp["lean"], "n_models": len(api.agents)}),
                  split=dict(train_episodes=len(tr), validation_episodes=len(va), test_episodes=len(te)),
                  coverage=rows, metrics=metrics, per_episode=groups, case_matches=cases,
                  checks=checks, source_sha256=evidence,
                  caveats=["已查看历史上的开发复核，不是外部盲测；不重新训练、选阈值。",
                           "有限窗参考实现：精确率匹配 (a,a+120min] 内风险起点，召回匹配 [r-120min,r)；同刻预警不算提前。",
                           "赛题未明确召回上限及风险区间重叠的全部细节；另列严格早于、无上限召回，不称唯一官方算法。",
                           "沿用逐月评测边界，不跨月匹配；未剔除缺少完整未来窗的尾部样本。",
                           "休市过滤依赖回放的后续时刻；上线须改用已知交易日历，不能当作已验证的在线过滤。"],
                  business_cvar=dict(status="unverified", reason="未提供真实交易日开盘价、夜盘交易日映射与连续合约处理证据；旧 CVaR 为滚动5日减仓后恢复协议。",
                                     evaluator="vol_surface/submission_metrics.py:first_alert_cvar"),
                  human_review=dict(status="pending", reason="现有自评和机器核验不等于可核验的独立人工平均分≥4。"))
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(metrics=metrics, checks=checks), ensure_ascii=False, indent=2))
    if not all(c["passed"] for c in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
