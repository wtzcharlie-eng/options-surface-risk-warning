"""Shared evidence boundary for the forecast UI, CLI, and conversational layer."""
from __future__ import annotations

import json
from pathlib import Path

from .predictive import forecast_latest, load_anchor_data, load_artifact

ROOT = Path(__file__).resolve().parents[1]


def load_report(model_dir: str | Path | None = None) -> dict:
    folder = Path(model_dir) if model_dir else ROOT / 'data_out' / 'predictive'
    path = folder / 'report.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def build_evidence(snapshot: dict, selected_days: int, report: dict | None = None) -> dict:
    """Keep evidence bounded, factual, and independent of model credentials."""
    horizons = snapshot.get('horizons', [])
    selected = next((h for h in horizons if int(h.get('days', h.get('horizon_days', 0))) == selected_days), {})
    report = report or {}
    return {
        'symbol': snapshot.get('symbol'),
        'as_of': snapshot.get('as_of'),
        'data_timestamp': snapshot.get('data_timestamp'),
        'horizon_trading_days': selected_days,
        'mode': '历史数据回放；未连接实时行情',
        'risk_scope': '曲面风险代理事件；不等同于组合亏损或VaR',
        'target_definition': report.get('target_description', report.get('target_definition', snapshot.get('target_description', '见模型报告'))),
        'forecast': selected,
        'data_quality': {
            'status': snapshot.get('status'),
            'age_minutes': snapshot.get('age_minutes'),
            'warnings': snapshot.get('warnings', []),
            'quality': snapshot.get('quality', {}),
        },
        'observed_signals': snapshot.get('signals', {}),
        'evaluation': snapshot.get('evaluation', {}).get(str(selected_days), selected.get('evaluation', {})),
        'evaluation_context': '完整测试期的事后统计；不是历史观察日当时已知的输入，未用于拟合预测或选择阈值。',
        'other_horizons': [{k: h.get(k) for k in ('days', 'probability', 'threshold', 'alert', 'model_status')} for h in horizons],
    }


def get_snapshot(symbol: str, as_of=None, anchor_dir=None, model_dir=None) -> tuple[dict, dict]:
    folder = Path(model_dir) if model_dir else ROOT / 'data_out' / 'predictive'
    history = load_anchor_data(str(anchor_dir or ROOT / 'data_out' / 'anchor'))
    artifact = load_artifact(str(folder / 'model.json'))
    snapshot = forecast_latest(artifact, history, symbol, as_of=as_of)
    return snapshot, load_report(folder)
