"""Shared evidence boundary for the forecast UI, CLI, and conversational layer."""
from __future__ import annotations

import json
from pathlib import Path

from .predictive import load_anchor_data

ROOT = Path(__file__).resolve().parents[1]


def default_model_dir(version: str = 'auto') -> Path:
    if version not in {'auto', 'v1', 'v2'}:
        raise ValueError('Unsupported predictive model version')
    if version == 'v2' or (version == 'auto' and (ROOT / 'data_out' / 'predictive_v2' / 'model.json').exists()):
        return ROOT / 'data_out' / 'predictive_v2'
    return ROOT / 'data_out' / 'predictive'


def load_model_artifact(path: str | Path) -> dict:
    """Select a versioned numeric JSON loader; never deserialize executable models."""
    metadata = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(metadata, dict):
        raise ValueError('Model JSON must be an object')
    version = metadata.get('schema_version')
    if type(version) is not int:
        raise ValueError('Predictive artifact schema must be an integer')
    if version == 1:
        from .predictive import load_artifact
    elif version == 2:
        from .predictive_v2 import load_artifact
    else:
        raise ValueError('Unsupported predictive artifact schema')
    return load_artifact(path)


def forecast_model(artifact: dict, history, symbol: str, as_of=None, **kwargs) -> dict:
    version = artifact.get('schema_version')
    if version == 1:
        from .predictive import forecast_latest
    elif version == 2:
        from .predictive_v2 import forecast_latest
    else:
        raise ValueError('Unsupported predictive artifact schema')
    return forecast_latest(artifact, history, symbol, as_of=as_of, **kwargs)


def load_report(model_dir: str | Path | None = None) -> dict:
    folder = Path(model_dir) if model_dir else default_model_dir()
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
        'model_version': snapshot.get('model_version', 'v1'),
        'prediction_task': snapshot.get('prediction_task', 'mixed_exceedance'),
        'prediction_task_definition': snapshot.get('prediction_task_definition', '未来窗口包含新异常及已发生异常的延续'),
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
        'evaluation_baseline': ('同任务、同校准处理的条件风险率统计基准' if snapshot.get('model_version') == 'v2' else '报告中的品种基础率和当前状态基线'),
        'operational_evaluation': snapshot.get('evaluation_details', {}).get(str(selected_days), {}).get('operational', {}),
        'evaluation_uncertainty': snapshot.get('evaluation_details', {}).get(str(selected_days), {}).get('paired_brier_bootstrap', {}),
        'evaluation_context': '完整测试期的事后统计；不是历史观察日当时已知的输入，未用于拟合预测或选择阈值。',
        'curve_quality': snapshot.get('quality', {}).get('horizon_probabilities_consistent'),
        'other_horizons': [{k: h.get(k) for k in ('days', 'probability', 'threshold', 'alert', 'model_status')} for h in horizons],
    }


def get_snapshot(symbol: str, as_of=None, anchor_dir=None, model_dir=None, version: str = 'auto') -> tuple[dict, dict]:
    folder = Path(model_dir) if model_dir else default_model_dir(version)
    history = load_anchor_data(str(anchor_dir or ROOT / 'data_out' / 'anchor'))
    artifact = load_model_artifact(folder / 'model.json')
    snapshot = forecast_model(artifact, history, symbol, as_of=as_of)
    return snapshot, load_report(folder)
