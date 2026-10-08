"""backtest — 对照极端事件标注表的回测验证。

读取 extreme_event.csv（GBK 编码），对事件品种在 [start_date, end_date] 窗口内
检查预警是否触发、首次触发提前时间。同时用正常期触发频率估测误报率。

评测指标
--------
- 召回率 recall   = 命中的事件数 / 总事件数
- 平均提前时间   = 事件开始前首次触发的时间（分钟/小时）
- 误报率 FPR      = 正常期（事件窗口外）触发 level>=2 的截面占比
- 准确率         = (命中 + 正常期未触发) / 总截面数
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

EVENTS_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extreme_event.csv")


def load_events(path: str | None = None, encoding: str = "gbk") -> pd.DataFrame:
    """读取极端事件标注表。返回 DataFrame[start_date, end_date, event_description, sectors_affected]。"""
    path = EVENTS_CSV if path is None else path
    rows = []
    with open(path, encoding=encoding) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    df = pd.DataFrame(rows)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    return df


def _parse_date(d: int) -> datetime:
    s = str(int(d))
    return datetime.strptime(s, "%Y%m%d")


def _affected_symbol(event_sector: str, symbol: str) -> bool:
    """事件板块是否覆盖某品种。极端事件表用中文板块名，这里做宽松匹配。"""
    if not isinstance(event_sector, str):
        return False
    # 品种中文名映射（覆盖数据集主要品种）
    sym_cn = {
        "ag": "白银", "au": "黄金", "si": "工业硅",
        "sc": "原油", "cu": "铜", "al": "铝", "zn": "锌", "ni": "镍", "pb": "铅", "sn": "锡",
        "rb": "螺纹", "hc": "热卷", "i": "铁矿石", "j": "焦煤", "jm": "焦煤", "lc": "碳酸锂",
        "p": "棕榈油", "y": "豆油", "a": "豆一", "m": "豆粕", "c": "玉米", "cs": "淀粉",
        "l": "塑料", "v": "PVC", "pp": "聚丙烯", "eg": "乙二醇", "eb": "苯乙烯",
        "ru": "橡胶", "fu": "燃油", "bu": "沥青", "sp": "纸浆", "pg": "液化气",
        "br": "丁二烯橡胶", "ps": "苯乙烯", "ao": "氧化铝",
    }
    cn = sym_cn.get(symbol)
    if cn and cn in event_sector:
        return True
    # 板块名泛匹配
    board_map = {"贵金属": ["ag", "au"], "有色": ["cu", "al", "zn", "ni", "pb", "sn", "lc", "ao"],
                 "黑色系": ["rb", "hc", "i", "j", "jm"],
                 "能源": ["sc", "fu", "pg", "bu"], "化工": ["l", "v", "pp", "eg", "eb", "ru", "br", "ps", "sp"],
                 "油脂油料": ["p", "y", "m", "a", "c"], "碳酸锂": ["lc"]}
    for board, syms in board_map.items():
        if board in event_sector and symbol in syms:
            return True
    return False


@dataclass
class BacktestResult:
    n_events: int
    n_hit: int
    recall: float
    avg_lead_minutes: float
    n_slices: int
    n_alert: int  # level>=2 截面数
    fpr: float
    accuracy: float
    per_event: list  # [(event, symbol, hit, lead_minutes, max_level)]


def backtest(
    alerts: pd.DataFrame,
    events: pd.DataFrame | None = None,
    symbol: str | None = None,
    min_level: int = 2,
) -> BacktestResult:
    """对单品种的预警序列做回测。

    alerts 需含列: timestamp(str YYYYMMDDHHMMSS), date(int), level, symbol
    events 为 None 时加载默认表
    """
    if events is None:
        events = load_events()
    alerts = alerts.copy()
    alerts["dt"] = pd.to_datetime(alerts["timestamp"], format="%Y%m%d%H%M%S", errors="coerce")

    per_event = []
    hits = 0
    leads = []
    relevant_events = 0
    for _, ev in events.iterrows():
        # 只统计覆盖该品种的事件：召回分母仅含相关事件，避免用不相关事件稀释召回率
        if symbol is not None:
            if not _affected_symbol(ev.get("sectors_affected", ""), symbol):
                continue
        if ev.start_date is pd.NaT:
            continue
        relevant_events += 1
        # 预警窗口：事件开始前 5 天到事件结束
        win_start = ev.start_date - pd.Timedelta(days=5)
        win_end = ev.end_date
        sub = alerts[(alerts["dt"] >= win_start) & (alerts["dt"] <= win_end)]
        if sub.empty:
            per_event.append((ev.get("event_description", ""), symbol, False, 0, 0))
            continue
        triggered = sub[sub["level"] >= min_level]
        if not triggered.empty:
            hits += 1
            first_t = triggered["dt"].iloc[0]
            lead = (ev.start_date - first_t).total_seconds() / 60.0
            leads.append(lead)
            per_event.append((ev.get("event_description", ""), symbol, True, lead, int(triggered["level"].max())))
        else:
            per_event.append((ev.get("event_description", ""), symbol, False, 0, int(sub["level"].max())))

    # 误报率：事件窗口外的 level>=2 比例（仅对覆盖该品种的事件剔除窗口）
    in_event = pd.Series(False, index=alerts.index)
    for _, ev in events.iterrows():
        if ev.start_date is pd.NaT:
            continue
        if symbol is not None and not _affected_symbol(ev.get("sectors_affected", ""), symbol):
            continue
        win_start = ev.start_date - pd.Timedelta(days=5)
        in_event |= (alerts["dt"] >= win_start) & (alerts["dt"] <= ev.end_date)
    normal = alerts[~in_event]
    n_alert = int((sub_alerts := normal[normal["level"] >= min_level]).shape[0]) if not normal.empty else 0
    fpr = float(n_alert / max(len(normal), 1))
    n_slices = len(alerts)
    accuracy = (hits + (len(normal) - n_alert)) / max(n_slices, 1)

    return BacktestResult(
        n_events=relevant_events,
        n_hit=hits,
        recall=float(hits / max(relevant_events, 1)),
        avg_lead_minutes=float(sum(leads) / max(len(leads), 1)) if leads else 0.0,
        n_slices=n_slices,
        n_alert=n_alert,
        fpr=fpr,
        accuracy=float(accuracy),
        per_event=per_event,
    )


def format_result(r: BacktestResult) -> str:
    lines = [
        f"回测结果: 事件={r.n_events} 命中={r.n_hit} 召回率={r.recall:.2%} 平均提前={r.avg_lead_minutes:.0f}分钟",
        f"截面={r.n_slices} 误报(level>=2)={r.n_alert} FPR={r.fpr:.2%} 准确率={r.accuracy:.2%}",
        "逐事件:",
    ]
    for desc, sym, hit, lead, maxlv in r.per_event:
        s = "  ✓" if hit else "  ✗"
        lines.append(f"{s} [{sym or '-'}] max_lv={maxlv} lead={lead:.0f}min  {desc[:40]}")
    return "\n".join(lines)
