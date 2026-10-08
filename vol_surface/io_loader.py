"""io_loader — 期权曲面数据读取与按品种族聚合。

数据在 archive/<交易所>/<品种月>/options/<年>/<月>/options_<年月>.parquet 下按
(交易所, 品种族, 到期月, 数据月份) 组织。一个完整 IV 曲面需要在同一时间戳下
收集同一品种族（如 ag）跨所有到期月的合约。本模块负责把这种层级结构摊平为
可直接喂给清洗/特征管线的 DataFrame。
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import pandas as pd

ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "archive")

# 交易所后缀 -> 目录名
EXCHANGE_MAP = {".SF": "SF", ".INE": "INE", ".GF": "GF", ".DF": "DF", ".ZF": "ZF"}

_UNDERLYING_RE = re.compile(r"^([a-zA-Z]+)(\d{3,4})$")


def symbol_root(underlying: str) -> str:
    """ag2306 -> ag；返回品种族前缀。"""
    m = _UNDERLYING_RE.match(underlying)
    return m.group(1) if m else underlying


def list_underlyings(root: str | None = None, exchange: str | None = None) -> list[str]:
    """枚举 archive 中出现的所有 underlying（品种+到期月）。

    Parameters
    ----------
    root : 仅返回该品种族的 underlying，例如 "ag"
    exchange : 仅返回该交易所目录
    """
    pattern = os.path.join(ARCHIVE_DIR, "*") if exchange is None else os.path.join(ARCHIVE_DIR, exchange)
    roots = []
    for exch_dir in glob.glob(pattern):
        if not os.path.isdir(exch_dir):
            continue
        for sym_dir in glob.glob(os.path.join(exch_dir, "*")):
            sym = os.path.basename(sym_dir)
            if root is None or symbol_root(sym) == root:
                roots.append(sym)
    return sorted(roots)


@dataclass
class SymbolFamily:
    """一个品种族（如 ag）在某数据月份里涉及的文件集合。"""

    root: str
    underlyings: list[str]
    files: list[str]


def _files_for_root(root: str, year: int | None = None, month: int | None = None) -> list[str]:
    """返回某品种族所有月度 parquet 文件。"""
    # archive/<EX>/<root*>/options/<YYYY>/<MM>/options_<YYYYMM>.parquet
    exch_glob = os.path.join(ARCHIVE_DIR, "*")
    files = []
    for exch_dir in glob.glob(exch_glob):
        sym_glob = os.path.join(exch_dir, f"{root}*")
        for sym_dir in glob.glob(sym_glob):
            if year is not None:
                yr_glob = os.path.join(sym_dir, "options", f"{year:04d}")
                if month is not None:
                    pat = os.path.join(yr_glob, f"{month:02d}", "options_*.parquet")
                else:
                    pat = os.path.join(yr_glob, "*", "options_*.parquet")
            else:
                pat = os.path.join(sym_dir, "options", "*", "*", "options_*.parquet")
            files.extend(glob.glob(pat))
    return sorted(files)


def load_symbol_family(
    root: str,
    year: int | None = None,
    month: int | None = None,
    files: list[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """加载一个品种族在指定（年/月）范围内的全部期权数据并合并。

    返回的 DataFrame 至少包含原始 23 列；按 timestamp, underlying, option_type, strike 排序。

    Parameters
    ----------
    root : 品种族前缀，如 "ag"
    year, month : 限定数据月份；都不给则读全量（注意大品种可能上千万行）
    files : 显式文件列表，优先于 year/month 检索
    columns : 只读这些列以省内存
    """
    if files is None:
        files = _files_for_root(root, year, month)
    if not files:
        raise FileNotFoundError(f"no parquet for root={root} year={year} month={month}")
    dfs = [pd.read_parquet(f, columns=columns) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    # timestamp 是 string "YYYYMMDDHHMMSS"
    df = df.sort_values(["timestamp", "underlying", "option_type", "strike"]).reset_index(drop=True)
    return df


def slice_at_timestamp(df: pd.DataFrame, timestamp: str) -> pd.DataFrame:
    """取出某时间戳下的全曲面截面。"""
    return df[df["timestamp"] == timestamp].copy()


def timestamps_in(df: pd.DataFrame) -> list[str]:
    """有序返回 df 内所有时间戳。"""
    return sorted(df["timestamp"].unique().tolist())


def ts_to_date(ts: str) -> int:
    """'20240222114500' -> 20240222。"""
    return int(ts[:8])


def ts_to_int(ts: str) -> int:
    """字符串时间戳转 int，便于比较/排序。"""
    return int(ts)
