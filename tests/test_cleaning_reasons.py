"""test_cleaning_reasons — `clean_slice(with_reasons=)` 的回归与正确性测试。

为什么必须有这个文件
--------------------
`with_reasons` 是借鉴另一份实现的 `quality_flags.annotate_quality`（只打标不删行）
后加的可选开关。本项目有 **21 处调用**依赖 `clean_slice` 的两元组返回，
任何行为变化都会静默扩散到特征、标签、指标乃至交付数字。

故本测试的第一条、也是最重要的一条断言是：**默认路径（with_reasons=False）
的输出与开关加入前逐位一致**。其余断言检查逐行原因本身的自洽性。

运行::

    python -m tests.test_cleaning_reasons
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vol_surface.cleaning import clean_slice
from vol_surface.io_loader import (load_symbol_family, slice_at_timestamp,
                                   timestamps_in)

_RESULTS = []


def check(name: str):
    def deco(fn):
        def wrapped(*a, **kw):
            try:
                msg = fn(*a, **kw)
                _RESULTS.append((name, True, msg or ""))
                print(f"  [PASS] {name}  {msg or ''}")
            except AssertionError as e:
                _RESULTS.append((name, False, str(e)))
                print(f"  [FAIL] {name}  {e}")
            except Exception as e:                             # noqa: BLE001
                _RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"  [ERROR] {name}  {type(e).__name__}: {e}")
        return wrapped
    return deco


def _slices(n: int = 8):
    df = load_symbol_family("ag", year=2024, month=2)
    tss = timestamps_in(df)
    out = []
    for ts in tss[:: max(1, len(tss) // n)][:n]:
        sl = slice_at_timestamp(df, ts)
        if len(sl) >= 15:
            out.append((ts, sl))
    return out


@check("C1 开关不改变默认行为（逐位一致）")
def test_default_unchanged(slices) -> str:
    """开关打开与否，**返回的 DataFrame 与各步计数必须完全相同**。

    这是本文件存在的首要理由：21 处调用走的是默认路径，
    若 with_reasons 的实现意外改动了 out/rep，缺陷会静默扩散到交付数字。
    """
    n = 0
    for ts, sl in slices:
        a, ra = clean_slice(sl)
        b, rb = clean_slice(sl, with_reasons=True)
        assert a.shape == b.shape, f"{ts} 形状不同 {a.shape} vs {b.shape}"
        assert a.equals(b), f"{ts} 清洗后的 DataFrame 不逐位相同"
        assert ra.as_dict() == rb.as_dict(), f"{ts} 各步计数不同"
        n += 1
    return f"{n} 个截面，开/关两路的 DataFrame 与计数逐位一致"


@check("C2 逐行原因与汇总计数自洽")
def test_reasons_match_counts(slices) -> str:
    """每个 reason_code 的出现次数必须等于 CleaningReport 里对应的计数。"""
    key = {"suspended": "dropped_suspended",
           "inactive_zombie": "dropped_inactive",
           "stale_price": "dropped_stale",
           "iv_out_of_bounds": "dropped_iv_bound",
           "dead_greeks": "dropped_dead_greeks",
           "neighbor_jump": "dropped_neighbor_jump"}
    n = 0
    for ts, sl in slices:
        _, rep = clean_slice(sl, with_reasons=True)
        cnt: dict = {}
        for codes in rep.reason_codes.values():
            for c in codes:
                cnt[c] = cnt.get(c, 0) + 1
        for code, field in key.items():
            assert cnt.get(code, 0) == getattr(rep, field), (
                f"{ts} {code} 逐行计数 {cnt.get(code, 0)} != rep.{field} "
                f"{getattr(rep, field)}")
        n += 1
    return f"{n} 个截面，6 类 reason_code 的逐行计数与汇总计数全部相等"


@check("C3 被保留的行原因为空，被剔除的行原因非空")
def test_kept_vs_dropped(slices) -> str:
    """保留/剔除与「有无原因」必须一一对应，且总数守恒。"""
    n = 0
    for ts, sl in slices:
        out, rep = clean_slice(sl, with_reasons=True)
        assert len(rep.reason_codes) == len(sl), f"{ts} 逐行表长度 != 输入行数"
        empty = {i for i, c in rep.reason_codes.items() if not c}
        nonempty = set(rep.reason_codes) - empty
        assert len(empty) == len(out), (
            f"{ts} 无原因的行数 {len(empty)} != 保留行数 {len(out)}")
        assert len(nonempty) == len(sl) - len(out), (
            f"{ts} 有原因的行数 {len(nonempty)} != 剔除行数 {len(sl) - len(out)}")
        n += 1
    return f"{n} 个截面，保留行↔无原因、剔除行↔有原因 一一对应且总数守恒"


@check("C4 原因可解释：抽查的剔除行确实满足其原因")
def test_reason_is_true(slices) -> str:
    """随机抽查若干条被剔除的行，验证它**真的**满足所记的原因。

    只验证判据独立于清洗顺序的两类（suspended / iv_out_of_bounds）——
    其余几类的判定基于前序步骤已过滤的子集，脱离上下文无法单独复算。
    """
    checked = 0
    for ts, sl in slices:
        _, rep = clean_slice(sl, with_reasons=True)
        for i, codes in rep.reason_codes.items():
            if "suspended" in codes:
                assert int(sl.loc[i, "suspend_flag"] or 0) == 1, \
                    f"{ts} 行{i} 标了 suspended 但 suspend_flag != 1"
                checked += 1
            if "iv_out_of_bounds" in codes:
                iv = float(sl.loc[i, "iv"])
                assert iv <= 0.01 or iv >= 3.0, \
                    f"{ts} 行{i} 标了 iv_out_of_bounds 但 iv={iv} 在界内"
                checked += 1
    assert checked > 0, "没有抽到任何可独立复算的剔除行"
    return f"抽查 {checked} 条剔除行，其所记原因均可独立复算为真"


def main() -> None:
    print("载入 ag 2024-02 ...")
    slices = _slices()
    print(f"  {len(slices)} 个截面\n")
    print("检查:")
    test_default_unchanged(slices)
    test_reasons_match_counts(slices)
    test_kept_vs_dropped(slices)
    test_reason_is_true(slices)
    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    print(f"\n{n_pass}/{len(_RESULTS)} 通过")
    sys.exit(0 if n_pass == len(_RESULTS) else 1)


if __name__ == "__main__":
    main()
