"""preflight — 交付前一键验收。跑一条命令，得到「能不能交付」的结论。

为什么要有它
------------
这个项目的交付事故**全部**不是「指标算错了」，而是：

- 包里漏了 `extreme_event.csv` → 仪表板首屏崩，而打包自检打「✓ 完整」；
- 写给评委的校验命令在包内必崩（默认路径指向包自己内部）；
- `build_platform.py` 在无原始数据时**静默**把旗舰交付物改坏、退出码 0；
- 门在该失败时通过（判据自指 / 探针选错 / 空集合 `all()` 返回 True / 只测了零件没测路径）；
- 写死的文字与现算的表格并列，慢慢变成假话。

共同点：**源目录里一切正常，问题只在「别人拿到包之后」才暴露**。
故本脚本的核心约定是——**它要能在解压出来的包里跑**，并且在那里跑才算数。

用法::

    python scripts/preflight.py              # 全量（约 6~10 分钟）
    python scripts/preflight.py --fast       # 跳过耗时门，用于改完随手自查
    python scripts/preflight.py --json out.json

退出码：0 = 可交付；1 = 有阻塞项；2 = 机器检查全过但仍有待人工确认项。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_out")
PY = sys.executable

# 在解压包里跑时，这些需要原始数据的检查会被自动跳过（并**标注为跳过**，不算通过）
NEEDS_RAW = {"平台可从原始数据重建"}


class R:
    """一条检查的结果。**没有「默认通过」**——必须显式给出 ok。"""

    def __init__(self, name, ok, msg="", blocking=True, skipped=False):
        self.name, self.ok, self.msg = name, ok, msg
        self.blocking, self.skipped = blocking, skipped

    @property
    def mark(self):
        return "跳过" if self.skipped else ("✓" if self.ok else "✗")


def _run(cmd, timeout=900):
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"超时（>{timeout}s）"
    except Exception as e:                                   # noqa: BLE001
        return 125, f"{type(e).__name__}: {e}"


def _has_raw() -> bool:
    return os.path.isdir(os.path.join(ROOT, "archive"))


def _in_package() -> bool:
    """当前是不是在「解压出来的包」里跑。"""
    return not _has_raw() and os.path.exists(os.path.join(ROOT, "PACKAGE.md"))


# ---------------------------------------------------------------- 机器可查

def chk_gates_drl(fast):
    if fast:
        return R("问题3 验证门 (G0–G14)", False, "", skipped=True)
    # **不加 `--emit`**：验收脚本不得改动它要验的东西。
    # 实测教训：带 --emit 会重写 gates.json，于是紧接着的「报告不旧于数据源」
    # 当场判负——**脚本自己把自己跑红了**。
    rc, out = _run([PY, "-W", "ignore", "-m", "drl.tests", "--underlying"])
    m = re.search(r"(\d+)/(\d+) 通过", out)
    if not m:
        return R("问题3 验证门 (G0–G14)", False, f"跑不出结果（rc={rc}）：{out[-200:]}")
    a, b = int(m.group(1)), int(m.group(2))
    fails = re.findall(r"\[(?:FAIL|ERROR)\] (\S+)", out)
    return R("问题3 验证门 (G0–G14)", a == b and rc == 0,
             f"{a}/{b}" + (f"，未过：{fails}" if fails else ""))


def chk_gates_kg(fast):
    if fast:
        return R("问题2 验证门 (K1–K11)", False, "", skipped=True)
    rc, out = _run([PY, "-W", "ignore", "-m", "kg.tests"])   # 同上，不写盘
    m = re.search(r"(\d+)/(\d+) 通过", out)
    if not m:
        return R("问题2 验证门 (K1–K11)", False, f"跑不出结果（rc={rc}）")
    a, b = int(m.group(1)), int(m.group(2))
    return R("问题2 验证门 (K1–K11)", a == b and rc == 0, f"{a}/{b}")


def chk_doc_numbers(fast):
    rc, out = _run([PY, "-W", "ignore", os.path.join("scripts", "check_doc_numbers.py")])
    bad = [l.strip() for l in out.splitlines() if l.strip().startswith("✗")]
    return R("文档数字与落盘一致", rc == 0,
             "全部一致" if rc == 0 else "；".join(bad[:3]))


def chk_package_exists():
    """包必须存在，且**不得旧于任何源文件**——改完不重打包是最常见的交付事故。

    ⚠ 这一条**只在源目录有意义**。在解压出来的包里，`tar` 保留了文件的原始
    mtime，而 tar.gz 本身是新建的，于是「文件比包新」必然成立——是误报。
    这个检查回答的是「你改完忘了重打包吗」，那个问题在包内不存在。
    """
    if _in_package():
        return R("交付包不旧于源文件", False,
                 "包内不适用（tar 保留原始 mtime，此判据只在源目录有意义）",
                 skipped=True)
    cands = [os.path.join(ROOT, "期权曲面风控预警系统.tar.gz"),
             os.path.join(os.path.dirname(ROOT), "期权曲面风控预警系统.tar.gz")]
    pkg = next((p for p in cands if os.path.exists(p)), None)
    if not pkg:
        return R("交付包存在", False, "找不到 tar.gz（源目录与父目录都没有）")
    pt = os.path.getmtime(pkg)
    newer = []
    for top in ("vol_surface", "kg", "drl", "scripts", "dashboard", "data_out"):
        d = os.path.join(ROOT, top)
        if not os.path.isdir(d):
            continue
        for r, ds, fs in os.walk(d):
            ds[:] = [x for x in ds if x != "__pycache__" and not x.startswith(".")]
            for f in fs:
                if f.endswith(".pyc"):
                    continue
                p = os.path.join(r, f)
                try:
                    if os.path.getmtime(p) > pt + 1:
                        newer.append(os.path.relpath(p, ROOT))
                except OSError:
                    pass
    for f in ("README.md", "PACKAGE.md", "extreme_event.csv"):
        p = os.path.join(ROOT, f)
        if os.path.exists(p) and os.path.getmtime(p) > pt + 1:
            newer.append(f)
    ok = not newer
    return R("交付包不旧于源文件", ok,
             f"{os.path.basename(pkg)}（{os.path.getsize(pkg) / 2**20:.1f} MB）"
             if ok else
             f"**包已过期**：{len(newer)} 个文件比它新，如 {newer[:3]}——请重跑 make_package.py")


def chk_package_verify():
    cands = [os.path.join(ROOT, "期权曲面风控预警系统.tar.gz"),
             os.path.join(os.path.dirname(ROOT), "期权曲面风控预警系统.tar.gz")]
    pkg = next((p for p in cands if os.path.exists(p)), None)
    if not pkg:
        return R("包内容自检", False, "找不到 tar.gz")
    rc, out = _run([PY, "-W", "ignore", os.path.join("scripts", "make_package.py"),
                    "--verify-only", "--out", pkg])
    m = re.search(r"必须存在的 (\d+) 项中缺 (\d+) 项", out)
    return R("包内容自检", rc == 0,
             f"{m.group(1)} 项必需文件齐全" if (rc == 0 and m) else out.strip()[-160:])


def chk_dashboard_deps():
    """仪表板首屏依赖。上一版包漏了 `extreme_event.csv`，`streamlit run` 直接崩。"""
    code = ("import sys; sys.path.insert(0,'.');"
            "from vol_surface.backtest import load_events; e=load_events();"
            "assert len(e)>0, '事件表为空'; print(len(e))")
    rc, out = _run([PY, "-W", "ignore", "-c", code], timeout=120)
    # 取**最后一行非空输出**而不是 `[-140:]`——后者会把 traceback 切成
    # 「失败：s」这种信息量为零的残句（实测）。报错要能直接指出缺了什么。
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    why = lines[-1] if lines else f"rc={rc}，无输出"
    return R("仪表板首屏依赖可用", rc == 0,
             f"extreme_event.csv 读到 {out.strip()} 条事件" if rc == 0
             else f"load_events() 失败 → {why[:160]}")


def chk_platform_payload():
    """平台 payload 不得含 NaN/undefined。字段路径写错时语法门与 grep 都看不见。"""
    p = os.path.join(DATA, "platform.html")
    if not os.path.exists(p):
        return R("平台 payload 无 NaN/undefined", False, "platform.html 不存在")
    s = open(p, encoding="utf-8").read()
    i = s.find("const D = ")
    if i < 0:
        return R("平台 payload 无 NaN/undefined", False, "找不到 payload")
    i += len("const D = ")
    d, j = 0, i
    while j < len(s):
        if s[j] == "{":
            d += 1
        elif s[j] == "}":
            d -= 1
            if d == 0:
                j += 1
                break
        j += 1
    txt = re.sub(r"(?<![\w.])NaN(?![\w.])", "null", s[i:j])
    try:
        D = json.loads(txt)
    except Exception as e:                                   # noqa: BLE001
        return R("平台 payload 无 NaN/undefined", False, f"payload 解析失败：{e}")
    hits = []

    def walk(o, path):
        if isinstance(o, str):
            if re.search(r"\b(NaN|undefined|\[object Object\])\b", o):
                hits.append(path)
        elif isinstance(o, list):
            for k, v in enumerate(o):
                walk(v, f"{path}[{k}]")
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}")
    walk(D, "payload")
    return R("平台 payload 无 NaN/undefined", not hits,
             "干净" if not hits else f"{len(hits)} 处：{hits[:3]}")


def chk_platform_rebuild(fast):
    """平台必须能**重建出同一个文件**。无原始数据时用 --keep-surfaces。"""
    if fast:
        return R("平台可重建且逐字节一致", False, "", skipped=True)
    p = os.path.join(DATA, "platform.html")
    if not os.path.exists(p):
        return R("平台可重建且逐字节一致", False, "platform.html 不存在")
    import hashlib
    before = hashlib.sha256(open(p, "rb").read()).hexdigest()
    cmd = [PY, "-W", "ignore", os.path.join("scripts", "build_platform.py")]
    if not _has_raw():
        cmd.append("--keep-surfaces")
    rc, out = _run(cmd, timeout=600)
    if rc != 0:
        return R("平台可重建且逐字节一致", False, f"重建失败（rc={rc}）：{out[-160:]}")
    after = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return R("平台可重建且逐字节一致", before == after,
             f"sha256 {before[:12]}" if before == after
             else f"重建后变了：{before[:12]} → {after[:12]}")


def chk_no_cache_dirs():
    """交付目录不得混入隐藏的本地产物目录（缓存/临时）。"""
    bad = []
    for r, ds, fs in os.walk(DATA):
        for d in list(ds):
            if not (d.startswith(".") or "cache" in d.lower() or "tmp" in d.lower()):
                continue
            if d in {"anchor", "drl", "kg"}:
                continue
            full = os.path.join(r, d)
            sz = sum(os.path.getsize(os.path.join(rr, f))
                     for rr, _, fz in os.walk(full) for f in fz)
            bad.append((os.path.relpath(full, ROOT), sz / 2 ** 20))
    return R("交付目录无缓存/临时目录", not bad,
             "无残留" if not bad else
             "；".join(f"{p}（{s:.0f} MB{'，空壳' if s < 0.001 else ''}）"
                       for p, s in bad[:4])
             + "　→ 空壳目录也要删掉，它会被评委看到")


def chk_report_freshness():
    """生成型报告不得旧于它的数据源。跑完门不补跑报告就会红。"""
    # index.html 由 platform.html 的 payload 生成，故不得旧于它
    # doc_*.html 由各自的 .md 渲染，故不得旧于源文件
    pairs = [("../index.html", ["platform.html"]),
             ("doc_paper.html", ["../方法与结果.md"]),
             ("doc_drl_report.html", ["drl_report.md"]),
             ("doc_kg_report.html", ["kg_report.md"]),
             ("doc_readme.html", ["../README.md"]),
             ("doc_package.html", ["../PACKAGE.md"]),
             ("drl_report.md", [os.path.join("drl", "gates.json"),
                                os.path.join("drl", "results.json"),
                                os.path.join("drl", "workpoint.json")]),
             ("kg_report.md", [os.path.join("kg", "gates.json")])]
    bad = []
    for rep, srcs in pairs:
        rp = os.path.join(DATA, rep)
        if not os.path.exists(rp):
            bad.append(f"{rep} 不存在")
            continue
        rt = os.path.getmtime(rp)
        for s in srcs:
            sp = os.path.join(DATA, s)
            if os.path.exists(sp) and os.path.getmtime(sp) > rt + 1:
                bad.append(f"{os.path.basename(rep)} 旧于 {s}")
    return R("报告不旧于其数据源", not bad,
             "全部新鲜" if not bad else "；".join(bad[:3])
             + "　→ 顺序应为：门 → 报告 → 平台")


def chk_delivery_numbers():
    """交付结论必须能从落盘读出来，而且**每个达标都要配判别力**。"""
    try:
        q1 = json.load(open(os.path.join(DATA, "q1_7sym.json"), encoding="utf-8"))
        v2 = json.load(open(os.path.join(DATA, "metrics_v2.json"), encoding="utf-8"))
    except Exception as e:                                   # noqa: BLE001
        return R("交付数字可从落盘读出", False, f"{type(e).__name__}: {e}")
    sn = q1["summary_new"]
    parts = [f"问题1 {sn['v2']['P']:.1%}/{sn['v2']['R']:.1%}"
             f"（判别力 {sn['v2_discrimination_pp']:+.1f}pp，"
             f"{sn['v2']['n_pass3']}/{sn['n_win']} 窗）",
             f"问题3 {v2['drl']['test']['precision']:.1%}"
             f"（判别力 {v2['discrimination_pp']:+.1f}pp）"]
    # 判别力必须存在且为正——否则「达标」来自口径退化而非模型
    ok = (sn.get("v2_discrimination_pp") is not None
          and v2.get("discrimination_pp") is not None
          and sn["v2_discrimination_pp"] > 0 and v2["discrimination_pp"] > 0)
    return R("交付数字可读且判别力为正", ok, "；".join(parts))


# ---------------------------------------------------------------- 需人工确认

MANUAL = [
    ("平台在浏览器里打开无异常",
     "双击 data_out/platform.html，逐节滚一遍：不应出现 NaN / undefined / "
     "裸 Markdown（`**` 或反引号原样显示）/ 空白卡片。"
     "机器只查了 payload，**渲染后的样子只有人能看**。"),
    ("每个「达标」旁边都有判别力或对照",
     "本项目的新口径自带退化（最强平凡策略精确率恰好压在 50% 线上）。"
     "抽查 3 处「✓ 达标」，确认同屏能看到判别力或平凡策略对照。"),
    ("仪表板能起来",
     "`streamlit run dashboard/app.py`，确认首屏不报错、四个等级都出现过。"
     "本脚本只验了它的数据依赖，没有真起服务。"),
    ("提交物清单与赛题要求核对",
     "赛题对提交材料（正文文档 / 演示 / 命名 / 目录结构）的具体要求，"
     "本脚本无从知晓，需你自己对照通知核一遍。"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="跳过耗时门（随手自查用）")
    ap.add_argument("--json", default=None, help="把结果落盘为 JSON")
    a = ap.parse_args()

    where = "解压出来的包内" if _in_package() else ("源目录（有原始数据）"
                                                if _has_raw() else "源目录（无原始数据）")
    print(f"\n交付前验收 · {time.strftime('%Y-%m-%d %H:%M')} · 运行位置：{where}")
    if not _in_package():
        print("  ⚠ 真正算数的是**在解压出来的包里**跑一遍——"
              "本项目的交付事故全部只在那里暴露。")
    print("=" * 66)

    checks = [chk_delivery_numbers(), chk_doc_numbers(a.fast),
              chk_gates_drl(a.fast), chk_gates_kg(a.fast),
              chk_report_freshness(), chk_platform_payload(),
              chk_platform_rebuild(a.fast), chk_no_cache_dirs(),
              chk_dashboard_deps(), chk_package_exists(), chk_package_verify()]

    print("\n【机器可查】")
    for c in checks:
        print(f"  {c.mark}  {c.name:<26} {c.msg}")

    print("\n【需人工确认】——机器查不了，但评委会看到")
    for t, how in MANUAL:
        print(f"  ☐  {t}\n       {how}")

    hard = [c for c in checks if not c.ok and not c.skipped]
    # 区分两种「跳过」：`--fast` 主动跳过（**不足以下结论**）与
    # 「本环境不适用」（如包内的 mtime 判据，跳过是正确行为，不影响结论）。
    # 早前把两者混为一谈，全量跑时也会印出「--fast 跳过 1 项」——话说反了。
    skipped_fast = [c for c in checks if c.skipped and a.fast
                    and "不适用" not in c.msg]
    skipped_na = [c for c in checks if c.skipped and "不适用" in c.msg]
    print("\n" + "=" * 66)
    if hard:
        print(f"✗ 不可交付：{len(hard)} 项阻塞")
        for c in hard:
            print(f"    · {c.name}：{c.msg}")
        code = 1
    elif skipped_fast:
        print(f"○ 机器检查未跑全（--fast 跳过 {len(skipped_fast)} 项），不足以下结论。"
              f"\n  正式交付前请去掉 --fast，并**在解压包里**再跑一次。")
        code = 2
    else:
        print("✓ 机器检查全部通过"
              + (f"（{len(skipped_na)} 项本环境不适用，已跳过）。" if skipped_na else "。"))
        print(f"  仍有 {len(MANUAL)} 项需人工确认（见上），确认完即可交付。")
        code = 0 if _in_package() else 2
        if not _in_package():
            print("  ○ 但这是在源目录跑的。请解压交付包，在包内再跑一次本脚本。")

    if a.json:
        json.dump({"where": where, "fast": a.fast,
                   "checks": [{"name": c.name, "ok": c.ok, "skipped": c.skipped,
                               "msg": c.msg} for c in checks],
                   "manual": [t for t, _ in MANUAL], "exit": code},
                  open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n→ {a.json}")
    sys.exit(code)


if __name__ == "__main__":
    main()
