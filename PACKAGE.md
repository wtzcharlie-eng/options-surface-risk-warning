# 交付包说明

> 期权波动率曲面风控预警系统 —— 宁证期货赛题（问题1/2/3 全覆盖）
> 本文件说明**这个包里有什么、缺什么、什么能直接跑**。项目本身的完整说明见 `README.md`。

---

## 先看这个

**先打开 `index.html`**（解压后就在根目录）—— 交付入口页，60 秒看清三个问题
做到了什么、以及每个结论各自的对照。页面上没有一个手写数字，全部由
`scripts/build_index.py` 从落盘 JSON 现算。

再往里一层是 **`data_out/platform.html`** —— 研究总览，单文件零依赖，
全部过程与证据：两个评测口径、七条已证伪路径、逐品种对照、未达标项。

**`方法与结果.md`** 是学术结构的正文（摘要 / 数据与口径 / 方法 / 实验与结果 /
验证与可复现 / 局限与讨论）。`README.md` 保留为**完整工程记录**——逐轮修订、
含被推翻的结论——在正文文末作为附录链过去。两者共用同一批落盘数字，不会各说各话。

五份文档都有**与入口页同主题的深色 HTML 版**（`data_out/doc_*.html`，带目录与表格样式，
由 `scripts/build_docs.py` 渲染）。入口页卡片指向的是 HTML 版；
**`.md` 正本仍在包里**，那是可 grep、可 diff 的那一份。

仪表板的主题在 `.streamlit/config.toml`，与其余四处同一套色值。

---

## 包内容

| 目录 | 内容 |
|---|---|
| `vol_surface/` | 曲面清洗、插值、特征提取、预警规则（问题1 算法库） |
| `kg/` | 风险传导知识图谱：实测层标定 + 诠释层解释生成（问题2） |
| `drl/` | 纯 NumPy Double DQN 智能体、环境、指标、15 道验证门（问题3） |
| `scripts/` | 数据集构建、训练、评测、报告与平台生成脚本（含打包脚本 `make_package.py`） |
| `dashboard/` | Streamlit 多页交互系统：曲面预警 · 传导推理 · DRL 推理 · 系统总览 |
| `tests/` | 单元测试 |
| `data_out/` | 全部交付产物 + 标注数据集 + 15 个模型 checkpoint + 8 个 sklearn 模型 |

**`data_out/` 关键内容**

- `platform.html` —— 总览平台（自包含）
- `anchor/` —— 标注数据集，7 品种 / 263 幕 / 138,794 截面（**这是口径源**）
- `drl/agent_seed*.npz` —— 15 个训练好的 checkpoint（交付版集成）
- `drl/results.json`、`workpoint.json`、`gates.json` —— 问题3 的评测产物与验证门
- `kg/` —— 图谱边、跨品种边、解释样本、11 道门结果、自评记录
- **`rule_ml_model.joblib` + `.fingerprint.json`** —— 问题1 连续口径达标配置的模型，
  指纹含特征名与**顺序**（模型按列序吃数，顺序变了不报错、只静默给出错误概率）
- **`model_<品种>.joblib`（7 个）** —— 仪表板 ML 层的 IsolationForest，`G14` 逐个检查
- **`alert_engine_golden.json`** —— `G13` 的金标准夹具（600 例固定输入→期望输出）
- `*.json` —— 各项专题实验的落盘结果（死区、CVaR、漂移、集成规模、2022 覆盖度、
  异常分缺陷等）
- `*_preregistration.md` —— **判据写于跑数之前**的预登记文件
- `drl_report.md` / `kg_report.md` —— 由脚本从 JSON 生成的报告，非手写
- `backtest_report.md` —— **历史快照**（2026-08-03 的旧口径记录），文件顶部自带声明，
  其数字不作为交付结论

---

## 未包含（有意排除）

| 未含 | 说明 |
|---|---|
| `options_merged.parquet`（5.7 G） | 原始行情数据（2.26 亿行 / 35 品种） |
| `archive/`（6.2 G） | 按合约月拆分的原始数据 |
| `data_out/anchor_model.joblib` | 早期 anchor 预测器，**训练区间覆盖测试期（泄漏）**，已被 `rule_ml_model.joblib` 取代，不被任何结论引用 |
| `data_out/.bf_cache/` | `build_features.py` 的载入/特征缓存，只加速、删掉不影响任何结果 |
| `data_out/drl_more/`、`drl_legacy_monthly/`、`drl_evrew/`、`drl_ctrl_old/`、`roll/`、`drl_c26/`、`drl_5seed_archive/` | 已证伪路径的实验归档。其**结论**已落盘在顶层 JSON 里（如 `roll_retrain.json`、`ensemble_scaling.json`），归档本体不影响复现 |

> ⚠️ **上一版包曾整类排除 `data_out/*.joblib`**，理由写的是「早期 sklearn 路线，
> 不被任何交付结论引用」。那条理由**后来失效了**：`rule_ml_model.joblib` 成了问题1
> 连续口径达标的依据、`model_*.joblib` 成了仪表板 ML 层，两者都被验证门检查。
> 按旧规则打出的包，解压后 `G13`/`G14` 与指纹门**会当场失败**。
> 根因是**排除规则用通配符，而通配符的语义随时间漂移**——
> 现已改为 `scripts/make_package.py` 显式列出「必须存在」的文件并逐个断言，
> 且再加一道**从代码里推**的检查（源码字面引用的顶层数据文件必须在包里）——
> 因为手写清单同样会漏：实测就漏过 `extreme_event.csv`，仪表板首屏即崩而自检照样打 ✓。

> ⚠️ `README.md` 中若干处引用了 `data_out/drl_5seed_archive/`（旧交付版备份，用于新旧对照），
> 该目录不在本包内。相关**数字**仍可在 `README.md` 正文与 `ensemble_scaling.json` 中查到。

---

## 什么能直接跑

安装依赖：`pip install -r requirements.txt`（核心只需 numpy / pandas / pyarrow / scipy；
`drl/` 与 `kg/` 是**纯 Python + NumPy，零新增依赖**）

```bash
# 0) **先跑这个** —— 交付前一键验收，给出「能不能交付」的结论
#    在**解压出来的包里**跑才算数（本项目的交付事故全部只在那里暴露）
python scripts/preflight.py
#    随手自查可加 --fast（跳过耗时门，但**不足以下交付结论**）

# 1) 文档数字核对 —— 逐条比对 README 与方法学文档里的数字是否与落盘一致
python scripts/check_doc_numbers.py

# 2) 问题3 的 15 道验证门（必须带 --underlying，否则只载入 26 维、与 32 维交付模型不匹配）
python -m drl.tests --underlying --emit

# 3) 问题2 的 11 道验证门
python -m kg.tests --emit

# 4) 结构性死区的 2×2 对照（约 2.5 分钟）
python scripts/dead_zone_study.py --underlying

# 5) 逐品种评测（含「同召回」对照，约 2.5 分钟）
python scripts/eval_per_symbol.py --underlying

# 6) 异常分缺陷的量化复算（README §7.25，秒级）
python scripts/ml_score_defect.py

# 7) 重新生成总览平台与报告 —— **顺序要紧**：门 → 报告 → 平台
python -m drl.tests --underlying --emit && python -m kg.tests --emit
python scripts/report_drl.py && python scripts/report_kg.py
python scripts/build_platform.py --keep-surfaces   # 包内无原始数据，须带此参数
python scripts/build_paper.py && python scripts/build_docs.py && python scripts/build_index.py
#   正文 → 报告页 → 入口页（顺序要紧：后者读前者的产物）
#   不带 --keep-surfaces 时会**报错退出**（而不是静默生成一个少了曲面的版本）

# 8) 交互系统（四页：曲面预警 / 传导推理 / DRL 推理 / 系统总览）
cd dashboard && streamlit run app.py
#   主题在 .streamlit/config.toml，与入口页同一套色值
#   三个子页直接调用 vol_surface / kg.reason / drl.api 交付本体，不复制逻辑

# 9) 校验本包内容完整（不重新打包）
python scripts/make_package.py --verify-only
```

**需要原始数据才能跑的**（本包未含原始数据）：

- `scripts/build_anchor_dataset.py` —— 从 `options_merged.parquet` / `archive/` 重建 `anchor/`
- `scripts/build_features.py`、`build_underlying.py` —— 特征与标的侧特征的构建
- `scripts/train_drl.py` —— 从头训练（本包已附 checkpoint，无需重训）

---

## 交付前验收（`scripts/preflight.py`）

一条命令给出「能不能交付」的结论。**在解压出来的包里跑才算数**——
本项目历次交付事故（包漏文件、校验命令自己崩、平台被静默改坏）
在源目录里全都看不出来。

```bash
tar -xzf 期权曲面风控预警系统.tar.gz
cd 期权曲面风控预警系统
python scripts/preflight.py
```

**退出码**：`0` 可交付 · `1` 有阻塞项 · `2` 未跑全或还在源目录（不足以下结论）

| 机器可查（11 项） | 它防的是什么 |
|---|---|
| 交付数字可读且判别力为正 | 新口径自带退化，「达标」若无判别力就不构成证据 |
| 文档数字与落盘一致 | 手写数字随重跑过期 |
| 问题3 / 问题2 验证门 | 15/15、11/11 |
| 报告不旧于其数据源 | 跑完门不补跑报告 |
| 平台 payload 无 NaN/undefined | 字段路径写错，语法门与 grep 都看不见 |
| 平台可重建且逐字节一致 | 交付的 HTML 与 builder 已脱节 |
| 交付目录无缓存/临时目录 | 本地产物混进交付 |
| 仪表板首屏依赖可用 | 上一版包漏了 `extreme_event.csv`，`streamlit run` 直接崩 |
| 交付包不旧于源文件 | 改完忘了重打包（**只在源目录判**，包内不适用） |
| 包内容自检 | 63 项必需文件齐全 |

**机器查不了、但评委会看到的 4 项**，脚本会列出来让你逐条确认：
平台在浏览器里的**渲染结果**（NaN / 裸 Markdown / 空白卡片只有人能看见）、
每个「达标」旁边有没有判别力或对照、仪表板能不能真起来、
以及提交物清单与赛题通知的具体要求（命名、目录结构、正文文档形式）。

---

## 三个容易踩的坑

**1. `data_out/features_*.parquet` 不是口径源。**
那是给仪表板用的**采样版**（每品种约 400 行、7 品种合计约 2,479 行）。真正的口径源是
`data_out/anchor/`（138,794 行）。拿采样版去复核文档里的中位数、分位数会对不上，
**那不是文档写错，是文件选错**。

**2. 报数字必须标口径。** 同一套系统在两个评测口径下会给出差很多的结论：

| （**旧口径**：截面级、120min 命中窗） | 事件窗口（20 窗等权） | anchor 连续测试集 |
|---|---|---|
| 命中基础率 | 27.5% | 24.4% |
| 规则引擎精确率 | 50.1% | 45.4% |

来源：`q1_7sym.json.summary_new.base / .P`、`dead_zone_study.json.grid_2x2['wp_new|dz_off'].rule.precision`。
⚠ **不要用 `metrics_v2.json` 的 44.96% 填这一格**——那是**新口径**（事件级），
与本表其余三格的旧口径只差 0.4pp，极易蒙混过去。
前者是围绕已知极端事件挑的时段，门槛天然更松。
**两列的工作点也不同，不要相减**——README 与平台里每个数字都标了口径。

**3. `data_out/backtest_report.md` 是历史快照，不是交付结论。**
它生成于 2026-08-03，用的是旧风险起点口径、原始 7 条规则、仅 4 个已校准品种，
且当时的 ML 层存在缺陷（见 README §7.25）。文件顶部自带声明，
`check_doc_numbers.py` 有专门一道门守着那段声明不被删掉。
**交付数字见 `q1_7sym.json` 与 README §7.11 / §7.18。**

---

## 一句话交付状态

- **问题1（必选）**：**新口径**（事件级、召回不设上限）下 20 个事件窗口精确率 55.6% ✓ /
  召回 96.3% ✓ / 提前 1461.9min ✓，13/20 窗口三项全达标，**判别力 +5.6pp**
  （最强平凡策略「只在开头报一次」精确率就有 50.0%——**该口径自带退化，
  每个「达标」都必须配判别力**；召回与提前时间在此口径下门槛形同虚设，
  1461.9min 不应读作「提前 24 小时预警」）。
  未达标的 7 个**全部只卡精确率**；**卡在召回门槛（<60%）的窗口数为 0**。
  ⚠ 这**不等于零漏报**——20 个窗口里有 12 个召回不足 100%（最低 84.6%），
  确实漏掉了一些风险起点，只是没有任何窗口低到 60% 的门槛线以下。
  连续口径经规则+ML 否决式融合后 53.86% ✓ / 62.25% ✓ / 48.1min ✓
  （**纯规则够不到**，见 README §7.21/§7.23）
- **问题2（加分）**：21/40 主干边、110/280 跨品种边，11/11 验证门通过；人工 5 分制自评 12 轮，**不宣称达标**
- **问题3（加分）**：新口径三项全达标（64.6% / 69.2% / 6520min，判别力 +17.1pp）；旧口径经结构性死区修正后 54.15% ✓ / 74.93% ✓ / 36.2min ✓，15/15 验证门通过

> ⚠️ **死区过滤把旧口径精确率抬了约 7pp，但那不是模型改进**：
> 同工作点下 DRL 47.13% → 54.15%，规则 45.36% → 52.36%，**增益同等**，
> **同工作点**下 DRL +7.02pp vs 规则 +7.00pp，判别力仅变 **+0.02pp**。
> 预登记判据（2×2 对角线口径）为 −0.26pp，两者结论一致：**全员抬升**。详见 README §7.19。

全部未达标项与七条已证伪路径，见 `platform.html` 的「未达标项与已证伪路径」一节。
