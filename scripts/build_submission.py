"""Single deterministic source for current delivery prose and portal.

Read audit and frozen results only. --check is read-only and detects prose drift.
Historical scripts are archived as *_legacy.py and cannot overwrite these by
accident through the supported build_index/build_paper entry points.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import sys
from urllib.parse import unquote, urlsplit
from html.parser import HTMLParser
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_docs import DOCS, render_doc


def load(name):
    return json.loads((ROOT / name).read_text())


def pct(value):
    return f'{value:.2%}'


def metric_row(label, m):
    return f'| {label} | {pct(m["precision"])} | {pct(m["recall"])} | {m["avg_lead_min"]:.2f} | {m["n_alert"]:,} |'


def payload():
    a = load('data_out/submission/audit.json')
    requirements = load('docs/competition_requirements.json')
    q1, q3 = a['metrics']['rule_ml'], a['metrics']['drl']
    migration = load('data_out/rule_ml_portable.validation.json')
    historic = load('data_out/q1_7sym.json')['summary_new']
    v2 = load('data_out/metrics_v2.json')['drl']['test']
    gate = '；'.join(f'{c["name"].upper()} {c["n_passed"]}/{c["n_total"]}' for c in a['checks'])
    rows = '\n'.join(metric_row(f'{name} · {label}', a['metrics'][key][protocol])
        for name, key in [('Q1 规则+ML','rule_ml'), ('Q3 DRL','drl')]
        for label, protocol in [('有限窗严格提前','finite_strict'),('闭区间历史实现','legacy_closed'),('严格提前、召回无上限','unlimited_strict')])
    files = {}
    files['方法与结果.md'] = f'''# 期权波动率曲面风控系统：方法与结果

> 交付审阅版 · {requirements['review_date']}。本版将比赛基础任务、历史实验和 V3 扩展分开呈现。所有当前分钟级表格来自 `data_out/submission/audit.json`，生成入口为 `scripts/build_submission.py`。功能可运行不等于所有比赛指标已获证明。

## 摘要与任务边界

系统以曲面及 Greeks 为感知输入，提取有业务含义的特征，输出 0 正常、1 关注、2 警告、3 危险及原因。问题 1 由清洗、插值、无套利校验、规则与轻量模型完成；问题 2 提供图谱路径及证据；问题 3 用 DQN 的状态、动作、奖励学习预警策略。V3 的 1–5 个交易日概率预测与 LLM 证据问答属于额外研究扩展，不能替代赛题的分钟级指标和业务 CVaR。

附带赛题允许历史分析及 Web/Jupyter 界面，不强制实时行情。当前是离线历史原型，尚未接入实时行情或执行交易。

## 数据、切分与事件

内置锚点共 7 个品种、263 个月幕、138,794 个截面。按原方案分为训练 {a['split']['train_episodes']} 幕、验证 {a['split']['validation_episodes']} 幕、测试 {a['split']['test_episodes']} 幕；训练截至 2024-12，验证截至 2025-06。品种和测试日期逐幕列在审计 JSON 的 `coverage`。并非每品种都有完整测试期，尤其 si 在 V3 最新窗口没有测试样本。

分钟级事件沿用原实现：以 ATM IV 或凸性违反代理超过滚动 95% 分位，未来 5 根判定，风险起点按 120 墙钟分钟去重。历史窗口为 320 根，属于“约 20 日”的近似，不能把不同交易时段品种都当成精确 20 个交易日。事件用未来数据定义，只用于离线标签；模型状态不读取未来事件。

这批历史及测试结果已被多轮开发查看。冻结参数复核没有重新训练或选阈值，但**不构成外部盲测，也不再称“测试集仅评一次”**。事件窗口是围绕已知事件选取的样本，不等同连续测试集。

## 方法与部署配置

Q1 使用 `vol_surface/` 的清洗、插值、套利校验、曲面/Greeks 特征和规则引擎。当前复核的是规则阈值系数 {a['frozen_configuration']['rule_ml']['zmult']}、ML 否决阈值 {a['frozen_configuration']['rule_ml']['thr']} 的固定配置：ML 概率低于阈值时将规则等级压为 0。仪表板旧 IsolationForest 层和该否决模型是不同配置，不能把本表成绩直接赋给任意默认页面。

Q3 状态为 26 项曲面特征、6 项标的侧特征及预警上下文；动作为 0–3，奖励对正确预警、误报、漏报及提前量记账。采用 15 个冻结 DQN 的 Q 值集成，当前有限窗工作点 lean={a['frozen_configuration']['drl']['lean']}。状态归一化与训练参数随检查点保存；推理接口为 `drl.api.AlertAgentAPI`。原 API 默认 lean=0，复现本表必须显式调用 `set_confidence_threshold({a['frozen_configuration']['drl']['lean']})`，并使用同样的回放过滤。

两种配置均沿用历史休市过滤。它读取回放中下一观测时刻以识别长休市，**在线应用须改由预先已知的官方交易日历提供**，不能把离线过滤当作已验证的因果在线功能。逐月评测不跨月匹配，尾部完整未来窗未另行剔除；这些边界仍需下一批独立数据验证。

## 当前复核：三种匹配口径

赛题明确要求召回的预警严格早于风险起点，但没有写明召回最长提前时间，也未完全约定风险区间重叠的匹配方法。本版不把某种解释称为唯一官方算法。

- 有限窗严格提前（主要稳健性参考，非完整风险区间交集评分）：精确率匹配 `(预警, 预警+120min]` 的风险起点；召回匹配 `[风险起点−120min, 风险起点)`。同刻检测不算提前。
- 闭区间历史实现：精确率和召回都包含同刻，保留为旧成绩的复核对照。
- 严格提前、召回无上限：精确率仍限两小时；召回及提前量可匹配本月内任意早于起点的预警。它说明不设上限的影响，不证明多日预测能力。

汇总按预警与风险起点计数合并（micro）；平均提前量按被召回风险起点数加权。一条预警可对应多个风险，不构成独立伯努利样本。

| 配置与口径 | 精确率 | 召回率 | 平均提前/分钟 | 预警截面数 |
|---|---:|---:|---:|---:|
{rows}

三项参考线为精确率 ≥50%、召回 ≥60%、平均提前 ≥30 分钟。口径、窗口和休市过滤变化会显著改变成绩；不能从任一列推断全部比赛业务要求达标。精确率排除同刻且匹配起点是额外从严的假设，并非声称原文要求同刻必为误报。逐幕计数与来源指纹见审计 JSON。

## 历史成绩的适用范围

原事件窗口报告 Q1：精确率 {pct(historic['v2']['P'])}、召回 {pct(historic['v2']['R'])}、平均提前 {historic['v2']['lead']:.0f} 分钟，采用事件合并及召回无上限口径。原 Q3 稀疏工作点：{pct(v2['precision'])} / {pct(v2['recall'])} / {v2['avg_lead_min']:.0f} 分钟，工作点与本版有限窗配置不同。原材料记载这是后续答复口径，但本次未收到答复原件，故仅作历史实验假设保留。

原文将窗口宏平均精确率与全部预警数混合做两比例 z 检验，而且时间样本有依赖。本版撤下该显著性论证；现有差异均为开发数据上的描述性结果。尚未建立按事件/日期配对的独立检验，不能称统计显著。

历史奖励比较亦须对应相同策略、过滤与奖励函数。规则累计奖励为负时，用其作为分母的“提升百分比”会失去通常业务含义；因此不据正负奖励差自动认定 DRL 达到加分项 ≥10%。

## 核心业务指标：CVaR 证据仍待补齐

赛题要求：不少于 3 个月连续测试，分别初始多头和空头各 1 手；首次 level≥2 预警后的下一个交易日开盘减半，之后不恢复、不重复减仓；计算多空日收益 CVaR95 的相对改善并取算术平均。无预警改善为 0，超过 10% 为显著有效的参考线。

原 `cvar_drl.json`、`cvar_workpoint.json` 使用滚动 5 日减仓再恢复；原价格脚本还以收盘价代开盘价。旧 13.35% 或 49.92% 的改善不能当作上述比赛协议结果。本版不填造新的比赛改善率。

已补 `vol_surface.submission_metrics.first_alert_cvar` 与 `scripts/evaluate_submission_cvar.py`：真实交易日开盘/收盘输入，首次预警一次减半，跨开盘跳空先按原仓位计算，再按半仓计算日内变化；后续保持半仓；多空分别统计。收益以初始固定名义本金归一化（对应假想 1 手的价格损益），CVaR95 对最差 5% 日损失按概率质量精确加权。基准尾部损失非正时相对改善不可定义，返回空值，不强行出正收益结论。未计费用、滑点、保证金或合约换月影响。

真实开盘价来源、夜盘交易日映射、交易日连续性与连续合约处理尚待提供。算法测试通过只证明规则实现，不能代替这些输入证据。

## 图谱解释、人工评分与案例

图谱包含特征、Greeks、标的及风险因子；提供 Schema、查询、动态边权更新、路径和可追溯证据。统计边及历史领先关系不构成已识别的因果效应，解释使用“可能影响”而非确定性因果判断。

KG 机器核验当前为 {next(c['n_passed'] for c in a['checks'] if c['name']=='kg')}/11。现有解释样本与自评能证明流程，不能自动证明可核验的人工平均 ≥4/5。请使用 `docs/人工解释评分表.csv` 补充评分者、样本、清晰度、逻辑、无交易建议和总分，并保留评语；不能把空评分行当作达标。

本次另外保留了严格有限窗下的命中与未匹配样例（首个符合条件的记录，非人工挑选）。样例时间、等级、当时曲面读数、事后风险起点和来源路径见审计 JSON 的 `case_matches`。风险起点属于离线标签，只用于核对，不是当时可见信息。

典型历史曲面、预警时间轴和推理路径可在 `data_out/platform.html`、`data_out/doc_dashboard.html`、`data_out/kg/kg_viz.html` 及 `data_out/kg_report.md` 查阅。这些页面属于历史展示；其中旧成绩应结合本报告的口径限制阅读。

## V3 扩展与证据问答

V3 输出未来 1–5 个交易日的曲面风险代理概率，沿用三轮时间隔离验证、校准和稳健风险率融合。最近一轮 528 个共同原点，1/2/3/4/5 日对 V2 的 Brier 降低约 2.9%/5.0%/6.6%/7.1%/7.7%；较早窗口部分期限变差，对自身强基准的 3–5 日技巧仍为负。它不是价格预测、账户损失概率或已验证的 CVaR 改善。

LLM 通过 OpenAI 兼容接口配置，回答引用本地证据字段，无 Key 可用本地解读。真实云端 API 尚需用户配置验证。详见 `README.md` 与 `data_out/predictive_v3/report.md`；本次不改 V1/V2/V3 算法和产物。

## 运行修复与复现

时间戳转换现在显式转为秒，消除了 pandas 推断微秒后被当成纳秒的错误。修复前 DRL 9/15，修复后 {gate}；未修改奖励参数、阈值或检查点。问题 1 冻结模型在原 scikit-learn 1.7.2 环境离线导出数值树，逐条核对 {migration['n_compared_rows']:,} 行概率，最大误差 {migration['max_probability_error']:.1g}，未训练；当前复算直接读取 JSON，避免旧 pickle 私有模块路径问题。

```sh
python -m pip install -r requirements-predictive.lock.txt
python scripts/preflight.py
python scripts/audit_submission.py
python scripts/build_submission.py --check
python scripts/make_package.py --out ../期权曲面预测风控系统.zip
```

`preflight` 只读核对数据、生成文档与运行门；运行检查通过不表示比赛业务指标齐备。`audit_submission` 重算分钟级指标并写入新的审计目录；它不修改历史模型及预测报告。原交付的逐字节快照在 `docs/legacy_delivery/`，赛题要求摘录和文件指纹见 `docs/competition_requirements.json`，未将含联系信息的赛题 DOCX 放入公开仓库。
'''
    files['docs/比赛要求对照.md'] = f'''# 比赛要求与验收证据

审阅日期：{requirements['review_date']}。依据为用户提供的原始赛题说明。源文件 SHA-256：`{requirements['source']['sha256']}`。仅记录任务要求及文件指纹，公开包不含原 DOCX 或联系信息。

## 任务与成果形式

| 要求 | 本项目证据 | 当前状态 |
|---|---|---|
| Q1 必选：清洗、插值、无套利、曲面/Greeks 风险特征 | `vol_surface/cleaning.py`、`interpolation.py`、`arbitrage.py`、`features.py` | 算法库与锚点数据已提供；原始行情未随包提供 |
| Q1 四级预警及原因 | `vol_surface/alert_engine.py`、`alert_rules.py` | 已实现；否决模型成绩须使用指定配置 |
| 曲面热力图、Greeks、预警时间轴 | `dashboard/app.py`、`data_out/doc_dashboard.html` | 交互与静态预览已提供；精简包缺完整原始合约 Greeks，交互页明确提示 |
| 可分析历史数据、Web/Jupyter | 两套 Streamlit 工作台与离线入口 | 已实现；原文不强制实时接入 |
| Q2 可选：Schema、图谱查询和路径 | `kg/schema.md`、`kg/`、`data_out/kg/kg_viz.html` | 代码与界面已提供，机器检查 {next(c['n_passed'] for c in a['checks'] if c['name']=='kg')}/11 |
| Q2 人工平均评分 ≥4/5 | 历史自评记录、`人工解释评分表.csv` | 待补可核验人工记录，不以机器检查替代 |
| Q3 可选：状态/动作/奖励、训练记录、基线和推理 API | `drl/`、`data_out/drl/results.json`、`drl/api.py` | 已实现，当前机器检查 {next(c['n_passed'] for c in a['checks'] if c['name']=='drl')}/15 |
| Q3 相对固定规则累计奖励或 CVaR 改善 ≥10% | 历史奖励及 CVaR 实验 | 不直接认定通过：负奖励分母与 CVaR 协议差异需说明 |
| 主指标：P≥50%、R≥60%、平均提前≥30min | `../方法与结果.md`、`../data_out/submission/audit.json` | 三种声明口径并列；严格有限窗为稳健性参考 |
| 核心业务：多空 CVaR95 改善、首次预警下个交易日开盘减半不恢复 | `../vol_surface/submission_metrics.py`、`../scripts/evaluate_submission_cvar.py` | 计算规则已补；真实开盘/交易日/连续合约数据证据未齐，当前未核验 |
| 源码、依赖、部署指南 | `../README.md`、`../PACKAGE.md`、锁定依赖与启动脚本 | 当前包收录并有 SHA-256 清单 |
| 离线比较、关键指标、典型案例 | `../方法与结果.md`、审计 JSON、历史图谱与仪表板案例 | 已提供；开发验证，不是盲测 |
| 1–5 日预测及 LLM 交互 | `../dashboard/predictive_app.py`、V3 报告 | 额外扩展，不替代比赛主指标 |

## 对口径歧义的处理

原赛题明确精确率在预警后两小时内检查风险，召回要求预警早于风险起点，但没有写清召回上限、告警合并和所有区间重叠细节。原交付声称后续答复采用事件合并及召回无上限；答复原件此次未提供，所以保留历史结果并单列，不能将其描述为附带赛题已明确的条款。

本版另算严格早于且限制两小时的稳健性参考，使用风险起点匹配。它不是自行修改官方评分标准，最终口径需由比赛评审确认。长达数千分钟的无上限提前量不代表已经具有数日预测能力。

## 仍需补齐的证据

1. 真实交易日开盘、收盘、夜盘归属和连续合约处理，覆盖不少于 3 个月且无缺失交易日；按首次预警永久减半协议复算多空 CVaR。
2. 如提交后续口径成绩，补充原答复的可核验记录，注明适用范围和版本。
3. 补充解释文本人工评分与评语，并给出样本选择和评分者信息；不虚构分数。
4. 锁定之后的新时间段盲测；在线休市过滤改用官方交易日历。

这些是证据或数据缺口，不应通过改写说明为“全达标”消除。完整机器协议、当前得分与历史对照见根目录方法报告。
'''
    files['PACKAGE.md'] = f'''# 交付包说明

本包是比赛原型与 V3 预测扩展的统一交付。先打开 `index.html` 阅读比赛对照，再按需要启动交互系统。运行门已复核：{gate}；这不等同比赛业务验收全部通过。

## 内容与版本

- `方法与结果.md`：分钟级固定配置的三种匹配口径、原结果解释、CVaR 缺口、V3 研究边界。
- `docs/比赛要求对照.md`：逐项对应原始赛题、源码/界面/报告证据与未完成证据。
- `docs/交付审阅与修订.md`：原交付问题、修复与追溯记录。
- `vol_surface/`、`kg/`、`drl/`、`dashboard/`、`scripts/`、`tests/`：算法、界面与验证。
- `data_out/anchor/`：7 品种、263 个月幕、138,794 截面，可离线重放。原始行情和大体积原档不在包内，不能声称从源行情完全重建。
- `data_out/drl/agent_seed*.npz`：15 个冻结 DQN；`rule_ml_portable.json` 为 Q1 数值树，保留源 joblib 及迁移校验记录。
- `data_out/predictive*`：冻结 V1/V2/V3 模型、报告与逐条预测；`docs/` 含相应实验设计与验收。
- `README.legacy.md`、`docs/legacy_delivery/` 和 `scripts/*_legacy.py`：原工程记录、原交付快照和旧构建逻辑，仅用于历史追溯。
- `requirements-predictive.lock.txt`、`.env.example`、`启动系统.command`：依赖、配置模板与本地启动。真实 Key、`.env`、Git 元数据、缓存及日志不收录。

## 安装与运行

建议 Python 3.12，在解压后的项目目录运行：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-predictive.lock.txt
python scripts/preflight.py
```

比赛原型（热力图、Greeks、时间轴、图谱及 DRL）：

```sh
python -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

V3 扩展（1–5 个交易日预测、版本比较、证据问答）：

```sh
python -m streamlit run dashboard/predictive_app.py --server.address 127.0.0.1 --server.port 8502
```

macOS 也可双击 `启动系统.command` 启动 V3。OpenAI 兼容服务按 `README.md` 填 Base URL、模型 ID 和自己的 Key；无 Key 时本地问答可用。真实远程服务尚待用户配置联通。

## 复算和材料更新

```sh
python scripts/audit_submission.py
python scripts/build_submission.py
python scripts/build_submission.py --check
python -m pytest tests/test_competition_time.py tests/test_submission_metrics.py tests/test_submission_delivery.py -q
python scripts/audit_predictive_v3.py
python scripts/smoke_competition.py
```

`audit_submission` 不训练、不选择新参数，写入 `data_out/submission/`，保存当前环境、逐幕覆盖、两种模型三套协议与来源指纹。`build_submission` 仅从证据生成当前 MD/HTML；兼容入口 `build_paper.py`、`build_index.py` 均转向它，避免重新生成旧成绩封面。

`scripts/preflight.py` 默认为只读全检查；`--fast` 跳过耗时运行门并明确标记未执行。程序正常结束只表示执行的机器检查通过，未证明 CVaR 或人工评分过线。

Q1 旧 joblib 采用 scikit-learn 1.7.2，且包含私有 `_loss` 模块路径，不能在当前环境直接加载。日常复算使用可移植 JSON。若必须重新导出，在隔离环境安装 `scikit-learn==1.7.2` 和 pandas/numpy/pyarrow/joblib，再运行 `scripts/export_rule_ml.py`；导出器不会训练，并对全量锚点预测逐条比对，失败不保留产物。源模型仅作为可信原项目资料保留，不加载未知来源 pickle。

## CVaR 专用输入

`prices.csv` 需要 `trading_date,open,close`，交易日由交易所日历确认、开盘必须真实、至少覆盖 3 个月连续交易记录。`alerts.csv` 需要 `decision_trading_date,level`，夜盘先映射至真实所属交易日。不同品种分别运行，不能将其价格串接。样例字段见 `docs/cvar_prices.schema.csv` 和 `docs/cvar_alerts.schema.csv`，只有表头，不含虚构成绩。

```sh
python scripts/evaluate_submission_cvar.py --prices prices.csv --alerts alerts.csv --out cvar_result.json
```

工具只验证计算规则和可检查的输入约束，无法凭日期列证明交易日连续性、真实开盘或合约处理。缺乏来源证据时不得作为正式比赛达标证明。当前包没有已核验的原赛题 CVaR 改善率。

## 打包与接收方验收

```sh
python scripts/make_package.py --out ../期权曲面预测风控系统.zip
python scripts/make_package.py --verify-only --out ../期权曲面预测风控系统.zip
```

包内 `MANIFEST.sha256.json` 记录所有收录文件的 SHA-256。打包器检查必要文件、排除本地凭据和缓存，再解压到临时目录，逐字节核对清单并在解压副本运行文档一致性检查。接收方在解压根目录再次运行 `python scripts/preflight.py`；它不需要原始行情，也不需要 API Key。历史构建器可能需要原始行情或旧目录结构，不属于当前验收路径。

## 必须随成绩一起说明

当前分钟级参考为开发历史复核，含逐月边界和基于回放时间表的休市过滤。旧事件窗口、召回无上限、5 日恢复 CVaR 均不能与当前参考成绩混用。图谱机器检查不等于人工 ≥4 分；V3 曲面概率不等于账户损失概率。详见比赛要求对照。
'''
    files['docs/交付审阅与修订.md'] = f'''# 原交付审阅与修订

本次按用户提供的赛题 DOCX、原 PACKAGE、方法结果和 HTML 入口审阅。下载目录中的原件没有修改；项目内的三份逐字节快照位于 `legacy_delivery/`，校验和见 `competition_requirements.json`。

| 原交付问题 | 本版修订 | 验证依据 |
|---|---|---|
| 无上限召回和上千分钟提前量作为封面主成绩 | 正文分列三种口径；分钟级参考和 V3 多日预测分开 | `../data_out/submission/audit.json` |
| 把后续答复当作原赛题确定条款 | 明示本次缺少答复原件，历史口径保留待核验 | 赛题 SHA 与要求摘录 |
| 宏平均窗口分数配总预警数做 z 检验 | 撤下统计显著性结论，保留描述性比较 | 新方法报告“历史成绩”节 |
| “测试只评一次”和大量开发实验并存 | 统一写成已查看历史上的开发复核 | 新方法报告与冻结参数 |
| 滚动 5 日恢复 CVaR 与赛题永久半仓混用 | 分开标注，补首次预警永久半仓工具；真实开盘证据未齐 | 新 CVaR 单元验证 |
| 当前环境 DRL 仅 9/15 | 显式时间单位换算，恢复 {gate}；不调参数 | 验证日志与时间回归测试 |
| Q1 旧 pickle 私有模块无法加载 | 在原 1.7.2 环境导出数值树，当前只读 JSON | 全量 {migration['n_compared_rows']:,} 行、概率误差 {migration['max_probability_error']:.1g} |
| 系统总览路径指向 dashboard/data_out，页面静默无数据；DRL 页面漏读 6 项标的特征 | 修正根目录并展示当前审计；DRL 改用完整 32 维锚点、明确单截面上下文 | 页面内容断言与烟测 |
| 新 README 与旧构建器/验收器路径脱节 | 新统一生成器，旧入口转接；旧脚本留档 | `build_submission.py --check` |
| 打包漏设计文档、锁定依赖和新入口 | 扩展收录清单、包内逐文件 SHA、解压副本复核 | `make_package.py --verify-only` |
| 文件存在被当作功能和指标通过 | 分开报告已实现、已复核、待证据 | 比赛要求对照 |

## 未改变与未完成

V1/V2/V3 算法、检查点及预测评估产物保留原样。DRL 数值检查点、奖励参数和工作点保留原样。修复影响的是旧分钟级运行对时间单位的解释，以及旧 Q1 模型在新环境的读取方式。

尚未获得真实交易日开盘与连续合约资料、后续口径答复原件、完整可核验人工评分和新的外部盲测。没有填造这些证据，也没有改称全指标达标。建议优先补核心业务 CVaR 的输入证据，再做锁定新时间段测试。
'''
    files['index.html'] = portal(a, gate)
    # Render current source prose; historical technical report pages carry a banner.
    for src, filename, title, sub in DOCS:
        rel = str(Path(src).relative_to(ROOT))
        md = files.get(rel)
        if md is None:
            md = Path(src).read_text()
        files['data_out/' + filename] = render_doc(src, title, sub, md)
    return files


def portal(a, gate):
    cards = ''
    for key, title, desc in [('rule_ml','Q1 · 规则与轻量模型','固定否决配置'),('drl','Q3 · 深度强化学习','15 个冻结 DQN')]:
        m = a['metrics'][key]['finite_strict']
        cards += f'''<article class="card"><span class="eyebrow">{title}</span><div class="number">{pct(m['precision'])}</div><p>有限窗严格提前 · 精确率</p><dl><div><dt>召回率</dt><dd>{pct(m['recall'])}</dd></div><div><dt>平均提前</dt><dd>{m['avg_lead_min']:.1f} 分钟</dd></div></dl><small>{desc} · 开发历史复核</small></article>'''
    matrix = ''.join(f'<tr><th>{name}</th><td>{value}</td><td>{status}</td></tr>' for name,value,status in [
        ('Q1 基础预警','清洗 / 无套利 / 曲面与 Greeks / 四级预警','已实现 · 配置须对应'),
        ('Q2 解释与传导','图谱路径、Schema、证据链','机器检查通过 · 人工评分待补'),
        ('Q3 自适应决策','冻结 DQN、基线、训练记录及推理 API','机器检查通过 · 加分幅度待核验'),
        ('核心业务 CVaR','首次预警 → 下个交易日开盘减半 → 持有至期末','复算规则已补 · 输入证据未齐'),
        ('预测与交互扩展','1–5 个交易日风险概率 + LLM 字段问答','V3 研究版 · 额外成果')])
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>期权曲面风控 · 比赛交付审阅版</title><style>
:root{{color-scheme:dark;--bg:#0b1420;--panel:#121f2d;--line:#2a3949;--muted:#a0b0c0;--text:#edf3f8;--mint:#77ddbf;--amber:#f0c478}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.7}}a{{color:var(--mint);text-decoration:none}}a:hover{{text-decoration:underline}}a:focus-visible{{outline:2px solid var(--mint);outline-offset:4px}}header,main,footer{{max-width:1180px;margin:auto;padding:24px 30px}}header{{display:flex;gap:20px;justify-content:space-between;border-bottom:1px solid var(--line);font-size:14px}}header span{{color:var(--muted)}}nav{{display:flex;gap:20px;flex-wrap:wrap}}.hero{{padding:42px 0 28px;max-width:920px}}.eyebrow{{font-size:12px;letter-spacing:.12em;color:var(--mint)}}h1{{font-size:clamp(30px,5vw,48px);line-height:1.28;letter-spacing:-.03em;margin:18px 0}}h2{{font-size:23px;margin:0 0 18px}}h3{{margin:0 0 8px}}p{{color:var(--muted);margin:10px 0}}.hero p{{font-size:17px;max-width:850px}}.actions{{display:flex;flex-wrap:wrap;gap:12px;margin-top:22px}}.button{{border:1px solid var(--line);padding:10px 18px;border-radius:7px;color:var(--text);background:var(--panel)}}.button.primary{{background:var(--mint);color:var(--bg);font-weight:650}}.notice{{border:1px solid #675438;border-left:4px solid var(--amber);background:#242319;padding:17px 20px;border-radius:8px;margin:4px 0 26px;color:#f2dfb4}}.notice strong{{color:var(--amber)}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}}.card{{border:1px solid var(--line);border-radius:10px;background:var(--panel);padding:22px}}.card p{{font-size:13px}}.number{{font-size:39px;line-height:1.3;margin-top:17px;font-variant-numeric:tabular-nums}}.number.word{{font-size:28px}}small{{color:var(--muted);font-size:12px}}dl{{margin:19px 0 14px}}dl div{{display:flex;gap:8px;justify-content:space-between;font-size:13px}}dt{{color:var(--muted)}}dd{{margin:0}}section{{margin-top:42px}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:8px}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{padding:14px 17px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}}th{{font-weight:600;min-width:150px}}td{{color:var(--muted)}}tr:last-child td,tr:last-child th{{border:0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.detail{{border:1px solid var(--line);padding:23px;border-radius:8px}}.detail p{{font-size:14px}}.detail a{{display:inline-block;margin:10px 18px 0 0}}code{{background:#1d2c3d;padding:2px 5px;border-radius:3px;font-size:12px;overflow-wrap:anywhere}}footer{{border-top:1px solid var(--line);font-size:12px;color:var(--muted);margin-top:40px}}.tag{{display:inline-block;padding:3px 9px;border:1px solid var(--line);border-radius:20px;font-size:12px;color:var(--muted);margin-bottom:10px}}@media(max-width:980px){{.cards{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:620px){{header,main,footer{{padding:20px}}header{{display:block}}nav{{margin-top:12px;gap:14px}}.hero{{padding-top:28px}}.cards,.grid{{grid-template-columns:1fr}}.number{{font-size:38px}}td,th{{padding:12px}}}}
</style></head><body><header><b>宁证期货赛题 · 期权曲面风控</b><nav><a href="data_out/doc_requirements.html">比赛对照</a><a href="data_out/doc_paper.html">方法与结果</a><a href="data_out/doc_package.html">运行与交付</a></nav><span>审阅版 · 2026.10.23</span></header><main>
<div class="hero"><span class="eyebrow">感知 → 分级预警 → 传导解释 → 决策优化</span><h1>让预警有证据，<br>让预测可复核。</h1><p>以期权波动率曲面与 Greeks 识别分钟级异常；通过知识图谱解释传导路径，以强化学习优化预警策略，并扩展未来 1–5 个交易日的风险概率与交互问答。</p><div class="actions"><a class="button primary" href="data_out/doc_requirements.html">查阅比赛要求与验收证据</a><a class="button" href="data_out/doc_delivery_review.html">查看原交付修订</a><a class="button" href="data_out/doc_package.html">安装与启动工作台</a></div></div>
<div class="notice"><strong>交付状态：</strong>基础、图谱和 DRL 原型已提供，运行核验 {html.escape(gate)}。赛题核心 CVaR 的真实开盘与交易日证据、可核验人工评分仍待补齐，当前不宣称全部指标达标。</div>
<div class="cards">{cards}<article class="card"><span class="eyebrow">预测与交互扩展</span><div class="number">1–5 <small>日</small></div><p>V3 曲面风险代理概率</p><dl><div><dt>对照</dt><dd>冻结 V1 / V2</dd></div><div><dt>问答</dt><dd>本地 / 兼容 LLM</dd></div></dl><small>开发验证 · 非账户损失概率</small></article><article class="card"><span class="eyebrow">核心业务证据</span><div class="number word" style="color:var(--amber)">CVaR 待核验</div><p>按赛题首次预警永久减半协议</p><dl><div><dt>计算规则</dt><dd>已补</dd></div><div><dt>真实开盘资料</dt><dd>待补</dd></div></dl><small>旧滚动 5 日实验不能替代</small></article></div>
<section><h2>比赛任务与当前证据</h2><div class="table-wrap"><table aria-label="比赛任务验收对照"><thead><tr><th>任务</th><th>已提供内容</th><th>当前状态</th></tr></thead><tbody>{matrix}</tbody></table></div><p><small>封面数字采用有限窗严格提前参考：精确率匹配后续两小时的风险起点，召回仅计起点前两小时预警。召回上限等歧义尚待正式确认；正文同时保留闭区间与无上限对照。</small></p></section>
<section class="grid"><article class="detail"><span class="tag">当前报告</span><h3>从赛题到可复现证据</h3><p>数据与切分、固定配置、逐项指标、方法限制、原交付修订和来源指纹。</p><a href="data_out/doc_paper.html">方法与结果 →</a><a href="data_out/submission/audit.json">机器可读审计 →</a><a href="data_out/doc_delivery_review.html">修订说明 →</a></article><article class="detail"><span class="tag">研究扩展</span><h3>V3 预测与证据问答</h3><p>五个期限保持概率一致，支持版本比较和字段引用问答。较早验证窗部分期限退步，不能宣称全面提升。</p><a href="data_out/doc_readme.html">运行与配置 →</a><a href="data_out/predictive_v3/report.md">V3 完整评估 →</a></article><article class="detail"><span class="tag">历史展示 · 请结合新报告阅读</span><h3>曲面与分级预警</h3><p>保留原曲面热力图、Greeks 和预警时间轴。历史页面中的旧口径成绩不是当前比赛验收结论。</p><a href="data_out/doc_dashboard.html">仪表板静态预览 →</a><a href="data_out/platform.html">原曲面展示 →</a></article><article class="detail"><span class="tag">图谱与决策设计</span><h3>推理路径与智能体</h3><p>节点、边语义、路径证据、DRL 状态与动作、奖励和推理接口。统计关联不作为确定因果。</p><a href="data_out/kg/kg_viz.html">图谱可视化 →</a><a href="data_out/doc_kg_schema.html">Schema →</a><a href="data_out/doc_drl_report.html">历史 DRL 报告 →</a></article></section>
<section><h2>接收方如何复现</h2><p>先打开本入口离线阅读，再依照交付说明安装锁定依赖。执行 <code>python scripts/preflight.py</code> 复核运行和材料一致性；执行 <code>python scripts/audit_submission.py</code> 重算分钟级证据。两套交互系统分别使用本机 8501 / 8502 端口。</p><p><a href="data_out/doc_package.html">查看完整运行、打包与 CVaR 输入说明 →</a></p></section>
</main><footer>历史研究原型 · 只输出风险提示 · 无交易执行。当前 MD 与 HTML 由统一生成器从冻结结果和当前审计生成；原件指纹及赛题要求见 <a href="docs/competition_requirements.json">来源记录</a>。原始行情和真实 API Key 不包含在交付包中。</footer></body></html>'''


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.refs=[]
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.refs.extend(v for k,v in attrs if k=='href')


def check_links(path, content):
    parser = Links(); parser.feed(content)
    bad = []
    for href in parser.refs:
        url = urlsplit(href)
        if url.scheme or url.netloc or not url.path: continue
        target = (path.parent / unquote(url.path)).resolve()
        if not target.is_relative_to(ROOT) or not target.exists(): bad.append(href)
    return bad


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); args=ap.parse_args()
    files=payload(); failures=[]
    for name, content in files.items():
        path=ROOT/name
        if args.check:
            if not path.exists() or path.read_text()!=content: failures.append('内容过期: '+name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content, encoding='utf-8')
    # Validate the active entry and current authored prose; historical appendices
    # deliberately retain old research links and are clearly labelled.
    for name in ['index.html','data_out/doc_paper.html','data_out/doc_package.html','data_out/doc_requirements.html','data_out/doc_delivery_review.html']:
        failures.extend(f'{name}: 链接失效 {x}' for x in check_links(ROOT/name,files[name]))
    for name, expected in load('data_out/submission/audit.json')['source_sha256'].items():
        if not (ROOT/name).exists() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            failures.append('审计来源变动: '+name)
    if failures: raise SystemExit('\n'.join(failures))
    print(f'{len(files)} 份当前交付材料一致；活动入口与当前报告链接有效；审计来源指纹一致。')


if __name__=='__main__': main()
