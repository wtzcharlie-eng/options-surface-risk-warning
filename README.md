# 期权曲面预测风控与交互系统

[![Predictive risk checks](https://github.com/wtzcharlie-eng/options-surface-risk-warning/actions/workflows/predictive-checks.yml/badge.svg)](https://github.com/wtzcharlie-eng/options-surface-risk-warning/actions/workflows/predictive-checks.yml)

在原项目上新增未来 **1–5 个交易日**的曲面风险预测，并通过可配置的 **OpenAI 兼容 LLM** 进行证据问答。当前内置历史数据回放，尚未连接实时行情。

GitHub 仓库：[wtzcharlie-eng/options-surface-risk-warning](https://github.com/wtzcharlie-eng/options-surface-risk-warning)。克隆后进入项目目录即可按下方步骤启动。每次推送到 `main` 或提交 Pull Request，GitHub Actions 会运行新增预测/LLM 测试、模型校验和检查及交互页面测试；自动测试仅使用本地或模拟 LLM，不需要 API Key。

## 启动

macOS 双击 `启动系统.command`，首次启动会安装依赖。推荐 Python 3.12；浏览器入口为 <http://127.0.0.1:8502>。

或在项目目录运行：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-predictive.lock.txt
python -m streamlit run dashboard/predictive_app.py --server.address 127.0.0.1 --server.port 8502
```

工作台可选择品种、历史观察日和 1/2/3/4/5 交易日预测周期，查看风险概率、阈值、信号贡献、时间外评估，并与风险助手连续对话。切换观察上下文会开启新对话。主界面重点展示 1、3、5 日三个期限。

## 配置大模型

在侧栏选择“OpenAI 兼容 API”，填写服务商提供的 Base URL、模型 ID、API Key，即可发送问题。也可以把 `.env.example` 复制为 `.env`：

```dotenv
RISK_LLM_PROVIDER=openai-compatible
RISK_LLM_BASE_URL=https://你的服务商提供的兼容地址/v1
RISK_LLM_MODEL=服务商的模型ID
RISK_LLM_API_KEY=在本机填写真实密钥
```

Base URL 是否包含 `/v1` 以服务商文档为准；系统自动追加 `/chat/completions`。本机服务选择 `local-openai`，使用回环地址（例如 `http://127.0.0.1:11434/v1`）及已经安装的模型 ID。本地服务允许显式不填 Key。

默认 `local` 模式不联网，也不会假装调用大模型。远程 API 的问题会包含当前风险快照和近期对话；Key 不写入报告、预测文件或聊天记录。本次已验证本地功能与模拟 API 请求，真实服务需用用户自己的配置联通。

助手回答附 `[forecast.probability]` 等字段引用，可在“查阅回答引用的字段”中核对。未知引用或缺乏证据的数字会触发本地解读回退。字段校验并不能保证所有自然语言解释绝对正确，风险结论仍应核对原始快照与评估。

## 预测目标与成绩

预测事件是：在未来 H 个完整观测交易日中，任一天的 ATM 隐波或曲面凸性违反代理最大值，超过**预测时点之前**最近 60 个完整交易日（至少 20 日）的 95% 分位阈值。阈值在观察时点冻结；事件包括异常状态延续，不只包括首次爆发。

每个预测以已完成日盘收盘为基准；夜盘归下一观测到的日盘日期。当前没有官方交易所日历，缺失日期和长空档采取保守剔除。具体规则及覆盖率见 [模型报告](data_out/predictive/report.md)。

新模型使用独立的未来标签，按全局时间顺序划分训练、校准、阈值选择、测试，剔除标签跨界样本，逐期限与品种报告 Brier、PR-AUC、AUC、预警率、精确率与召回率，并与固定概率及当前状态基线比较。按验证表现，在逻辑回归、简单状态条件概率和固定基础率之间选模。逻辑回归贡献可追溯到实际计算；简单状态模型另有训练分组及样本数说明。各候选参数、校准器、模式和阈值仅使用测试之前的数据。

开发中已查看过时间留出段，并修复了优化器未收敛和候选回退逻辑，因此本报告是**开发期时间外评估，不是外部盲测**。最终 1 日模型总体 Brier 为 0.172，相比固定概率基线误差低约 33.0%，相比更强的状态基线低约 4.0%；2–5 日整体未超过强状态基线，尚不能宣称长期预测能力已得到验证。工业硅 si 无测试期样本。完整数字与限制见报告。

- [新模型评估报告](data_out/predictive/report.md)
- [机器可读报告](data_out/predictive/report.json)
- [完整测试预测](data_out/predictive/test_predictions.csv)
- [项目审阅与升级说明](升级说明.md)
- [原始项目说明（保留）](README.legacy.md)

## 训练与导出

```sh
python scripts/train_predictive.py
python scripts/forecast_risk.py --symbol sc --as-of 2026-04-14T23:59:59 --days 1 --question '预测效果可靠吗？' --out risk_snapshot.json
python scripts/forecast_risk.py --symbol ag
python -m pytest tests -q
python scripts/smoke_dashboard.py
```

不指定 `--as-of` 时按当前上海时间检查数据时效。附带数据距当前日期已较久，当前预测将明确标记过期并暂停输出概率。历史回放也会拒绝模型训练/选点完成之前的时点，以免把训练内预测伪装成样本外结果。

模型保存在 `data_out/predictive/model.json`，仅含数值参数与元数据；新模块不加载旧 joblib/pickle。训练可以通过 `--data-dir`、`--output-dir` 和三个全局时间边界参数指定其他数据与切分。命令行预测支持 `--anchor-dir`、`--model-dir`。工作台通过 `RISK_ANCHOR_DIR` 指定特征根目录，刷新页面即可重新读取变动文件。

## 接入新数据

数据布局保持 `data_out/anchor/<symbol>/<YYYY-MM>/features.parquet`。每行必须包含本地上海时间 `_timestamp`（`YYYYMMDDHHMMSS`）和原始曲面特征。模型使用的字段清单位于 `vol_surface/predictive.py` 的 `RAW_FEATURES`；核心 ATM 隐波和凸性代理不可缺失，日盘需覆盖固定 15 个观测时点。重复时间戳、无法解析的时间戳会被拒绝。禁止把原标签 `y`、未来风险起点或未来停盘检测写成预测特征。

可复用原 `vol_surface/` 清洗和特征生成链更新数据；原始 `archive/` 未随本项目提供，重建源行情需要另接数据源。要转为持续运行，还需接入行情调度、官方交易日历、输入质量监测及上线后的效果漂移监测。本版本提供的是这些环节可接入的本地预测工作台。

项目没有账户持仓、组合敞口或损益样本。因此结果是曲面风险代理概率，不是价格涨跌判断、组合损失概率或账户 VaR/CVaR。

跨期限一致性检查：未来累计事件概率应随期限延长而不下降。独立期限模型估计若发生冲突，系统保留原始估计以便研究，同时暂停预测预警并提示，不进行事后概率修饰。报告中的指标评价原始模型；实际运行还会执行缺失、过期、分布与期限一致性检查。
