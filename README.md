# 期权曲面预测风控与交互系统

[![Predictive risk checks](https://github.com/wtzcharlie-eng/options-surface-risk-warning/actions/workflows/predictive-checks.yml/badge.svg)](https://github.com/wtzcharlie-eng/options-surface-risk-warning/actions/workflows/predictive-checks.yml)

在原项目上新增未来 **1–5 个交易日**的曲面风险预测，并通过可配置的 **OpenAI 兼容 LLM** 进行证据问答。默认使用 V3 稳健融合研究模型，可切换冻结的 V1/V2 对照。当前内置历史数据回放，尚未连接实时行情。

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

工作台可选择模型版本、品种、历史观察日和 1/2/3/4/5 交易日预测周期，查看风险概率、阈值、信号贡献、时间外评估，并与风险助手连续对话。切换观察上下文会开启新对话。主界面重点展示 1、3、5 日三个期限。

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

预测事件是：在未来 H 个完整观测交易日中，任一天的 ATM 隐波或曲面凸性违反代理最大值，超过**预测时点之前**最近 60 个完整交易日（至少 20 日）的 95% 分位阈值。阈值在观察时点冻结。V2/V3 分开建模：当前正常时预测新增异常；当前已异常时预测未来窗口内再次超阈值。后一任务不代表连续异常时长或恢复时间。

每个预测以已完成日盘收盘为基准；夜盘归下一观测到的日盘日期。当前没有官方交易所日历，缺失日期和长空档采取保守剔除。具体规则及覆盖率见 [V3 模型报告](data_out/predictive_v3/report.md)。

V3 沿用 V2 的离散条件风险率、1/5/20/60 日因果特征和三个训练候选，新增独立校准段的受约束斜率、向整体收缩的品种偏置，并让统计基准接受同样校准。特征风险率只以 25%/50% 权重与统计基准融合；按每任务过去验证段的配对误差改善减日期块标准误选择整条曲线，支持或改善不足则回退基准。分品种校准与每任务统一选择相结合，减少逐品种小样本选模和选阈值的波动。五个期限共用融合权重，累计概率仍随期限延长而不下降。

选择统计基准时只展示参考概率，关闭预测预警。V3 的权重作用于每日条件风险率，不能解释为累计概率或单项特征的可加贡献；V2 原逻辑模型贡献仍按其条件风险率对数几率解释。助手可核对校准偏差、运行覆盖和包含弃权的召回统计。

三轮扩展窗口验证均按全局时间隔离训练、校准、验证和测试，按实际第 5 日标签结束时间清除跨界样本；所有期限使用相同完整五日原点。报告同时给出原始概率误差、门禁后的误报与召回、包含弃权的分母以及日期块重采样区间。最终交互系统使用第三轮原样模型，不用其测试标签重训。

**这是开发期滚动验证，不是外部盲测**：全部历史与 V2 结果此前已查看。最近一轮 528 个共同原点中，V3 对 V2 的 1/2/3/4/5 日 Brier 分别降低约 **2.9%/5.0%/6.6%/7.1%/7.7%**。三个窗口的 1 日均有改善，但前两个窗口的部分 2–5 日期限变差，不能宣称全面提升。

对 V3 自身更强的同任务校准基准，最近一轮 1/2 日技巧分数约 +3.3%/+1.7%，3–5 日仍为负。正常状态下新增异常任务的 1 日改善仅约 +0.2%，区间跨零，5 日概率仍明显偏低。最近一轮 1 日全原点召回从 V2 的 6.7% 提高至 59.8%，但误报也从 24 增至 70 次；这同时受选模覆盖和阈值变化影响，不是独立的准确度证据。工业硅 si 无测试期样本。完整数字与限制见报告。

- [V3 滚动评估报告](data_out/predictive_v3/report.md)
- [V3 机器可读报告与可靠性表](data_out/predictive_v3/report.json)
- [V3 完整测试预测](data_out/predictive_v3/test_predictions.csv)
- [V3 固定实验设计](docs/V3实验设计.md)
- [V2 滚动评估报告](data_out/predictive_v2/report.md)
- [V2 机器可读报告](data_out/predictive_v2/report.json)
- [V2 完整测试预测](data_out/predictive_v2/test_predictions.csv)
- [V2 固定实验设计](docs/V2实验设计.md)
- [冻结 V1 评估报告](data_out/predictive/report.md)
- [项目审阅与升级说明](升级说明.md)
- [原始项目说明（保留）](README.legacy.md)

## 训练与导出

```sh
python scripts/train_predictive_v3.py
python scripts/forecast_risk.py --symbol sc --as-of 2026-04-14T23:59:59 --days 1 --question '预测效果可靠吗？' --out risk_snapshot.json
python scripts/forecast_risk.py --version v1 --symbol sc --as-of 2026-04-14T23:59:59
python scripts/forecast_risk.py --version v2 --symbol sc --as-of 2026-04-14T23:59:59
python scripts/forecast_risk.py --symbol ag
python -m pytest tests/test_predictive*.py tests/test_risk_assistant.py tests/test_v2_assistant_evidence.py tests/test_v3_assistant_evidence.py -q
python scripts/smoke_dashboard.py
python scripts/audit_predictive_v3.py
```

不指定 `--as-of` 时按当前上海时间检查数据时效。附带数据距当前日期已较久，当前预测将明确标记过期并暂停输出概率。历史回放也会拒绝模型训练/选点完成之前的时点，以免把训练内预测伪装成样本外结果。

默认模型保存在 `data_out/predictive_v3/model.json`，V1/V2 分别保留在 `data_out/predictive/`、`data_out/predictive_v2/`，仅含数值参数与元数据；新模块不加载旧 joblib/pickle。V3 训练支持 `--data-dir`、`--output-dir`、`--symbols`、`--v2-predictions`，三轮时间设置固定。对新数据或品种子集研究，须提供同源、同切分的 V2 逐条预测；对照样本或标签不完全匹配时会中止，不会静默缩小样本。原 V1/V2 训练入口仍可用，但应另指定输出目录，保留冻结对照。

命令行预测支持 `--version auto|v1|v2|v3`、`--anchor-dir`、`--model-dir`。工作台通过 `RISK_ANCHOR_DIR` 指定特征根目录；数据变动会触发缓存更新。自动校验额外独立复算已发布分数、运行统计和最后一轮持久化系数预测，核对模型、源文件、实验设计与冻结对照的校验和。

## 接入新数据

数据布局保持 `data_out/anchor/<symbol>/<YYYY-MM>/features.parquet`。每行必须包含本地上海时间 `_timestamp`（`YYYYMMDDHHMMSS`）和原始曲面特征。模型使用的字段清单位于 `vol_surface/predictive.py` 的 `RAW_FEATURES`；核心 ATM 隐波和凸性代理不可缺失，日盘需覆盖固定 15 个观测时点。重复时间戳、无法解析的时间戳会被拒绝。禁止把原标签 `y`、未来风险起点或未来停盘检测写成预测特征。

可复用原 `vol_surface/` 清洗和特征生成链更新数据；原始 `archive/` 未随本项目提供，重建源行情需要另接数据源。要转为持续运行，还需接入行情调度、官方交易日历、输入质量监测及上线后的效果漂移监测。本版本提供的是这些环节可接入的本地预测工作台。

项目没有账户持仓、组合敞口或损益样本。因此结果是曲面风险代理概率，不是价格涨跌判断、组合损失概率或账户 VaR/CVaR。

V2/V3 在全部三轮测试中跨期限冲突为 0；V1 仍保留原始独立估计，并在冲突时暂停预警。实际运行还执行缺失、过期和分布检查：动态原始特征严重超出训练范围或训练常量特征出现变化时，整条曲线暂停。报告区分包含统计基准的概率可用率与特征模型信号覆盖率；可用率提高不等于预测更准确。
