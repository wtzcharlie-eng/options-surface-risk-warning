# 交付包说明

本包是比赛原型与 V3 预测扩展的统一交付。先打开 `index.html` 阅读比赛对照，再按需要启动交互系统。运行门已复核：DRL 15/15；KG 11/11；这不等同比赛业务验收全部通过。

## 内容与版本

- `方法与结果.md`：分钟级固定配置的三种匹配口径、原结果解释、CVaR 缺口、V3 研究边界。
- `docs/比赛要求对照.md`：逐项对应原始赛题、源码/界面/报告证据与未完成证据。
- `docs/交付审阅与修订.md`：原交付问题、修复与追溯记录。
- `vol_surface/`、`kg/`、`drl/`、`dashboard/`、`scripts/`、`tests/`：算法、界面与验证。
- `data_out/anchor/`：7 品种、263 个月幕、138,794 截面，可离线重放。原始行情和大体积原档不在包内，不能声称从源行情完全重建。
- `data_out/drl/agent_seed*.npz`：15 个冻结 DQN；`rule_ml_portable.json` 为 Q1 数值树，保留源 joblib 及迁移校验记录。
- `data_out/predictive*`：冻结 V1/V2/V3 模型、报告与逐条预测；`docs/` 含相应实验设计与验收。
- `README.legacy.md`、`docs/legacy_delivery/` 和 `scripts/*_legacy.py`：原工程记录、原交付快照和旧构建逻辑，仅用于历史追溯。
- `requirements-predictive.lock.txt`、`.env.example`、`启动系统.command`：依赖、配置模板与本地启动。开发用 `.env`、Git 元数据、缓存及日志不收录；比赛专用包另按显式选项生成根目录 `.env`，密钥包含情况见 PACKAGE_PROFILE.json。

## 决赛平台统一入口 · 2026-10-24

面向附件5的最终部署材料以根目录 `部署说明.md` 为准。Linux 主入口为 `python scripts/serve_submission.py`，一个 SERVICE_PORT（默认 8000）提供全部五个页面，MODEL_* 参数从根目录 .env 读取。平台已指定的服务为 Qwen `qwen3.8-27b-vllm`，兼容协议不代表使用 OpenAI 境外服务。

通用源码包不包含真实 Key。`make_package.py --competition` 生成无密钥的合规 .env；只有显式添加 `--include-model-key` 时才从受限本机配置生成比赛私有包，后者仅上传比赛平台，不得公开分发。Dockerfile 始终排除 .env，不将密钥写入镜像。实际平台侧部署和访问验收须在上传后另行完成。

以下双端口启动方式作为独立本机研究入口保留，决赛部署使用上述统一入口。

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
