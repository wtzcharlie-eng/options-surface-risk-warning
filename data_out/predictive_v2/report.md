# V2 联合风险率模型：开发期滚动验证

已完成日盘收盘后，联合预测未来1–5个观测交易日内首次未来超阈值的累计概率。ATM隐波或凸性违反代理任一日最大值超过预测日前最近60个完整交易日（至少20日）95%分位即为事件；阈值在预测原点冻结。当前正常与已异常状态分开建模。所有期限共享一条离散时间风险率曲线，累计概率为1−∏(1−风险率)。这是曲面异常代理，不是持仓亏损概率或VaR。观测交易日不是官方交易所日历。

**全部历史和 V1 结果此前已在开发中查看。本报告是开发期滚动验证，不能称为外部盲测。**

预先固定三个历史测试窗口；每一轮仅用其之前的数据训练、校准、按验证误差选候选和设置预警阈值。五个期限统一采用未来 5 个完整观测交易日标签齐全的预测原点；所有边界按实际第 5 日标签结束时间清除重叠。最终交互系统使用第三轮原样模型，未使用第三轮测试标签重新拟合。

候选限定为同状态分品种条件风险率基准、固定正则联合风险率、固定 126 观测日半衰期联合风险率。预测正常状态下新增异常和当前异常下未来再次超阈值分别拟合；后者不代表持续不中断或恢复时间。在过去验证段按每个品种/任务五期限平均 Brier 选择，并优先简单基准；基准只展示统计概率、不产生增量预测预警。

## 三轮总体结果

Brier 越低越好；skill=1−模型 Brier/独立校准的条件风险率基准 Brier。同日全部品种共同进行 10 个观测日期移动块重采样，500 次、固定种子 42。区间仅是条件于已拟合模型的探索性区间。

| 轮次 | 测试起点 | 期限 | 原点数 | 事件率 | Brier | 对条件基准 skill | skill 95%区间 | 运行可用率 | 运行误报数 | 运行召回率 |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 2024-12-19 | 1 | 763 | 15.2% | 0.0931 | 3.4% | [0.3%, 6.6%] | 100.0% | 68 | 12.9% |
| 1 | 2024-12-19 | 2 | 763 | 21.6% | 0.1245 | 6.1% | [-0.9%, 12.7%] | 100.0% | 125 | 23.6% |
| 1 | 2024-12-19 | 3 | 763 | 26.3% | 0.1528 | 6.8% | [-2.9%, 16.3%] | 100.0% | 169 | 32.8% |
| 1 | 2024-12-19 | 4 | 763 | 30.8% | 0.1721 | 8.1% | [-3.1%, 19.7%] | 100.0% | 210 | 42.1% |
| 1 | 2024-12-19 | 5 | 763 | 34.7% | 0.1860 | 9.8% | [-2.8%, 22.5%] | 100.0% | 237 | 47.9% |
| 2 | 2025-06-20 | 1 | 478 | 14.2% | 0.0907 | 4.2% | [0.9%, 7.4%] | 99.8% | 25 | 7.4% |
| 2 | 2025-06-20 | 2 | 478 | 20.5% | 0.1158 | 7.1% | [-0.5%, 13.5%] | 99.8% | 32 | 11.2% |
| 2 | 2025-06-20 | 3 | 478 | 24.9% | 0.1395 | 8.7% | [-2.7%, 18.5%] | 99.8% | 29 | 12.6% |
| 2 | 2025-06-20 | 4 | 478 | 29.3% | 0.1569 | 8.6% | [-3.0%, 19.5%] | 99.8% | 29 | 12.1% |
| 2 | 2025-06-20 | 5 | 478 | 33.5% | 0.1697 | 10.2% | [-2.7%, 22.0%] | 99.8% | 29 | 13.1% |
| 3 | 2025-10-17 | 1 | 528 | 33.9% | 0.1884 | -0.5% | [-5.3%, 3.3%] | 97.0% | 24 | 6.7% |
| 3 | 2025-10-17 | 2 | 528 | 46.2% | 0.2377 | -3.9% | [-12.6%, 2.6%] | 97.0% | 55 | 14.8% |
| 3 | 2025-10-17 | 3 | 528 | 54.4% | 0.2733 | -9.0% | [-21.2%, -0.2%] | 97.0% | 62 | 16.7% |
| 3 | 2025-10-17 | 4 | 528 | 60.4% | 0.2905 | -11.9% | [-26.8%, -1.2%] | 97.0% | 96 | 23.8% |
| 3 | 2025-10-17 | 5 | 528 | 64.8% | 0.2975 | -15.3% | [-31.3%, -3.7%] | 97.0% | 88 | 24.6% |

## 最近一轮：新增风险与当前异常分别评价

| 任务 | 期限 | 原点 | 事件率 | Brier | 对条件基准 skill | 原始预警 | 运行预警 | 运行误报 | 弃权 | 正标签原点片段召回 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| onset | 1 | 355 | 17.7% | 0.1555 | 1.4% | 32 | 32 | 23 | 8 | 14.3% |
| persistence | 1 | 173 | 67.1% | 0.2561 | -2.9% | 5 | 4 | 1 | 8 | 2.7% |
| onset | 2 | 355 | 29.6% | 0.2354 | 0.8% | 86 | 86 | 53 | 8 | 34.9% |
| persistence | 2 | 173 | 80.3% | 0.2424 | -14.9% | 6 | 5 | 2 | 8 | 2.1% |
| onset | 3 | 355 | 39.2% | 0.2907 | -2.0% | 105 | 105 | 60 | 8 | 38.5% |
| persistence | 3 | 173 | 85.5% | 0.2375 | -32.0% | 6 | 5 | 2 | 8 | 1.9% |
| onset | 4 | 355 | 46.8% | 0.3274 | -5.3% | 167 | 167 | 95 | 8 | 53.8% |
| persistence | 4 | 173 | 88.4% | 0.2147 | -39.1% | 6 | 5 | 1 | 8 | 3.6% |
| onset | 5 | 355 | 52.4% | 0.3476 | -8.7% | 167 | 167 | 87 | 8 | 53.8% |
| persistence | 5 | 173 | 90.2% | 0.1948 | -48.3% | 6 | 5 | 1 | 8 | 3.6% |

## 最近一轮分品种结果

| 品种 | 期限 | 模式 | 原点 | Brier | 对条件基准 skill | 可用率 | 运行精确率 | 运行召回率 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ag | 1 | hazard_baseline | 96 | 0.2089 | 0.0% | 85.4% | — | 0.0% |
| au | 1 | hazard_baseline,joint_logistic_decay | 89 | 0.1471 | 0.4% | 100.0% | 0.0% | 0.0% |
| cu | 1 | hazard_baseline,joint_logistic_decay | 61 | 0.2392 | 12.7% | 100.0% | 50.0% | 16.7% |
| lc | 1 | joint_logistic,joint_logistic_decay | 105 | 0.2248 | -13.6% | 99.0% | 50.0% | 10.7% |
| rb | 1 | hazard_baseline,joint_logistic | 68 | 0.1372 | -1.9% | 100.0% | 11.1% | 10.0% |
| sc | 1 | hazard_baseline,joint_logistic_decay | 109 | 0.1727 | 1.9% | 99.1% | 40.0% | 5.3% |
| ag | 2 | hazard_baseline | 96 | 0.2233 | 0.0% | 85.4% | — | 0.0% |
| au | 2 | hazard_baseline,joint_logistic_decay | 89 | 0.1933 | 1.6% | 100.0% | 35.7% | 28.6% |
| cu | 2 | hazard_baseline,joint_logistic_decay | 61 | 0.2192 | 22.7% | 100.0% | 55.6% | 21.7% |
| lc | 2 | joint_logistic,joint_logistic_decay | 105 | 0.3775 | -35.4% | 99.0% | 42.9% | 6.5% |
| rb | 2 | hazard_baseline,joint_logistic | 68 | 0.2090 | -2.8% | 100.0% | 27.8% | 27.8% |
| sc | 2 | hazard_baseline,joint_logistic_decay | 109 | 0.1803 | 8.4% | 99.1% | 40.0% | 16.3% |
| ag | 3 | hazard_baseline | 96 | 0.2251 | 0.0% | 85.4% | — | 0.0% |
| au | 3 | hazard_baseline,joint_logistic_decay | 89 | 0.2320 | 2.5% | 100.0% | 50.0% | 33.3% |
| cu | 3 | hazard_baseline,joint_logistic_decay | 61 | 0.2138 | 28.0% | 100.0% | 66.7% | 23.5% |
| lc | 3 | joint_logistic,joint_logistic_decay | 105 | 0.4971 | -57.7% | 99.0% | 42.9% | 5.0% |
| rb | 3 | hazard_baseline,joint_logistic | 68 | 0.2537 | -2.0% | 100.0% | 27.0% | 40.0% |
| sc | 3 | hazard_baseline,joint_logistic_decay | 109 | 0.1794 | 8.6% | 99.1% | 45.0% | 17.0% |
| ag | 4 | hazard_baseline | 96 | 0.2284 | 0.0% | 85.4% | — | 0.0% |
| au | 4 | hazard_baseline,joint_logistic_decay | 89 | 0.2654 | 1.9% | 100.0% | 60.7% | 34.7% |
| cu | 4 | hazard_baseline,joint_logistic_decay | 61 | 0.1866 | 33.8% | 100.0% | 77.3% | 31.5% |
| lc | 4 | joint_logistic,joint_logistic_decay | 105 | 0.5527 | -69.0% | 99.0% | 66.7% | 8.7% |
| rb | 4 | hazard_baseline,joint_logistic | 68 | 0.2787 | -5.2% | 100.0% | 34.8% | 53.3% |
| sc | 4 | hazard_baseline,joint_logistic_decay | 109 | 0.1785 | 9.8% | 99.1% | 29.9% | 35.1% |
| ag | 5 | hazard_baseline | 96 | 0.2212 | 0.0% | 85.4% | — | 0.0% |
| au | 5 | hazard_baseline,joint_logistic_decay | 89 | 0.2754 | 1.8% | 100.0% | 67.9% | 34.5% |
| cu | 5 | hazard_baseline,joint_logistic_decay | 61 | 0.1644 | 39.8% | 100.0% | 86.4% | 33.9% |
| lc | 5 | joint_logistic,joint_logistic_decay | 105 | 0.5903 | -81.1% | 99.0% | 66.7% | 8.0% |
| rb | 5 | hazard_baseline,joint_logistic | 68 | 0.2876 | -7.7% | 100.0% | 41.3% | 55.9% |
| sc | 5 | hazard_baseline,joint_logistic_decay | 109 | 0.1814 | 5.7% | 99.1% | 31.3% | 36.2% |

## 运行门禁与评价口径

历史回放在每个已完成日盘收盘的时刻作预测，数据年龄为零。缺失、历史不足、严重超出训练分布、已选择的原始常量训练特征发生变化、未训练任务和跨期限不一致按实际运行规则弃权。运行召回率保留全部原点为分母，弃权当作没有预警；原始概率 Brier 与门禁之后的告警统计分别报告，不能把弃权静默删掉后宣称整体改善。门禁后概率分数只在明确列出的可用子集上计算，不代表全原点准确度。

正标签原点片段指同品种/期限连续的正标签预测原点，不是真实事件去重。预测任务子集或缺日会切断片段；片段内首次告警位置不是实际事件提前天数。历史收盘回放无法评价实时行情延迟、断流或当前不完整交易日；实时服务仍须执行过期门禁。

| 轮次 | 跨期限冲突原点 | 门禁弃权原点 | 最晚选模标签结束 |
|---|---:|---:|---|
| 1 | 0 | 0 | 2024-12-18T15:00:00 |
| 2 | 0 | 1 | 2025-06-19T15:00:00 |
| 3 | 0 | 16 | 2025-10-16T15:00:00 |

## 与冻结 V1 对照

只比较第三轮且不早于 V1 模型可用时间的相同完整 5 日原点。前两轮不能使用当时尚未生成的 V1 模型。

| 期限 | 共同原点 | V1 Brier | V2 Brier | V1 运行预警 | V2 运行预警 | V1 告警准入率 | V2 告警准入率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 528 | 0.1788 | 0.1884 | 15 | 36 | 29.0% | 97.0% |
| 2 | 528 | 0.2622 | 0.2377 | 37 | 91 | 28.8% | 97.0% |
| 3 | 528 | 0.3122 | 0.2733 | 49 | 110 | 29.0% | 97.0% |
| 4 | 528 | 0.3225 | 0.2905 | 60 | 172 | 29.2% | 97.0% |
| 5 | 528 | 0.2985 | 0.2975 | 95 | 172 | 29.2% | 97.0% |

## 使用边界

- Market-surface proxy, not portfolio loss, VaR or CVaR.
- Existing history and prior V1 results were inspected before V2; all intervals and walk-forward results are development evidence.
- Thresholds frozen at each origin; normal-origin onset and currently-abnormal persistence are different tasks. Persistence target is any future exceedance, not uninterrupted persistence or time-to-recovery.
- Positive-label origin runs are proxy episodes, not deduplicated actual event dates. No event lead-time claim is made.
- Observed session calendar and >7-day gaps conservatively exclude missing data and some long holidays.
- Bootstrap preserves date-level cross-symbol dependence but conditions on fitted vintages, has no multiple-comparison adjustment, and cannot prove future gains.
- Candidate fallback and thresholds are selected only by preceding validation data; no test-driven fallback is applied. Validation F1 is a provisional product objective; business false-alarm and missed-event costs are not supplied.
- Final artifact is exactly the latest fold vintage; no refit uses its test labels.

完整逐轮/分任务/分品种原始与运行指标、探索性区间、校验和及固定设置见 report.json。预测 CSV/Parquet 保存弃权原因和每条标签结束日期。

模型 SHA-256：`7dfb4914da1a4081a73a4606df091038b82e4fac4bc4736021a97f7af6beb145`
