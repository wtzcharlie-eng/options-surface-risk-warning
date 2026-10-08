"""vol_surface — 期权波动率曲面数据处理与风险特征工程算法库。

模块组成
--------
io_loader      : 品族/月度 parquet 读取与按品种族聚合
cleaning       : 曲面数据清洗（休市、不活跃、IV 边界、卡死 Greeks、邻居跳变）
interpolation  : SVI 参数化拟合 + RBF 兜底，构建规则网格 IV(dte, moneyness)
arbitrage      : 无套利条件校验（日历 / 蝶式 / put-call parity）
features       : 可量化风险特征工程
alert_rules   : 规则引擎，输出可解释触发原因
alert_model    : 轻量级 IsolationForest 异常检测
alert_engine   : 规则 + ML 融合，输出 0-3 级预警与触发原因清单
backtest       : 对照极端事件标注表的回测验证
"""

from .io_loader import load_symbol_family, list_underlyings  # noqa: F401
from .cleaning import clean_slice, CleaningReport  # noqa: F401

__all__ = ["load_symbol_family", "list_underlyings", "clean_slice", "CleaningReport"]
