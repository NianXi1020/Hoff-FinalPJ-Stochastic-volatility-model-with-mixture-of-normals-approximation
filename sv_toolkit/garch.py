"""ARCH/GARCH 基准模型的极大似然拟合辅助函数。"""
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from arch import arch_model
except ImportError:  # pragma: no cover - 提示用户安装依赖
    arch_model = None  # type: ignore


def fit_garch_11(r: np.ndarray, dist: str = "normal") -> Tuple[Optional[object], Dict[str, float]]:
    """
    拟合 GARCH(1,1) 模型并返回 (模型结果对象, 参数摘要)。
    若未安装 arch 包，则返回 (None, {"error": ...}) 友好提示。
    """
    if arch_model is None:
        return None, {"error": "arch package not installed; pip install arch"}

    am = arch_model(r, vol="Garch", p=1, q=1, dist=dist, mean="Constant")
    res = am.fit(disp="off")
    params = res.params.to_dict()
    params = {k: float(v) for k, v in params.items()}
    return res, params


def fit_arch_q(r: np.ndarray, q: int = 5, dist: str = "normal") -> Tuple[Optional[object], Dict[str, float]]:
    """
    拟合 ARCH(q) 模型用于对比。q 默认 5，可按需调整。
    若 arch 未安装，会返回错误提示。
    """
    if arch_model is None:
        return None, {"error": "arch package not installed; pip install arch"}

    am = arch_model(r, vol="ARCH", p=q, dist=dist, mean="Constant")
    res = am.fit(disp="off")
    params = res.params.to_dict()
    params = {k: float(v) for k, v in params.items()}
    return res, params


__all__ = ["fit_garch_11", "fit_arch_q"]
