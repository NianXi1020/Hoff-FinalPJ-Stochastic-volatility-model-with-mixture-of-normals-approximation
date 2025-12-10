"""用于带单一外生变量的 SV 模型辅助函数。"""
from typing import Optional, Tuple

import numpy as np

from .mcmc import run_mcmc_sv


def zscore_series(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """将序列做零均值单位方差标准化，避免尺度影响回归系数。"""
    arr = np.asarray(values, dtype=float)
    mu = np.nanmean(arr)
    sigma = np.nanstd(arr)
    if not np.isfinite(sigma) or sigma < eps:
        return np.zeros_like(arr)
    return (arr - mu) / sigma


def prepare_exog_series(
    df,
    column: str,
    log_transform: bool = True,
    zscore: bool = True,
) -> Optional[np.ndarray]:
    """从 DataFrame 中提取指定列，按需对数变换并标准化。"""
    if column not in df.columns:
        print(f"Warning: column {column} not found; skip exog")
        return None
    series = df[column].astype(float)
    if log_transform:
        series = np.log(series + 1.0)
    arr = series.to_numpy()
    if zscore:
        arr = zscore_series(arr)
    return arr


def run_sv_with_exog(
    r: np.ndarray,
    y_star: np.ndarray,
    exog: np.ndarray,
    **mcmc_kwargs,
) -> Tuple[dict, np.ndarray]:
    """包装调用 run_mcmc_sv，确保外生变量传入并返回标准化后的 exog。"""
    exog = np.asarray(exog, dtype=float)
    results = run_mcmc_sv(r, y_star, exog_state=exog, **mcmc_kwargs)
    return results, exog
