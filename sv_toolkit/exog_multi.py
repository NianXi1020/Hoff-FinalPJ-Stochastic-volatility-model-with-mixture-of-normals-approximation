"""处理多外生变量的 SV 模型辅助函数。"""
from typing import Iterable, List, Optional

import numpy as np

from .exog_sv import zscore_series


def prepare_multi_exog(
    df,
    columns: Iterable[str],
    log_transform: bool = True,
    zscore: bool = True,
    fill_value: float = 0.0,
) -> Optional[np.ndarray]:
    """从 DataFrame 提取多个列，按需对数变换与标准化，返回形状 (T, K) 的矩阵。"""
    arrays: List[np.ndarray] = []
    for col in columns:
        if col not in df.columns:
            print(f"Warning: column {col} not found; skip in exog matrix")
            continue
        series = df[col].astype(float)
        if log_transform:
            series = np.log(series + 1.0)
        arr = series.to_numpy()
        if zscore:
            arr = zscore_series(arr)
        arrays.append(arr)

    if not arrays:
        return None

    exog_mat = np.column_stack(arrays)
    if fill_value is not None:
        exog_mat = np.nan_to_num(exog_mat, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return exog_mat
