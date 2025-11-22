"""数据加载与预处理相关函数。所有注释均为中文，方便理解。"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# 默认的混合常数 c，用于构造 y_star
DEFAULT_C = 0.001


def list_csv_files(data_dir: Path, max_files: int = 5) -> List[Path]:
    """
    列出指定目录下的 CSV 文件路径（按名称排序），方便用户挑选需要处理的文件。
    参数中 max_files 控制打印多少个示例文件，避免一次性输出过多。
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    csv_files = sorted(data_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} csv files under {data_dir}")
    if not csv_files:
        return []

    preview = csv_files[:max_files]
    print("Preview of csv files:")
    for path in preview:
        print(f" - {path.name}")
    if len(csv_files) > max_files:
        print(f"... ({len(csv_files) - max_files} more files not shown)")
    return csv_files


def _filter_by_contract(df: pd.DataFrame, contract_code: Optional[str]) -> pd.DataFrame:
    """
    按合约代码过滤数据，若 contract_code 为 None 则直接返回原始数据。
    """
    if contract_code is None:
        return df
    mask = df["contract_code"] == contract_code
    return df.loc[mask].copy()


def _filter_by_time(df: pd.DataFrame, start_time: Optional[pd.Timestamp], end_time: Optional[pd.Timestamp]) -> pd.DataFrame:
    """
    按起止时间过滤数据，输入可以是 pandas.Timestamp 或字符串。
    """
    if start_time is not None:
        df = df[df["index"] >= pd.to_datetime(start_time)]
    if end_time is not None:
        df = df[df["index"] <= pd.to_datetime(end_time)]
    return df


def compute_returns(df: pd.DataFrame, c: float = DEFAULT_C) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    根据 DataFrame 计算对数收益率 r_t 与 y_star = log(r_t^2 + c)。
    返回 (r, y_star, 带有新列的 df)。
    """
    df = df.sort_values("index").copy()
    df["close"] = df["close"].astype(float)
    log_price = np.log(df["close"].values)
    r = 100.0 * np.diff(log_price)

    # 与收益率对齐，去掉第一行
    df = df.iloc[1:].copy()
    df["r"] = r
    y_star = np.log(r ** 2 + c)
    df["y_star"] = y_star
    return r, y_star, df


def load_single_file(
    file_path: Path,
    contract_code: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    读取单个 CSV，完成时间排序、合约筛选、时间窗口截取，并计算 r 与 y_star。
    max_rows 参数可限制读取的行数，便于快速测试。
    """
    file_path = Path(file_path)
    print(f"Loading file: {file_path}")
    df = pd.read_csv(file_path)

    # 确保时间列为 datetime
    df["index"] = pd.to_datetime(df["index"])
    df = _filter_by_contract(df, contract_code)
    df = df.sort_values("index")
    df = _filter_by_time(df, start_time, end_time)

    if max_rows is not None:
        df = df.head(max_rows)
        print(f"Restricted to first {max_rows} rows for quick demo")

    r, y_star, df_processed = compute_returns(df)
    print(f"Finished computing returns with length {len(r)}")
    return r, y_star, df_processed


def load_dataset(
    data_dir: Path,
    contract_code: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    max_files: int = 1,
    max_rows_per_file: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    读取指定目录下的若干 CSV 文件，并将它们串联后计算 r 与 y_star。
    默认只读取第一个文件，可以通过 max_files 控制数量，避免一次性载入过大数据。
    """
    csv_files = list_csv_files(data_dir, max_files=max_files)
    if not csv_files:
        raise FileNotFoundError(f"No csv files found under {data_dir}")

    all_r: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_df: List[pd.DataFrame] = []

    for file_path in csv_files[:max_files]:
        r, y_star, df_processed = load_single_file(
            file_path,
            contract_code=contract_code,
            start_time=start_time,
            end_time=end_time,
            max_rows=max_rows_per_file,
        )
        all_r.append(r)
        all_y.append(y_star)
        all_df.append(df_processed)

    concatenated_df = pd.concat(all_df, ignore_index=True)
    concatenated_r = np.concatenate(all_r)
    concatenated_y = np.concatenate(all_y)
    print(f"Concatenated {len(all_df)} files, total length {len(concatenated_r)}")
    return concatenated_r, concatenated_y, concatenated_df
