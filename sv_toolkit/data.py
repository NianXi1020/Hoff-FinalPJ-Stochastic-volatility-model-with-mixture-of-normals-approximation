"""数据加载与预处理相关函数。所有注释均为中文，方便理解。"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def get_contract_symbol_from_path(file_path: Path) -> str:
    """
    根据文件名解析合约代号（下划线前的英文部分），并转换为大写。
    例如 "AG_主力合约_1m数据.csv" -> "AG"，"A_主力合约_1m数据.csv" -> "A"。
    若解析失败，则返回 "UNKNOWN"。
    """
    stem = Path(file_path).stem
    first_part = stem.split("_")[0].upper()
    return first_part or "UNKNOWN"


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


def load_contracts_in_dir(
    data_dir: Path,
    contract_code: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    max_files: int = 1,
    max_rows_per_file: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    按“合约”为粒度读取目录中的 CSV，返回以合约 symbol 为键的字典。

    返回示例：
    {
        "AG": {"symbol": "AG", "file_path": Path(...), "r": np.ndarray, "y_star": np.ndarray, "df": DataFrame},
        "A": {...},
    }
    """
    csv_files = list_csv_files(data_dir, max_files=max_files)
    if not csv_files:
        raise FileNotFoundError(f"No csv files found under {data_dir}")

    datasets: Dict[str, Dict[str, Any]] = {}
    for file_path in csv_files[:max_files]:
        symbol = get_contract_symbol_from_path(file_path)
        r, y_star, df_processed = load_single_file(
            file_path,
            contract_code=contract_code,
            start_time=start_time,
            end_time=end_time,
            max_rows=max_rows_per_file,
        )
        datasets[symbol] = {
            "symbol": symbol,
            "file_path": Path(file_path),
            "r": r,
            "y_star": y_star,
            "df": df_processed,
        }

    print(f"Loaded {len(datasets)} contract(s): {list(datasets.keys())}")
    return datasets


def load_dataset(
    data_dir: Path,
    contract_code: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    max_files: int = 1,
    max_rows_per_file: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    保留旧接口但不再串联不同合约，而是返回按 symbol 划分的字典。
    建议新代码直接调用 load_contracts_in_dir。返回结构与 load_contracts_in_dir 相同。
    """
    return load_contracts_in_dir(
        data_dir=data_dir,
        contract_code=contract_code,
        start_time=start_time,
        end_time=end_time,
        max_files=max_files,
        max_rows_per_file=max_rows_per_file,
    )
