"""批量处理多个合约并按时间戳输出结果的辅助函数。"""
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .data import get_contract_symbol_from_path, list_csv_files, load_single_file
from .mcmc import run_mcmc_sv
from .plotting import (
    plot_acf_returns,
    plot_intraday_pattern,
    plot_mixture_usage,
    plot_param_posterior,
    plot_return_histogram,
    plot_returns,
    plot_standardized_residuals,
    plot_vol_and_abs_returns,
    plot_volatility,
)


def make_timestamped_root(output_base: Path) -> Path:
    """创建带时间戳的根输出文件夹，如 outputs/SV_20251201_143000。"""
    output_base = Path(output_base)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = output_base / f"SV_{ts}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def extract_contract_tag(file_path: Path) -> str:
    """从文件名中提取合约英文代号，默认使用下划线前的部分并转为大写。"""
    return get_contract_symbol_from_path(file_path)


def save_param_summary(samples: Dict[str, np.ndarray], output_dir: Path, contract_tag: str, extra_info: Optional[Dict] = None) -> None:
    """保存参数后验统计到 CSV，并将运行信息写入 txt。"""
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mu_chain = samples["mu"]
    alpha_chain = samples["alpha"]
    beta_chain = samples["beta"]
    tau2_chain = samples["tau2"]

    rows = [
        ("mu", mu_chain),
        ("alpha", alpha_chain),
        ("beta", beta_chain),
        ("tau2", tau2_chain),
    ]

    gamma_chain = samples.get("gamma")
    if gamma_chain is not None and gamma_chain.size > 0:
        gamma_chain = np.atleast_2d(gamma_chain)
        for idx in range(gamma_chain.shape[1]):
            rows.append((f"gamma_{idx+1}", gamma_chain[:, idx]))

    summary = {
        "param": [name for name, _ in rows],
        "post_mean": [chain.mean() for _, chain in rows],
        "post_sd": [chain.std() for _, chain in rows],
        "q2.5": [np.quantile(chain, 0.025) for _, chain in rows],
        "q97.5": [np.quantile(chain, 0.975) for _, chain in rows],
    }

    df_summary = pd.DataFrame(summary)
    csv_path = output_dir / f"params_{contract_tag}.csv"
    df_summary.to_csv(csv_path, index=False)

    if extra_info is not None:
        txt_path = output_dir / f"run_info_{contract_tag}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for k, v in extra_info.items():
                f.write(f"{k}: {v}\n")


def run_batch_for_all_contracts(
    data_dir: Path,
    output_base: Path = Path("outputs"),
    max_files: int = 5,
    sample_every: int = 1,
    state_exog_col: Optional[str] = None,
    log_exog: bool = True,
    mcmc_kwargs: Optional[Dict] = None,
) -> Path:
    """批量处理多个 CSV，每个合约单独输出参数与图像，返回根输出路径。

    支持子采样（sample_every）和在状态方程中使用的外生变量（state_exog_col）。
    """
    data_dir = Path(data_dir)
    mcmc_kwargs = mcmc_kwargs or {}

    root_out = make_timestamped_root(output_base)
    csv_files = list_csv_files(data_dir, max_files=max_files)

    for file_path in csv_files[:max_files]:
        contract_tag = extract_contract_tag(file_path)
        print(f"Processing {file_path.name} (contract tag = {contract_tag})")

        contract_out_dir = root_out / contract_tag
        contract_out_dir.mkdir(parents=True, exist_ok=True)

        r, y_star, df, exog = load_single_file(
            file_path,
            sample_every=sample_every,
            state_exog_col=state_exog_col,
            log_exog=log_exog,
        )
        samples = run_mcmc_sv(r, y_star, exog_state=exog, **mcmc_kwargs)

        extra_info = {
            "file_name": file_path.name,
            "contract_tag": contract_tag,
            "T": len(r),
            "sample_every": sample_every,
            "state_exog_col": state_exog_col,
            **{k: v for k, v in mcmc_kwargs.items()},
        }
        save_param_summary(samples, contract_out_dir, contract_tag, extra_info=extra_info)

        title_suffix = f"{contract_tag}"
        plot_returns(df, output_dir=contract_out_dir, title_suffix=title_suffix)
        plot_return_histogram(r, output_dir=contract_out_dir, title_suffix=title_suffix)
        plot_acf_returns(r, output_dir=contract_out_dir, title_suffix=title_suffix)
        plot_intraday_pattern(df, output_dir=contract_out_dir, title_suffix=title_suffix)
        plot_param_posterior(samples, output_dir=contract_out_dir, title_suffix=title_suffix)
        plot_volatility(samples["h"], df, output_dir=contract_out_dir, title_suffix=title_suffix)
        plot_vol_and_abs_returns(samples["h"], df, output_dir=contract_out_dir, title_suffix=title_suffix)
        plot_standardized_residuals(r, samples, output_dir=contract_out_dir, title_suffix=title_suffix)
        if "s" in samples and len(samples["s"]) > 0:
            plot_mixture_usage(samples["s"], output_dir=contract_out_dir, title_suffix=title_suffix)

    return root_out


def run_mcmc_for_multiple_contracts(datasets: Dict[str, Dict], mcmc_kwargs: Optional[Dict] = None) -> Dict[str, Dict]:
    """在内存中的合约字典上循环运行 MCMC，返回同样按 symbol 分类的结果。"""
    mcmc_kwargs = mcmc_kwargs or {}
    results: Dict[str, Dict] = {}

    for symbol, contract_data in datasets.items():
        r = contract_data["r"]
        y_star = contract_data["y_star"]
        exog = contract_data.get("exog")

        print(f"Running MCMC for contract {symbol} with T={len(r)}")
        try:
            out = run_mcmc_sv(r, y_star, exog_state=exog, **mcmc_kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Skip {symbol} due to error: {exc}")
            continue

        results[symbol] = {"mcmc": out, "meta": {"symbol": symbol, "T": len(r)}}

    return results
