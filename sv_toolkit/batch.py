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

    summary = {
        "param": ["mu", "alpha", "beta", "tau2"],
        "post_mean": [mu_chain.mean(), alpha_chain.mean(), beta_chain.mean(), tau2_chain.mean()],
        "post_sd": [mu_chain.std(), alpha_chain.std(), beta_chain.std(), tau2_chain.std()],
        "q2.5": [np.quantile(mu_chain, 0.025), np.quantile(alpha_chain, 0.025), np.quantile(beta_chain, 0.025), np.quantile(tau2_chain, 0.025)],
        "q97.5": [np.quantile(mu_chain, 0.975), np.quantile(alpha_chain, 0.975), np.quantile(beta_chain, 0.975), np.quantile(tau2_chain, 0.975)],
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
    mcmc_kwargs: Optional[Dict] = None,
) -> Path:
    """批量处理多个 CSV，每个合约单独输出参数与图像，返回根输出路径。"""
    data_dir = Path(data_dir)
    mcmc_kwargs = mcmc_kwargs or {}

    root_out = make_timestamped_root(output_base)
    csv_files = list_csv_files(data_dir, max_files=max_files)

    for file_path in csv_files[:max_files]:
        contract_tag = extract_contract_tag(file_path)
        print(f"Processing {file_path.name} (contract tag = {contract_tag})")

        contract_out_dir = root_out / contract_tag
        contract_out_dir.mkdir(parents=True, exist_ok=True)

        r, y_star, df = load_single_file(file_path)
        samples = run_mcmc_sv(r, y_star, **mcmc_kwargs)

        extra_info = {
            "file_name": file_path.name,
            "contract_tag": contract_tag,
            "T": len(r),
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
