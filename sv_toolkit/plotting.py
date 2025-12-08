"""绘图函数，标题与图例使用英文。"""
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats

from .mixture import Q

sns.set_style("whitegrid")


def _ensure_dir(path: Path) -> None:
    """确保输出目录存在。"""
    path.mkdir(parents=True, exist_ok=True)


def plot_returns(df, output_dir: Path, title_suffix: str = "sample") -> Path:
    """绘制收益率随时间变化图，自动在停盘/周末断线，保存 PNG 文件。"""
    _ensure_dir(output_dir)
    time_index = df["index"].values
    r = df["r"].values

    # 重建规则时间网格，并在缺数据处插入 NaN，避免跨夜画斜线
    time_reg, series_reg = _regularize_time_series(time_index, {"r": r})
    r_reg = series_reg["r"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_reg, r_reg, color="tab:blue", linewidth=0.8, label="returns")
    ax.set_title(f"Returns over time ({title_suffix})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Return")
    ax.legend()
    fig.autofmt_xdate()

    output_path = output_dir / f"returns_{title_suffix}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _infer_time_freq(time_index: np.ndarray) -> Optional[pd.Timedelta]:
    """根据时间索引推断一个“典型频率”（取最常见的相邻差值）。"""
    if len(time_index) < 2:
        return None
    idx = pd.DatetimeIndex(time_index)
    diffs = idx[1:] - idx[:-1]
    if len(diffs) == 0:
        return None
    # 取众数（最常见的时间间隔），避免偶尔的缺失点干扰
    freq = pd.Series(diffs).mode().iloc[0]
    if freq <= pd.Timedelta(0):
        return None
    return freq


def _regularize_time_series(
    time_index: np.ndarray,
    series_dict: Dict[str, np.ndarray],
    freq: Optional[pd.Timedelta] = None,
) -> (np.ndarray, Dict[str, np.ndarray]):
    """
    给定时间索引和若干同长度序列，扩展到规则时间网格并用 NaN 填补缺失点。
    返回新的时间索引和同名的序列字典。
    """
    if len(time_index) == 0:
        return time_index, series_dict

    if freq is None:
        freq = _infer_time_freq(time_index)
    if freq is None:
        # 无法推断频率时，原样返回
        return time_index, series_dict

    idx = pd.DatetimeIndex(time_index)
    full_index = pd.date_range(idx.min(), idx.max(), freq=freq)

    out_dict: Dict[str, np.ndarray] = {}
    for name, values in series_dict.items():
        s = pd.Series(values, index=idx)
        s_full = s.reindex(full_index)
        out_dict[name] = s_full.to_numpy()

    return full_index.to_pydatetime(), out_dict


def plot_volatility(h_samples: np.ndarray, df, output_dir: Path, title_suffix: str = "sample") -> Path:
    """使用后验均值与置信区间绘制条件波动率轨迹，在停盘区间断线。"""
    _ensure_dir(output_dir)
    vol_mean = np.exp(h_samples.mean(axis=0) / 2)
    vol_low = np.exp(np.percentile(h_samples, 2.5, axis=0) / 2)
    vol_high = np.exp(np.percentile(h_samples, 97.5, axis=0) / 2)

    time_index = df["index"].values

    # 对波动率三个序列一起做规则化，保证同一时间网格
    time_reg, series_reg = _regularize_time_series(
        time_index,
        {
            "vol_mean": vol_mean,
            "vol_low": vol_low,
            "vol_high": vol_high,
        },
    )
    vol_mean_reg = series_reg["vol_mean"]
    vol_low_reg = series_reg["vol_low"]
    vol_high_reg = series_reg["vol_high"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_reg, vol_mean_reg, color="tab:orange", label="Posterior mean volatility")
    ax.fill_between(time_reg, vol_low_reg, vol_high_reg, color="tab:orange", alpha=0.2, label="95% CI")
    ax.set_title(f"Latent volatility ({title_suffix})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Volatility")
    ax.legend()
    fig.autofmt_xdate()

    output_path = output_dir / f"volatility_{title_suffix}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_return_histogram(r: np.ndarray, output_dir: Path, title_suffix: str = "sample") -> Path:
    """绘制收益直方图与正态分布对比，可展示厚尾特征。"""
    _ensure_dir(output_dir)
    mu_hat = np.mean(r)
    sigma_hat = np.std(r)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x_grid = np.linspace(r.min(), r.max(), 400)

    # 线性密度轴
    axes[0].hist(r, bins=80, density=True, alpha=0.45, label="Empirical returns")
    axes[0].plot(x_grid, stats.norm.pdf(x_grid, loc=mu_hat, scale=sigma_hat), label="Gaussian")
    axes[0].set_title(f"Return distribution ({title_suffix})")
    axes[0].set_xlabel("Return")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # 对数密度轴
    axes[1].hist(r, bins=80, density=True, alpha=0.45, label="Empirical returns")
    axes[1].plot(x_grid, stats.norm.pdf(x_grid, loc=mu_hat, scale=sigma_hat), label="Gaussian")
    axes[1].set_title(f"Return distribution (log scale, {title_suffix})")
    axes[1].set_xlabel("Return")
    axes[1].set_ylabel("Log density")
    axes[1].set_yscale("log")
    axes[1].legend()

    fig.tight_layout()
    output_path = output_dir / f"return_hist_{title_suffix}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_acf_returns(r: np.ndarray, output_dir: Path, title_suffix: str = "sample", lags: int = 50) -> Path:
    """绘制 r 与 |r| 的自相关函数，突出波动率聚集。"""
    _ensure_dir(output_dir)
    from statsmodels.graphics.tsaplots import plot_acf  # 延迟导入减少依赖冲突

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(r, lags=lags, zero=False, ax=axes[0])
    axes[0].set_title(f"ACF of returns ({title_suffix})")
    plot_acf(np.abs(r), lags=lags, zero=False, ax=axes[1])
    axes[1].set_title(f"ACF of |returns| ({title_suffix})")

    fig.tight_layout()
    output_path = output_dir / f"acf_{title_suffix}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_intraday_pattern(df, output_dir: Path, title_suffix: str = "sample") -> Path:
    """绘制日内平均绝对收益，展示开盘收盘的高波动特征。"""
    _ensure_dir(output_dir)
    df_tmp = df.copy()
    # 将时间戳转换为易于绘图的字符串，避免 matplotlib 处理 datetime.time 时出错
    df_tmp["time_of_day"] = df_tmp["index"].dt.strftime("%H:%M")
    df_tmp["abs_r"] = df_tmp["r"].abs()
    grouped = df_tmp.groupby("time_of_day")["abs_r"].mean().sort_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    # 使用字符串标签作为 x 轴刻度，防止 datetime.time 类型导致的转换错误
    ax.plot(grouped.index, grouped.values, color="tab:green")
    ax.set_title(f"Intraday pattern of |returns| ({title_suffix})")
    ax.set_xlabel("Time of day")
    ax.set_ylabel("Average |return|")
    fig.autofmt_xdate()

    fig.tight_layout()
    output_path = output_dir / f"intraday_{title_suffix}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_param_posterior(samples: Dict[str, np.ndarray], output_dir: Path, title_suffix: str = "sample") -> Path:
    """绘制 mu/alpha/beta/tau2 的 trace 与后验直方图。"""
    _ensure_dir(output_dir)
    params = ["mu", "alpha", "beta", "tau2"]
    fig, axes = plt.subplots(len(params), 2, figsize=(12, 8))

    for i, name in enumerate(params):
        chain = samples.get(name)
        ax_trace = axes[i, 0]
        ax_trace.plot(chain, linewidth=0.6)
        ax_trace.set_title(f"Trace of {name} ({title_suffix})")
        ax_trace.set_xlabel("Iteration")
        ax_trace.set_ylabel(name)

        ax_hist = axes[i, 1]
        ax_hist.hist(chain, bins=40, density=True, alpha=0.7)
        ax_hist.set_title(f"Posterior of {name} ({title_suffix})")
        ax_hist.set_xlabel(name)
        ax_hist.set_ylabel("Density")

    fig.tight_layout()
    output_path = output_dir / f"params_{title_suffix}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_vol_and_abs_returns(h_samples: np.ndarray, df, output_dir: Path, title_suffix: str = "sample") -> Path:
    """将潜在波动率与绝对收益同图展示，突出聚集特征，并在时间缺口处断线。"""
    _ensure_dir(output_dir)
    h_mean = h_samples.mean(axis=0)
    vol_mean = np.exp(h_mean / 2.0)
    abs_r = df["r"].abs().values
    # 缩放后的 |r|，避免量纲差太大
    scaled_abs_r = abs_r / np.median(abs_r)

    time_index = df["index"].values

    # 对两个序列一起扩展到规则时间网格
    time_reg, series_reg = _regularize_time_series(
        time_index,
        {
            "vol_mean": vol_mean,
            "scaled_abs_r": scaled_abs_r,
        },
    )
    vol_mean_reg = series_reg["vol_mean"]
    scaled_abs_r_reg = series_reg["scaled_abs_r"]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(time_reg, vol_mean_reg, label="Posterior mean vol", color="tab:orange")
    ax1.plot(time_reg, scaled_abs_r_reg, alpha=0.45, label="Scaled |returns|", color="tab:blue")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    ax1.set_title(f"Volatility vs |returns| ({title_suffix})")
    ax1.legend()
    fig.autofmt_xdate()

    fig.tight_layout()
    output_path = output_dir / f"vol_abs_{title_suffix}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_standardized_residuals(r: np.ndarray, samples: Dict[str, np.ndarray], output_dir: Path, title_suffix: str = "sample") -> Path:
    """检查标准化残差是否接近 N(0,1)。"""
    _ensure_dir(output_dir)
    h_mean = samples["h"].mean(axis=0)
    mu_mean = samples["mu"].mean()
    sigma_t = np.exp(h_mean / 2.0)
    z = (r - mu_mean) / sigma_t

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(z, bins=80, density=True, alpha=0.65)
    x_grid = np.linspace(z.min(), z.max(), 400)
    axes[0].plot(x_grid, stats.norm.pdf(x_grid), label="N(0,1)")
    axes[0].set_title(f"Std residuals ({title_suffix})")
    axes[0].set_xlabel("z")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    stats.probplot(z, dist="norm", plot=axes[1])
    axes[1].set_title(f"QQ-plot ({title_suffix})")

    fig.tight_layout()
    output_path = output_dir / f"std_resid_{title_suffix}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_mixture_usage(s_samples: np.ndarray, output_dir: Path, title_suffix: str = "sample") -> Path:
    """展示混合指标的经验频率与理论权重对比。"""
    _ensure_dir(output_dir)
    s_flat = s_samples.ravel()
    k = len(Q)
    counts = np.bincount(s_flat, minlength=k)
    freq_emp = counts / counts.sum()
    indices = np.arange(k)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(indices - 0.15, freq_emp, width=0.3, label="Empirical")
    ax.bar(indices + 0.15, Q, width=0.3, label="Theoretical")
    ax.set_xticks(indices)
    ax.set_xticklabels([f"k={i}" for i in range(k)])
    ax.set_ylabel("Weight")
    ax.set_title(f"Mixture usage ({title_suffix})")
    ax.legend()

    fig.tight_layout()
    output_path = output_dir / f"mixture_usage_{title_suffix}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
