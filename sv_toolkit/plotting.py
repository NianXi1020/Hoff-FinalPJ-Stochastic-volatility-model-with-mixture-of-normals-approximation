"""绘图函数，标题与图例使用英文，并处理时间轴空档问题。"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .mixture import Q

sns.set_style("whitegrid")


def _ensure_dir(path: Path) -> None:
    """确保输出目录存在。"""
    path.mkdir(parents=True, exist_ok=True)


def _prepare_axis(
    time_index: pd.Series,
    use_step_index: bool = True,
    max_labels: int = 8,
):
    """构造横轴刻度与标签，默认使用交易步数避免时间空档导致的长直线。

    返回 (x_axis, tick_locs, tick_labels)，供后续绘图函数统一使用。
    """

    times = pd.to_datetime(time_index)
    n = len(times)
    if n == 0:
        return np.array([]), np.array([]), []

    x_axis = np.arange(n) if use_step_index else times.to_numpy()
    step = max(1, n // max_labels)
    tick_locs = np.arange(0, n, step)
    tick_labels = [times.iloc[i].strftime("%Y-%m-%d %H:%M") for i in tick_locs]
    return x_axis, tick_locs, tick_labels


def _iter_segments(
    times: pd.Series, use_step_index: bool = True, max_gap_minutes: int = 180
):
    """按时间空档切分连续区间。use_step_index=True 时只返回全量切片。"""

    if use_step_index:
        yield slice(None)
        return

    ts = pd.to_datetime(times)
    if len(ts) == 0:
        return

    gap = pd.Timedelta(minutes=max_gap_minutes)
    start = 0
    for i in range(len(ts) - 1):
        if ts.iloc[i + 1] - ts.iloc[i] > gap:
            yield slice(start, i + 1)
            start = i + 1
    yield slice(start, len(ts))


def plot_returns(
    df,
    output_dir: Path,
    title_suffix: str = "sample",
    use_step_index: bool = True,
    max_gap_minutes: int = 180,
    max_xticks: int = 8,
) -> Path:
    """绘制收益率随时间变化图，默认使用交易步数横轴避免休市空档产生突变。"""
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(10, 4))

    x_axis, tick_locs, tick_labels = _prepare_axis(
        df["index"], use_step_index=use_step_index, max_labels=max_xticks
    )
    for seg in _iter_segments(df["index"], use_step_index=use_step_index, max_gap_minutes=max_gap_minutes):
        ax.plot(x_axis[seg], df["r"].values[seg], color="tab:blue", linewidth=0.8, label="returns" if seg.start == 0 else None)

    ax.set_title(f"Returns over time ({title_suffix})")
    ax.set_xlabel("Trading step" if use_step_index else "Time")
    ax.set_ylabel("Return")
    ax.legend()
    ax.set_xticks(x_axis[tick_locs])
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    output_path = output_dir / f"returns_{title_suffix}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_volatility(
    h_samples: np.ndarray,
    df,
    output_dir: Path,
    title_suffix: str = "sample",
    use_step_index: bool = True,
    max_gap_minutes: int = 180,
    max_xticks: int = 8,
) -> Path:
    """使用后验均值与置信区间绘制条件波动率轨迹。"""
    _ensure_dir(output_dir)
    vol_mean = np.exp(h_samples.mean(axis=0) / 2)
    vol_low = np.exp(np.percentile(h_samples, 2.5, axis=0) / 2)
    vol_high = np.exp(np.percentile(h_samples, 97.5, axis=0) / 2)

    fig, ax = plt.subplots(figsize=(10, 4))
    x_axis, tick_locs, tick_labels = _prepare_axis(
        df["index"], use_step_index=use_step_index, max_labels=max_xticks
    )
    for seg in _iter_segments(df["index"], use_step_index=use_step_index, max_gap_minutes=max_gap_minutes):
        ax.plot(x_axis[seg], vol_mean[seg], color="tab:orange", label="Posterior mean volatility" if seg.start == 0 else None)
        ax.fill_between(
            x_axis[seg], vol_low[seg], vol_high[seg], color="tab:orange", alpha=0.2, label="95% CI" if seg.start == 0 else None
        )

    ax.set_title(f"Latent volatility ({title_suffix})")
    ax.set_xlabel("Trading step" if use_step_index else "Time")
    ax.set_ylabel("Volatility")
    ax.legend()
    ax.set_xticks(x_axis[tick_locs])
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

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


def plot_vol_and_abs_returns(
    h_samples: np.ndarray,
    df,
    output_dir: Path,
    title_suffix: str = "sample",
    use_step_index: bool = True,
    max_gap_minutes: int = 180,
    max_xticks: int = 8,
) -> Path:
    """将潜在波动率与绝对收益同图展示，突出聚集特征。"""
    _ensure_dir(output_dir)
    h_mean = h_samples.mean(axis=0)
    vol_mean = np.exp(h_mean / 2.0)
    abs_r = df["r"].abs().values
    scaled_abs_r = abs_r / np.median(abs_r)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    x_axis, tick_locs, tick_labels = _prepare_axis(
        df["index"], use_step_index=use_step_index, max_labels=max_xticks
    )
    for seg in _iter_segments(df["index"], use_step_index=use_step_index, max_gap_minutes=max_gap_minutes):
        ax1.plot(x_axis[seg], vol_mean[seg], label="Posterior mean vol" if seg.start == 0 else None, color="tab:orange")
        ax1.plot(x_axis[seg], scaled_abs_r[seg], alpha=0.45, label="Scaled |returns|" if seg.start == 0 else None, color="tab:blue")

    ax1.set_xlabel("Trading step" if use_step_index else "Time")
    ax1.set_ylabel("Value")
    ax1.set_title(f"Volatility vs |returns| ({title_suffix})")
    ax1.legend()
    ax1.set_xticks(x_axis[tick_locs])
    ax1.set_xticklabels(tick_labels, rotation=45, ha="right")

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
