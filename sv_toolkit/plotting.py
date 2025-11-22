"""绘图函数，标题与图例使用英文。"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_returns(df, output_dir: Path, title_suffix: str = "sample") -> Path:
    """
    绘制收益率随时间变化图，保存 PNG 文件。
    """
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["index"], df["r"], color="tab:blue", linewidth=0.8, label="returns")
    ax.set_title(f"Returns over time ({title_suffix})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Return")
    ax.legend()
    fig.autofmt_xdate()

    output_path = output_dir / f"returns_{title_suffix}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_volatility(h_samples: np.ndarray, df, output_dir: Path, title_suffix: str = "sample") -> Path:
    """
    使用后验均值与置信区间绘制条件波动率轨迹。
    """
    _ensure_dir(output_dir)
    vol_mean = np.exp(h_samples.mean(axis=0) / 2)
    vol_low = np.exp(np.percentile(h_samples, 2.5, axis=0) / 2)
    vol_high = np.exp(np.percentile(h_samples, 97.5, axis=0) / 2)

    fig, ax = plt.subplots(figsize=(10, 4))
    time_index = df["index"].values
    ax.plot(time_index, vol_mean, color="tab:orange", label="Posterior mean volatility")
    ax.fill_between(time_index, vol_low, vol_high, color="tab:orange", alpha=0.2, label="95% CI")
    ax.set_title(f"Latent volatility ({title_suffix})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Volatility")
    ax.legend()
    fig.autofmt_xdate()

    output_path = output_dir / f"volatility_{title_suffix}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
