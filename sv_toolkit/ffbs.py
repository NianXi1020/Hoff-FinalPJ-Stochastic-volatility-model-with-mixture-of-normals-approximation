"""卡尔曼滤波与 FFBS 采样函数。"""
from typing import Dict, Optional

import numpy as np

from .mixture import M, M0, V2


def _exog_term(exog: Optional[np.ndarray], gamma: Optional[np.ndarray], t: int) -> float:
    """安全计算第 t 期外生项的线性部分，若未提供则返回 0。"""
    if exog is None or gamma is None:
        return 0.0
    x_t = np.atleast_1d(exog[t])
    g_vec = np.atleast_1d(gamma)
    return float(np.dot(g_vec, x_t))


def kalman_filter(
    y_star: np.ndarray,
    s: np.ndarray,
    alpha: float,
    beta: float,
    tau2: float,
    gamma: Optional[np.ndarray],
    exog: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    针对给定的混合指标 s 和参数 (alpha, beta, tau2)，执行一轮卡尔曼滤波。
    返回字典包含 a, P, a_pred, P_pred，便于 FFBS 使用。
    """
    T = len(y_star)
    a = np.zeros(T)
    P = np.zeros(T)
    a_pred = np.zeros(T)
    P_pred = np.zeros(T)

    # 观测偏移量与方差
    mean_shift = np.take(M, s) - M0
    R = np.take(V2, s)

    # 初始状态：AR(1) 平稳分布
    a0 = alpha / (1 - beta)
    P0 = tau2 / (1 - beta ** 2)
    a_prev = a0
    P_prev = P0

    for t in range(T):
        # 预测步骤
        a_pred_t = alpha + beta * a_prev + _exog_term(exog, gamma, t)
        P_pred_t = beta ** 2 * P_prev + tau2

        # 更新步骤
        y_tilde = y_star[t] - mean_shift[t]
        K_t = P_pred_t / (P_pred_t + R[t])
        a_t = a_pred_t + K_t * (y_tilde - a_pred_t)
        P_t = (1 - K_t) * P_pred_t

        a_pred[t] = a_pred_t
        P_pred[t] = P_pred_t
        a[t] = a_t
        P[t] = P_t

        a_prev = a_t
        P_prev = P_t

    return {"a": a, "P": P, "a_pred": a_pred, "P_pred": P_pred}


def ffbs_sample_h(
    y_star: np.ndarray,
    s: np.ndarray,
    alpha: float,
    beta: float,
    tau2: float,
    rng: np.random.Generator,
    gamma: Optional[np.ndarray],
    exog: Optional[np.ndarray],
) -> np.ndarray:
    """
    使用 Carter-Kohn FFBS 算法一次性采样 h_{1:T}。
    """
    T = len(y_star)
    filt = kalman_filter(y_star, s, alpha, beta, tau2, gamma, exog)
    a = filt["a"]
    P = filt["P"]
    a_pred = filt["a_pred"]
    P_pred = filt["P_pred"]

    h = np.zeros(T)
    # 先抽取 h_T
    h[T - 1] = rng.normal(a[T - 1], np.sqrt(P[T - 1]))

    for t in range(T - 2, -1, -1):
        C_t = beta * P[t] / P_pred[t + 1]
        mu_t = a[t] + C_t * (h[t + 1] - a_pred[t + 1])
        V_t = P[t] - C_t ** 2 * P_pred[t + 1]
        V_t = max(V_t, 1e-10)  # 数值稳定性保护
        h[t] = rng.normal(mu_t, np.sqrt(V_t))

    return h
