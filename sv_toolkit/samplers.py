"""参数与混合指标的采样函数。"""
from typing import Optional, Tuple

import numpy as np
from numpy.random import Generator
from scipy.stats import invgamma, norm

from .mixture import M, M0, Q, V2


def sample_s(y_star: np.ndarray, h: np.ndarray, rng: Generator) -> np.ndarray:
    """
    根据当前的 h 与 y_star，对每个时间点的混合指标 s_t 进行抽样。
    """
    T = len(y_star)
    s = np.zeros(T, dtype=int)
    for t in range(T):
        z_t = y_star[t] - h[t]
        log_w = np.log(Q) + norm.logpdf(z_t, M - M0, np.sqrt(V2))
        log_w = log_w - log_w.max()
        w = np.exp(log_w)
        w = w / w.sum()
        s[t] = rng.choice(len(Q), p=w)
    return s


def sample_alpha_beta_tau2(
    h: np.ndarray,
    rng: Generator,
    exog: Optional[np.ndarray] = None,
    prior_var: float = 100.0,
    a0: float = 2.0,
    b0: float = 2.0,
) -> Tuple[float, float, float, Optional[np.ndarray]]:
    """
    通过共轭更新采样状态方程参数 alpha, beta, tau2，若 exog 不为 None，则一并采样
    gamma 系数向量。
    """
    Y = h[1:]

    if exog is not None:
        exog = np.asarray(exog)
        exog_slice = exog[1:]
        if exog_slice.ndim == 1:
            exog_slice = exog_slice.reshape(-1, 1)
        if exog_slice.shape[0] != len(Y):
            raise ValueError("Exogenous series length must match h")
        X = np.column_stack([np.ones(len(h) - 1), h[:-1], exog_slice])
    else:
        X = np.column_stack([np.ones(len(h) - 1), h[:-1]])

    XtX = X.T @ X
    XtY = X.T @ Y

    # 先验
    V0_inv = np.eye(X.shape[1]) / prior_var

    # 后验协方差与均值
    Sigma_n = np.linalg.inv(V0_inv + XtX)
    mu_n = Sigma_n @ XtY

    # 后验逆伽马参数
    resid = Y - X @ mu_n
    b_n = b0 + 0.5 * (resid @ resid)
    a_n = a0 + 0.5 * len(Y)

    tau2 = invgamma.rvs(a=a_n, scale=b_n, random_state=rng)

    cov_mat = Sigma_n * tau2
    draw = rng.multivariate_normal(mu_n, cov_mat)
    alpha = float(draw[0])
    beta = float(draw[1])
    gamma = draw[2:] if exog is not None else None
    return alpha, beta, float(tau2), gamma


def sample_mu(r: np.ndarray, h: np.ndarray, rng: Generator, prior_var: float = 100.0) -> float:
    """
    根据当前的波动路径 h，对均值参数 mu 进行共轭采样。
    """
    sigma2 = np.exp(h)
    precision = 1.0 / sigma2
    lambda_prior = 1.0 / prior_var

    lambda_post = lambda_prior + precision.sum()
    m_post = (precision @ r) / lambda_post
    mu = rng.normal(m_post, np.sqrt(1.0 / lambda_post))
    return float(mu)
