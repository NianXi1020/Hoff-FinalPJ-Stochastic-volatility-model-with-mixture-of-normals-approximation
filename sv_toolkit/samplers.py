"""参数与混合指标的采样函数。"""
from typing import Tuple

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


def sample_alpha_beta_tau2(h: np.ndarray, rng: Generator, prior_var: float = 100.0, a0: float = 2.0, b0: float = 2.0) -> Tuple[float, float, float]:
    """
    通过共轭更新采样状态方程参数 alpha, beta, tau2。
    """
    Y = h[1:]
    X = np.column_stack([np.ones(len(h) - 1), h[:-1]])

    XtX = X.T @ X
    XtY = X.T @ Y

    # 先验
    V0_inv = np.eye(2) / prior_var

    # 后验协方差与均值
    Sigma_n = np.linalg.inv(V0_inv + XtX)
    mu_n = Sigma_n @ XtY

    # 后验逆伽马参数
    resid = Y - X @ mu_n
    b_n = b0 + 0.5 * (resid @ resid)
    a_n = a0 + 0.5 * len(Y)

    tau2 = invgamma.rvs(a=a_n, scale=b_n, random_state=rng)

    cov_ab = Sigma_n * tau2
    alpha, beta = rng.multivariate_normal(mu_n, cov_ab)
    return float(alpha), float(beta), float(tau2)


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
