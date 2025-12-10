"""主 MCMC 循环，包含进度打印。"""
from typing import Dict, List, Optional

import numpy as np
from numpy.random import default_rng

from .ffbs import ffbs_sample_h
from .mixture import Q
from .samplers import sample_alpha_beta_tau2, sample_mu, sample_s


def run_mcmc_sv(
    r: np.ndarray,
    y_star: np.ndarray,
    n_iter: int = 200,
    burn_in: int = 50,
    thin: int = 1,
    rng_seed: int = 2025,
    progress_every: int = 20,
    store_s: bool = True,
    exog_state: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """运行 SV 模型的 MCMC，并返回参数与波动路径样本。"""
    rng = default_rng(rng_seed)
    T = len(r)

    if exog_state is not None and len(exog_state) != T:
        raise ValueError("exog_state length must match r and y_star")

    mu = 0.0
    alpha = 0.0
    beta = 0.9
    tau2 = 0.1
    gamma = None
    h = np.full(T, np.log(np.var(r) + 1e-6))
    s = rng.integers(low=0, high=len(Q), size=T)

    saved_mu: List[float] = []
    saved_alpha: List[float] = []
    saved_beta: List[float] = []
    saved_gamma: List[np.ndarray] = []
    saved_tau2: List[float] = []
    saved_h: List[np.ndarray] = []
    saved_s: List[np.ndarray] = []

    for it in range(1, n_iter + 1):
        h = ffbs_sample_h(y_star, s, alpha, beta, tau2, rng, gamma, exog_state)
        s = sample_s(y_star, h, rng)
        alpha, beta, tau2, gamma = sample_alpha_beta_tau2(h, rng, exog=exog_state)
        mu = sample_mu(r, h, rng)

        if it % progress_every == 0:
            print(
                f"Iter {it}: mu={mu:.4f}, alpha={alpha:.4f}, beta={beta:.4f}, tau2={tau2:.4f}"
            )

        if it > burn_in and (it - burn_in) % thin == 0:
            saved_mu.append(mu)
            saved_alpha.append(alpha)
            saved_beta.append(beta)
            saved_tau2.append(tau2)
            saved_h.append(h.copy())
            if gamma is not None:
                saved_gamma.append(np.atleast_1d(gamma).copy())
            if store_s:
                saved_s.append(s.copy())

    results = {
        "mu": np.array(saved_mu),
        "alpha": np.array(saved_alpha),
        "beta": np.array(saved_beta),
        "tau2": np.array(saved_tau2),
        "h": np.array(saved_h),
    }
    if saved_gamma:
        results["gamma"] = np.array(saved_gamma)
    if store_s:
        results["s"] = np.array(saved_s)
    return results
