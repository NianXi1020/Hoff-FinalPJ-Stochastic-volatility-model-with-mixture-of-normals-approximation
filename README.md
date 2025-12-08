# Stochastic Volatility Notebook Demo

This repository contains a Jupyter Notebook plus helper modules for fitting a
stochastic volatility (SV) model with the Kim–Shephard–Chib normal-mixture
approximation. The notebook is designed to run on large collections of
1-minute futures CSVs, while giving you options to process only a few files or
a subset of rows so that experiments stay lightweight.

## Data placement

- Place all CSVs under a directory named `2005年__20250905` that sits alongside
  the repo root. When you start Jupyter inside `notebooks/`, the default path
  used by the demo notebook is `../2005年__20250905`.
- Filenames are expected to start with the English contract code followed by an
  underscore (e.g., `AG_主力合约_1m数据.csv`, `BU_主力合约_1m数据.csv`). The code
  before the first underscore becomes the contract tag (e.g., `AG`, `BU`) used
  in titles, legends, and output names to avoid font issues with Chinese text.

## Key files

- `notebooks/sv_mixture_demo.ipynb`: guided workflow with Chinese comments,
  English print statements/plot labels, and progress prints. It loads multiple
  contracts in one run, loops over them with separate RNG seeds, and writes
  each contract's outputs into its own subfolder under the timestamped root.
  It demonstrates per-contract loading, optional subsampling, MCMC,
  diagnostics, and exporting PNGs/CSVs.
- `sv_toolkit/`: Python package with data loading, FFBS/MCMC samplers, plotting
  utilities, batch runners, and simple ARCH/GARCH benchmarks.
- `outputs/`: auto-created folder for timestamped runs (e.g.,
  `outputs/SV_20250101_120000/AG/`). Each contract gets its own subfolder with
  figures and parameter summaries. You can still point plots to `figures/` if
  you prefer a fixed location.

## Quickstart (single contract via notebook)

1. Launch Jupyter and open `notebooks/sv_mixture_demo.ipynb`.
2. Verify the first cell has `data_dir = Path('../2005年__20250905')` (default).
3. Adjust the sampling controls in the first code cell:
   - `max_files`: limit how many CSVs to read (start with `1`).
   - `max_rows_per_file`: cut each CSV to a manageable size (e.g., `5000`).
   - `sample_every`: subsample in time (e.g., `5` to keep every 5th minute).
   - `state_exog_col`: optional state covariate column name (e.g., `volume`).
4. Run the notebook. It prints progress every few iterations and saves PNGs plus
   parameter summaries into `outputs/SV_YYYYMMDD_HHMMSS/<contract_tag>/`.

## Running multiple contracts

- Use `sv_toolkit.data.load_contracts_in_dir` to return a dictionary keyed by
  contract tag. Loop over it directly (as shown in the demo notebook), or call
  `sv_toolkit.batch.run_mcmc_for_multiple_contracts` to automate the per-
  contract loop in memory.
- For disk-backed runs, `sv_toolkit.batch.run_batch_for_all_contracts` creates a
  timestamped root under `outputs/`, builds one subfolder per contract, and
  writes:
  - PNG diagnostics (returns, ACF, intraday pattern, volatility paths,
    standardized residuals, mixture usage, etc.).
  - `params_<tag>.csv` summarizing posterior mean/SD/quantiles for
    `mu, alpha, beta, tau2` (and `gamma` if you include a state covariate).
  - `run_info_<tag>.txt` capturing data size, sampler settings, and file name.
  - PNGs use trading-step x-axes by default to avoid long straight lines across
    weekend/night gaps; labels still show representative timestamps.

## Subsampling and state covariates

- `sample_every` (int) down-samples the time axis before computing returns,
  shrinking runtime without changing the model structure.
- `state_exog_col` (str or None) pulls a numeric column (e.g., `volume`) from
  each CSV and attaches it to the state equation. The sampler automatically
  samples a `gamma` coefficient alongside `alpha, beta, tau2` when this is
  provided.

## GARCH/ARCH benchmarks

- Optional: install the `arch` package to use the thin wrappers in
  `sv_toolkit.garch`:
  - `fit_garch_11(r, mean='constant')`
  - `fit_arch_q(r, q=5, mean='constant')`
- These functions fit on the same subsampled returns you pass into SV, and they
  return fitted parameters plus conditional variance series for quick
  comparisons.

## Tips for large datasets

- Start with one CSV and a small row cap (e.g., `max_rows_per_file=3000`).
- Increase `sample_every` to 5–10 for long spans to cut runtime.
- Use the timestamped outputs to keep experiments organized; each run is
  isolated in its own folder.

## Module overview

- `sv_toolkit.data`: per-contract loading with optional subsampling and state
  covariates; returns a dict keyed by contract tag.
- `sv_toolkit.mixture`, `sv_toolkit.ffbs`, `sv_toolkit.samplers`: mixture
  constants, Kalman/FFBS routines, and Gibbs steps (including `gamma` when
  covariates are present).
- `sv_toolkit.mcmc`: high-level `run_mcmc_sv` driver with progress prints and
  storage of `mu, alpha, beta, tau2, gamma, h, s` chains.
- `sv_toolkit.plotting`: PNG-ready diagnostics (histogram vs Gaussian, ACF,
  intraday pattern, trace/density, volatility overlays, standardized residuals,
  mixture usage).
- `sv_toolkit.batch`: timestamped batch runners and helpers to save summaries
  plus figures per contract.
- `sv_toolkit.garch`: optional ARCH/GARCH benchmarks via the `arch` package.
