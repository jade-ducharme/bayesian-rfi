# bayesian-rfi

A Bayesian pipeline for detecting and characterizing radio-frequency interference (RFI) in Murchison Widefield Array (MWA) observations.

Based on [*Interference Meets Inference: Bayesian Time-Series Modeling of Radio-Frequency Interference*](https://arxiv.org/abs/2609.26904).

The pipeline uses [SSINS](https://github.com/mwilensky768/SSINS) to preprocess observations and [NumPyro](https://num.pyro.ai/en/stable/) to estimate the probability that each time sample contains RFI. Results are saved to HDF5 and can be written to [pyuvdata `UVFlag`](https://pyuvdata.readthedocs.io/en/latest/uvflag.html) files.

## Setup

1. Install the required Python packages: `pyuvdata`, `SSINS`, `numpyro`, `jax`, `arviz`, `h5py`, `astropy`, `pandas`, `matplotlib`, and `seaborn`.
2. Edit the paths in `config.py` for your system.

Run the commands below from the repository root.

## Pipeline

Run these five stages **in order**. Stages with multiple tasks should be run for every valid task index, typically using a SLURM job array.

| Stage | Script | Purpose |
| --- | --- | --- |
| 1 | `preprocess.py` | Reduce raw FITS data with SSINS and generate per-sub-band time series. |
| 2 | `fit.py` | Fit a Bayesian mixture model for each pointing, TV sub-band, and polarization. |
| 3 | `postprocess.py` | Check convergence, classify samples, and save results and diagnostics. |
| 4 | `summarize_season.py` | Generate season-level plots and statistics. |
| 5 | `propagate.py` | Write RFI probabilities back into `UVFlag` files. |

### Commands

```bash
# 1. Preprocess observations (one task per night/pointing)
python preprocess.py <job_index>

# 2. Fit the model (one task per pointing/sub-band/polarization)
python fit.py <job_index>

# 3. Process the fitted chains (same task grid as fit.py)
python postprocess.py <job_index>

# 4. Summarize the season (one task only)
python summarize_season.py 0

# 5. Update UVFlag files (tasks cover separate groups of files)
python propagate.py <job_index>
```

Replace `<job_index>` with an integer. To see how many tasks a stage needs, run:

```bash
python fit.py --n-jobs        # 50 tasks: indices 0–49
python preprocess.py --n-jobs # task count depends on the data
python propagate.py --n-jobs
```

To run `fit.py` as a SLURM array, for example:

```bash
#SBATCH --array=0-49
python fit.py "$SLURM_ARRAY_TASK_ID"
```

Each stage supports `--help`

### Main outputs

| Stage | Output |
| --- | --- |
| Preprocess | Per-sub-band `.npz` time series and initial `UVFlag` `.h5` files |
| Fit | Posterior chains in `<RUN_DIR>/chains/` (`.nc`), plus notes and selected-night lists |
| Postprocess | `<RUN_DIR>/processed/season_results.h5`, diagnostic plots, and soft-label samples |
| Summarize | Plots in `<RUN_DIR>/processed/seasonal_statistics/` |
| Propagate | **Updates the existing `UVFlag` files in place** |

The propagated metric is the posterior mean RFI probability. Its posterior standard deviation is stored in `weights_array` (an experimental convention, not a standard inverse-variance weight). Sub-bands excluded from fitting (due to e.g. excessive SSINS flags) are assigned a metric of `1.0` and a weight of `0.0`.

## Optional: half-season tests

For reproducibility checks, split the observations into two halves and fit them separately. Use `--half 0` or `--half 1`.

```bash
# Example: first chronological half
python fit.py <job_index> --run-dir out_chrono --split-method chrono --half 0
python postprocess.py <job_index> --run-dir out_chrono --half 0
python summarize_season.py 0 --run-dir out_chrono --half 0
```

Use `--split-method random --seed 17` instead for a reproducible random split. For random splits, the seed is appended to the run directory name. Split results are saved separately from full-season results.

## Repository files

- `config.py` — shared settings and job definitions
- `mixture_model.py` — model and dataset construction
- `preprocess.py`, `fit.py`, `postprocess.py`, `summarize_season.py`, `propagate.py` — pipeline stages
- `utils/` — shared command-line, I/O, modeling, and plotting helpers

## Contact

Questions: jade_ducharme@brown.edu

## License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for details.