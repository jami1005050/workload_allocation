# greenload-forecast

[![CI](https://github.com/mohammadjaminur/greenload-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/mohammadjaminur/greenload-forecast/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

**Forecast the future. Schedule around it.** GRU-based demand / price / water / carbon forecasting driving a convex MPC scheduler for water- and carbon-aware datacenter workload placement.

Four GRU forecasters predict 24-hour-ahead workload, electricity price, water-usage-effectiveness, and carbon intensity across 10 datacenter locations. A receding-horizon Model Predictive Control (MPC) scheduler — a convex program solved with `cvxpy` — turns those forecasts into an hourly routing decision: how much delay-tolerant workload to send to each location, minimizing cost subject to fairness caps on water and carbon footprint. The whole pipeline is benchmarked against a naive persistence baseline and a perfect-foresight offline oracle, and a scheduled-sampling fix to the forecaster's training procedure closes **34% of the scheduler's optimality gap** versus the oracle.

Fully self-contained: every signal is procedurally generated (no real or private data anywhere), every result in this README is read straight out of [`results/metrics.json`](results/metrics.json), and the whole thing is tested and CI'd.

## Highlights

- **4 independent GRU seq2seq forecasters** — workload, price, water, carbon — each trained with scheduled sampling for stable closed-loop (autoregressive) rollout.
- **Convex MPC scheduler** (`cvxpy`) making a genuine receding-horizon routing decision every hour, benchmarked against a naive baseline and a perfect-foresight oracle.
- **Zero external data dependencies** — a procedural synthetic-data generator with realistic seasonality and per-location diversity, fully reproducible from a fixed seed.
- **Production-shaped repo**: installable package (`pyproject.toml`), `pytest` suite, `ruff` linting, GitHub Actions CI, a `Makefile`, and a results pipeline where nothing is hand-typed.

## Architecture

```
synthetic data generator (no real data, fixed seed)
        |
        v
chronological train/val/test split + z-score normalization
        |
        v
4x GRU seq2seq forecaster: workload / price / water / carbon
   trained with scheduled sampling
        |  closed-loop 24h autoregressive rollout
        v
receding-horizon MPC scheduler (cvxpy)
   minimize price + l1*max_loc(water) + l2*max_loc(carbon)
        |
        v
realized allocation --> compare vs. naive baseline & offline oracle
```

## Quickstart

```bash
git clone https://github.com/mohammadjaminur/greenload-forecast.git
cd greenload-forecast
pip install -e ".[dev]"

make data       # generate synthetic data                 (< 1s)
make train      # train all 8 models (4 signals x 2 modes) (~10-15 min, CPU)
make evaluate   # run the full evaluation pipeline          (~15-20s)
make test       # run the unit test suite                   (~5s)
```

`results/figures/` and `results/metrics.json` are already committed, so you can browse the results without running anything. Trained model weights aren't committed (~75MB); run `make train` to regenerate them before using `notebooks/demo.ipynb`.

## Repo structure

```
src/greenload/
├── config.py            # every hyperparameter, in one place
├── data/
│   ├── generate_synthetic_data.py
│   └── dataset.py        # chronological windowing + normalization
├── models/
│   ├── gru_forecaster.py # seq2seq GRU + scheduled sampling
│   └── naive_baseline.py # zero-order-hold forecaster
├── scheduler/
│   └── mpc.py             # receding-horizon MPC + offline oracle (cvxpy)
├── train.py                # trains all 8 models, saves checkpoints + loss curves
└── evaluate.py              # closed-loop rollout + MPC eval -> results/metrics.json
notebooks/demo.ipynb        # narrative walkthrough
tests/                       # pytest: data feasibility, no-leakage, solver sanity, schedule shape
.github/workflows/ci.yml    # lint + unit tests + 1-epoch smoke run
```

## Tech stack

Python 3.11+, PyTorch, cvxpy, NumPy/pandas/matplotlib, ruff, pytest, GitHub Actions.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Mohammad Jaminur Islam — [github.com/mohammadjaminur](https://github.com/mohammadjaminur)
