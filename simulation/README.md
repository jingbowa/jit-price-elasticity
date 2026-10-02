# Replication Package

**Paper:** *Scalable Just-in-Time Price Elasticity Estimation*
**Authors:** Jingbo Wang (Chinese University of Hong Kong) and Yufeng Huang (University of Rochester)

This package contains the simulation code and pre-computed results for all Monte Carlo tables and figures in the paper.

---

## Code-to-Manuscript Mapping

> **Note:** Script names do not follow the paper's numbering. The tables below give the script for each table and figure.

### Main Tables

| Manuscript | Description | Simulation Script | Formatting | Result File |
|---|---|---|---|---|
| **Table 1, Panel A** | Baseline RC logit models | `table_1_models.py` | `all_tables.py → Table_1.tex` | `table_1_models.json` |
| **Table 1, Panel B** | Supply-side equilibrium | `table_5_supply.py` | `all_tables.py → Table_1.tex` | `table_5_supply.json` |
| **Table 1, Panel C** | Complementarity & variety | `table_4_extra_models.py` | `all_tables.py → Table_1.tex` | `table_4_extra_models.json` |
| **Table 2** | Literature-calibrated DGPs: elasticities and implied prices | `table_6_literature.py` | `all_tables.py → Table_2.tex` | `table_6_literature.json` |
| **Table 3** | Implied optimal prices and computation time (baseline DGPs) | `table_6_baseline_pricing.py` | `all_tables.py → Table_3.tex` | `table_6_baseline_pricing.json` |
| **Table 4, Panel A** | Computation time (J=4) | `table_2_scalability.py` | `all_tables.py → Table_4.tex` | `table_2_scalability.json` |
| **Table 4, Panel B** | Computation time (J=20) | `table_2_scalability_20.py` | `all_tables.py → Table_4.tex` | `table_2_scalability_20.json` |
| **Table 5** | MSE-minimizing tuning (own elasticity) | `table_3_tuning.py` + `table_3_tuning_extended.py` | `all_tables.py → Table_5.tex` | `table_3_tuning_full.json`, `table_3_tuning_full_extended.json` |
| **Table 6** | Empirical: NYC ride-sharing elasticities | see `../empirical/` (`3_estimate_temporal_elasticity.py`) | — | `step3_elasticity_temporal_periods.csv` |

### Appendix Tables

| Manuscript | Description | Simulation Script | Formatting | Result File |
|---|---|---|---|---|
| **Table A1** | Parameter values for literature-calibrated DGPs | `calibrate_literature.py` | — (values entered in LaTeX) | `calibrated_params.json` |
| **Table A2** | Robustness (sample size & products) | `table_A2_robustness.py` | `all_tables.py → Table_A2.tex` | `table_A2_robustness.json` |
| **Table A3, Panel A** | Tuning: cross-price elasticity | `table_3_tuning.py` + `table_3_tuning_extended.py` | `all_tables.py → Table_A3.tex` | `table_3_tuning_full.json`, `table_3_tuning_full_extended.json` |
| **Table A3, Panel B** | Tuning: reduced-form polynomial | `table_3_tuning_reduced_form.py` | `all_tables.py → Table_A3.tex` | `table_3_tuning_reduced_form.json` |
| **Table A4** | Extra primitive parameter values | `table_A4_extra_primitives.py` | `all_tables.py → Table_A4.tex` | `table_A4_extra_primitives.json` |
| **Table A5** | Empirical: NYC zone-level elasticity distribution | see `../empirical/` (`2a_estimate_zone_elasticity_all_nyc.py`) | — | `step3_fig_all_nyc_pre_winsorization_*.csv` |
| *(not in the paper)* | PyBLP computation time by market count | `table_appendix_pyblp.py` | — | `table_appendix_pyblp.json` |
| *(not in the paper)* | BNN vs polynomial timing | `table_polynomial_time.py` | — (writes directly) | — |

### Figures

| Manuscript | Description | Computation | Drawing | Result File |
|---|---|---|---|---|
| **Figure 1(a)** | Elasticity profiles: correlated BLP | `figure_BLP.py` | `all_figures.py → fig_BLP.png` | `fig_elas_curve_BLP.json` |
| **Figure 1(b)** | Elasticity profiles: complementarity | `figure_complements.py` | `all_figures.py → fig_complements.png` | `fig_elas_curve_complements.json` |
| **Figure 1(c)** | Elasticity profiles: variety & quantity | `figure_variety.py` | `all_figures.py → fig_variety.png` | `fig_elas_curve_variety.json` |
| **Figure 2** | Empirical: Manhattan elasticity map | see `../empirical/` (`2b_...py` + `visualize_manhattan_elasticity_map.py`) | — | `step3_fig_manhattan_elasticity_map.pdf` |
| **Figure A2** | Optimal tuning visualization | same runs as Table 5 | `all_figures.py → optimal_tuning_own.png` | `tables/table_3_opt_tuning_own.csv` (written by `all_tables.py`) |

All scripts above are relative to `simulation_codes/`, `figures/`, or `tables/` as appropriate.

---

## Setup

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- NVIDIA GPU with CUDA 11.8 support (optional; CPU-only execution is supported)

### Environment

Create and activate the conda environment. Two specifications are provided: `environment_ubuntu.yml` pins the exact versions of the Ubuntu server that generated the results; `environment_windows.yml` is a portable specification for other platforms.

```bash
conda env create -f environment_ubuntu.yml   # or environment_windows.yml
conda activate Price_elasticity
```

Key dependencies: Python 3.10, PyTorch 2.4, NumPy 1.26, Numba 0.60, SciPy 1.13, Pandas 2.2. The complementarity and variety/quantity DGPs (Table 1 Panel C and Figures 1b/1c) additionally require `NumbaMinpack` (`pip install NumbaMinpack`), which builds from source; we recommend installing it on Linux (it requires a C/Fortran toolchain on Windows).

---

## Directory Structure

```
simulation/
├── README.md                   # This file
├── environment_ubuntu.yml      # Conda environment (exact versions, results server)
├── environment_windows.yml     # Conda environment (portable specification)
│
├── bnn_modules/                # Core estimator implementations
│   ├── bnn_torch.py            #   PyTorch implementation (GPU/CPU)
│   ├── bnn_numba.py            #   Numba-accelerated implementation (CPU)
│   └── bnn_module_supply.py    #   Supply-side variant
│
├── DGP_models/                 # Data-generating processes
│   ├── model_module_torch.py   #   Logit, independent RC logit, correlated RC logit
│   ├── model_module_extra.py   #   Complementarity and variety/quantity models
│   ├── model_module_blp.py     #   Parameterized BLP discrete choice DGP
│   ├── model_module_rf.py      #   Reduced-form polynomial models
│   └── model_module_rc.py      #   Correlated RC logit for Tables A2 and A4
│
├── simulation_codes/           # Scripts that run Monte Carlo simulations
│   ├── table_1_models.py       #   → Table 1 Panel A
│   ├── table_2_scalability.py  #   → Table 4 Panel A
│   ├── table_2_scalability_20.py  # → Table 4 Panel B
│   ├── table_3_tuning.py       #   → Table 5, Table A3 Panel A
│   ├── table_3_tuning_extended.py #  → Table 5, Table A3 Panel A
│   ├── table_3_tuning_reduced_form.py # → Table A3 Panel B
│   ├── table_4_extra_models.py #   → Table 1 Panel C
│   ├── table_5_supply.py       #   → Table 1 Panel B
│   ├── table_6_literature.py   #   → Table 2
│   ├── table_6_baseline_pricing.py # → Table 3
│   ├── calibrate_literature.py #   → Table A1 (one-time calibration)
│   ├── table_A2_robustness.py  #   → Table A2
│   ├── table_A4_extra_primitives.py # → Table A4
│   ├── _mc_common.py           #   Monte Carlo loop shared by the two scripts above
│   ├── table_appendix_pyblp.py
│   └── table_polynomial_time.py
│
├── figures/                    # Figure scripts and the figure files
│   ├── figure_BLP.py           #   computes the curves of Figure 1(a)
│   ├── figure_complements.py   #   computes the curves of Figure 1(b)
│   ├── figure_variety.py       #   computes the curves of Figure 1(c)
│   ├── all_figures.py          #   draws Figure 1(a)-(c) and Figure A2 from the result files
│   ├── fig_BLP.png, fig_complements.png, fig_variety.png   # Figure 1(a)-(c)
│   └── optimal_tuning_own.png  #   Figure A2
│
├── tables/                     # Table builder and its output
│   ├── all_tables.py           #   writes Table_1.tex ... Table_A4.tex, numbered as in the paper
│   ├── Table_1.tex ... Table_5.tex, Table_A2.tex, Table_A3.tex, Table_A4.tex
│   └── table_3_opt_tuning_own.csv   # input of Figure A2
│
└── simulation_results/         # Pre-computed simulation outputs (JSON)
    ├── table_1_models.json
    ├── table_2_scalability.json
    ├── table_2_scalability_20.json
    ├── table_3_tuning_full.json
    ├── table_3_tuning_full_extended.json
    ├── table_3_tuning_reduced_form.json
    ├── table_4_extra_models.json
    ├── table_5_supply.json
    ├── table_6_baseline_pricing.json
    ├── table_6_literature.json
    ├── calibrated_params.json
    ├── table_appendix_pyblp.json
    ├── fig_elas_curve_BLP.json
    ├── fig_elas_curve_complements.json
    ├── fig_elas_curve_variety.json
    ├── table_A2_robustness.json
    └── table_A4_extra_primitives.json
```

---

## How to Reproduce

### Building the tables and figures from the result files

The `simulation_results/` directory contains the simulation outputs behind the tables and figures. Two commands build every simulation table and figure from them, in seconds:

```bash
cd simulation
python tables/all_tables.py     # writes tables/Table_1.tex ... Table_5.tex, Table_A2.tex, Table_A3.tex, Table_A4.tex
python figures/all_figures.py   # writes figures/fig_BLP.png, fig_complements.png, fig_variety.png, optimal_tuning_own.png
```

The tables are numbered and laid out as in the paper. `all_figures.py` is the script that drew the published figures; on the machine that drew them (Windows 11, matplotlib 3.9.2, seaborn 0.13.2) it writes the same four files byte for byte. On other systems the figures are the same up to fonts and rendering details. Run `all_tables.py` first: it writes `tables/table_3_opt_tuning_own.csv`, which Figure A2 reads.

### Re-running the simulations

For Tables 1, 5 and A3 and Figures 1 and A2, the files in `simulation_results/` hold the estimates saved from the original runs, the ones reported in the paper; `all_tables.py` and `all_figures.py` reproduce the printed tables and figures from them exactly, without running the simulations again. A new run of these simulations gives slightly different numbers. Part of the simulation code runs in parallel, JIT-compiled loops (Numba). In these loops each thread is given its own random seed, which we cannot control from the script. The simulations therefore run without a fixed seed, and each run gives slightly different numbers. The hardware and the software environment can also lead to slightly different results.

Tables 2, 3 and A1 fix their seeds. Table 4 and the time columns of Table 3 report timings and depend on the hardware.

### Tables A2 and A4

`table_A2_robustness.py` and `table_A4_extra_primitives.py` produce Tables A2 and A4. Their output is in `simulation_results/` and their run logs are in `../logs/`. The printed Tables A2 and A4 came from earlier runs, so the tables built here come from new runs and are close to the printed ones but not identical.

### Full replication (re-running simulations from scratch)

To regenerate the JSON result files themselves, run the simulation scripts in `simulation_codes/`. Each script is self-contained:

```bash
python simulation_codes/table_1_models.py
python simulation_codes/table_2_scalability.py
python simulation_codes/table_2_scalability_20.py
python simulation_codes/table_3_tuning.py
python simulation_codes/table_3_tuning_extended.py
python simulation_codes/table_3_tuning_reduced_form.py
python simulation_codes/table_4_extra_models.py
python simulation_codes/table_5_supply.py
python simulation_codes/table_6_baseline_pricing.py
python simulation_codes/table_6_literature.py
python simulation_codes/table_A2_robustness.py
python simulation_codes/table_A4_extra_primitives.py
python simulation_codes/table_appendix_pyblp.py
python simulation_codes/table_polynomial_time.py
```

Results are saved as JSON files in `simulation_results/`, overwriting the pre-computed outputs. Then run `python tables/all_tables.py` to regenerate the LaTeX tables.

Note: `calibrate_literature.py`, `table_6_literature.py`, `table_6_baseline_pricing.py`, `table_A2_robustness.py` and `table_A4_extra_primitives.py` skip configurations already present in their output JSON files (a resume feature for long runs). To force full regeneration, delete or rename the corresponding file in `simulation_results/` first.

### Re-computing the figure curves

```bash
python figures/figure_BLP.py          # rewrites simulation_results/fig_elas_curve_BLP.json (an hour or more)
python figures/figure_complements.py  # same for Figure 1(b); needs NumbaMinpack
python figures/figure_variety.py      # same for Figure 1(c); needs NumbaMinpack
python figures/all_figures.py         # draws the four figures from the result files
```

Figures are saved as `.png` files in `figures/`: `fig_BLP.png`, `fig_complements.png`, `fig_variety.png` (the three panels of Figure 1) and `optimal_tuning_own.png` (Figure A2).

---

## Runtime Notes

**Please read before re-running.** Building the tables and figures from the shipped result files takes seconds. Re-running all simulations from scratch requires several days of total compute; the heaviest scripts are flagged below. Times refer to the results server (dual Xeon, RTX 3090) unless stated otherwise.

- **Table 1, Panel A** (`table_1_models.py`): 50 Monte Carlo iterations per model; minutes on GPU.
- **Table 1, Panel B** (`table_5_supply.py`): Solves equilibrium prices for all 20,000 markets per iteration via `joblib` (shipped `n_jobs=80`); adjust `n_jobs` to your core count. Expect roughly 4 minutes per Monte Carlo iteration on a 32-thread workstation, i.e., about 10 hours for the full 3 models × 50 iterations; scales with cores.
- **Table 1, Panel C** (`table_4_extra_models.py`): The variety/quantity model is the slowest single script (~12 hours on the server); requires `NumbaMinpack`.
- **Table 2** (`table_6_literature.py`) and **Table 3** (`table_6_baseline_pricing.py`): Table 2 takes about 15 minutes on GPU; Table 3 runs PyBLP and takes a few hours (requires `pyblp`). Both resume from configurations already present in their output JSONs (see the note under "Full replication"). Table 2 uses pre-calibrated parameters stored in `calibrated_params.json`; to re-run calibration from scratch: `python simulation_codes/calibrate_literature.py`.
- **Table 4** (`table_2_scalability.py` / `table_2_scalability_20.py`): Benchmarks Numba CPU, PyTorch CPU, and PyTorch GPU; requires a GPU for full replication. Reported times are hardware-specific — expect qualitative, not numerical, agreement on different machines.
- **Table 5 / Table A3** (`table_3_tuning.py`, `table_3_tuning_extended.py`, `table_3_tuning_reduced_form.py`): 1,000 Monte Carlo iterations per tuning value and product count — the most compute-intensive tables (many hours to days in total). GPU strongly recommended.
- **Figures 1(a)–1(c)** (`figures/figure_*.py`): Each traces elasticity curves over a price grid with many Monte Carlo draws; allow an hour or more per figure. Figures 1(b)/1(c) require `NumbaMinpack`.
- **Tables A2 and A4** (`table_A2_robustness.py`, `table_A4_extra_primitives.py`): on one RTX 3090, Table A4 takes about a minute, Table A2 Panels A and B about 11 minutes together and Panel C (T = 5,120,000, m = 480) about 2.5 hours.
- **PyBLP appendix** (`table_appendix_pyblp.py`): Requires the `pyblp` package (install via `pip install pyblp`).

---

## Hardware

The simulation results in the paper were generated on a workstation running Ubuntu 22.04.3 LTS, equipped with dual Intel Xeon Gold 6348 CPUs and a single NVIDIA GeForce RTX 3090 GPU ("the Ubuntu server"). The figure files of Figures 1 and A2 were drawn from these results on a Windows machine, the one that produced the empirical part of the paper (see `../empirical/README.md`).
