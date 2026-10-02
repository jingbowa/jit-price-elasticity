# Replication Package

**Paper:** *Scalable Just-in-Time Price Elasticity Estimation*
**Authors:** Jingbo Wang (Chinese University of Hong Kong) and Yufeng Huang (University of Rochester)
**Contact:** jingbowang@cuhk.edu.hk; yufeng.huang@simon.rochester.edu

This package has two self-contained components:

- **`simulation/`** — all Monte Carlo results: Tables 1-5, Tables A1-A4, Figures 1 and A2. See `simulation/README.md` for the full code-to-exhibit mapping, setup, and pre-computed results.
- **`empirical/`** — the NYC ride-sharing application: Figure 2, Table 6, Table A5, and associated in-text statistics. See `empirical/README.md`.

## Data availability

- The simulation component generates all data (no external data used).
- The empirical component uses public NYC Taxi & Limousine Commission trip records (High-Volume For-Hire Vehicle, license HV0003, January-February 2024), freely available at <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>. The full construction pipeline (raw records to analysis data, following Section 5.1 of the paper) is included, and the processed market-level analysis data are shipped in `empirical/outputs/` so the estimation step can be run without downloading the raw data.

## Environments

Both components run on Python 3.10. Pinned dependencies are listed in each component (`simulation/environment_*.yml`, `empirical/requirements.txt`). A CUDA-capable GPU is recommended for the empirical component (tested on an NVIDIA RTX 4090) but not required.

## Reproducibility

The package reproduces the results of the paper. For two groups of results, a new run gives numbers that are close to the printed ones but not identical:

- **Simulations (Tables 1, 5 and A3; Figures 1 and A2).** Part of the simulation code runs in parallel, JIT-compiled loops (Numba). In these loops each thread is given its own random seed, which we cannot control from the script. The simulations therefore run without a fixed seed, and each run gives slightly different numbers. The hardware and the software environment can also lead to slightly different results. The package includes the estimates saved from the original runs in `simulation/simulation_results/`; from these files, the scripts reproduce the printed tables and figures exactly, without running the simulations again. See `simulation/README.md`.
- **Table 6.** The same code on the same data can give slightly different output on different GPUs. The table is reproduced exactly on the machine that produced it (NVIDIA RTX 4090) and with slightly different values on other hardware. See `empirical/README.md`.

## Logs

The folder `logs/` holds the terminal output of four runs. `table_A2_run.log` and `table_A4_run.log` are the runs that wrote the result files of Tables A2 and A4, on the Ubuntu server described in `simulation/README.md`. `table_6_reference_machine.log` and `other_hardware/table_6_rtx3090.log` are Table 6 on the reference machine described in `empirical/README.md` and on the Ubuntu server with an RTX 3090.

## Acknowledgement

We thank the Code and Data Editor of *Management Science* and the CASCAD team for their careful verification of this package, which made it clearer and more complete.
