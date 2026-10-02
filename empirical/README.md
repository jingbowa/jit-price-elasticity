# Empirical Application: NYC Ride-Sharing Elasticities

**Paper:** *Scalable Just-in-Time Price Elasticity Estimation*
**Authors:** Jingbo Wang (Chinese University of Hong Kong) and Yufeng Huang (University of Rochester)

This folder reproduces the empirical results in the paper (Section 5 and Online Appendix E): Figure 2, Table 6, Table A5, and the associated in-text statistics. The full pipeline is included, from the public raw trip records to the paper exhibits.

## Data

**Analysis data — INCLUDED.** `outputs/step2_aggregated_markets/markets/step2_markets_2024-*.parquet` are the market-level analysis files used by all estimation scripts (7,462,961 cells; small CSV samples of the same schema in `markets/samples/`). A market is a pickup zone x dropoff zone x date x hour cell; columns include ride counts, average trip price, time/distance, passenger wait time, and the Hausman-style price instrument (`trip_price_iv`). Because these files are shipped, **all paper exhibits can be reproduced directly from this package (start at step 3) — downloading the raw data is not required**. `outputs/step2_aggregated_markets/zone_selection/step2_restriction_summary.csv` records the sample statistics reported in Section 5.1 (12,165,159 trips, 97 zones, mean price $20.33).

**Other included inputs:**
- `data/taxi_zone_lookup.csv`, `data/taxi_zones/` — TLC zone lookup and shapefile.
- `data/New_York_City_Regular_All_Formulations_Retail_Gasoline_Prices.csv` — NYC retail gasoline prices (EIA), used for auxiliary driver-cost variables.

**Raw data — not included (~5 GB); only needed to rebuild the analysis data from scratch with the step 1-2 code.** NYC Taxi & Limousine Commission High-Volume For-Hire Vehicle trip records, January and February 2024, from <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>. Download `fhvhv_tripdata_2024-01.parquet` and `fhvhv_tripdata_2024-02.parquet` and place them in `data/main_data/`.

## Setup

```
pip install -r requirements.txt
```

The estimation scripts use the GPU when CUDA is available and fall back to the CPU otherwise. The machine that produced the empirical part of the paper ("the reference machine" below and in the logs): Windows 11 Pro (10.0.26200), AMD Ryzen 9 9950X, 254 GB RAM, NVIDIA GeForce RTX 4090 (24 GB), Python 3.10, PyTorch 2.5.1 (CUDA 11.8 build). The Table 6 log of this machine (`../logs/table_6_reference_machine.log`) was made with PyTorch 2.6.0 (CUDA 12.4 build, driver 610.60), with identical output.

## Run

From this folder (`empirical/`). Step 1-2 (data construction) are only needed if rebuilding from raw data; otherwise start at step 3.

```
# Step 1: process raw inputs (requires data/main_data/, see Data above)
python code_latest/step1_data_preparation/1_prepare_gasoline.py
python code_latest/step1_data_preparation/2_prepare_zones.py
python code_latest/step1_data_preparation/3_prepare_zone_distances.py
python code_latest/step1_data_preparation/4_prepare_trips.py

# Step 2: sample restrictions and market-level aggregation (Section 5.1)
python code_latest/step2_analysis_preparation/1_identify_high_traffic_zones.py
python code_latest/step2_analysis_preparation/2_aggregate_to_markets.py

# Step 3: elasticity estimation and figures
python code_latest/step3_elasticity_estimation/2b_estimate_zone_elasticity_manhattan.py
python code_latest/step3_elasticity_estimation/3_estimate_temporal_elasticity.py
python code_latest/step3_elasticity_estimation/2a_estimate_zone_elasticity_all_nyc.py
python code_latest/utilities/visualize_manhattan_elasticity_map.py
python code_latest/utilities/visualize_all_nyc_elasticity_map.py
```

The scripts locate the package root from their own path, so they can also be invoked from other working directories. Note: `4_prepare_trips.py` writes ~1.3 GB per month to `outputs/step1_processed_data/trips/` (these intermediate files are not shipped; the step-2 outputs built from them are).

## Code-to-exhibit mapping

| Paper exhibit | Script(s) | Output file(s) |
|---|---|---|
| Figure 2 (Manhattan elasticity map) | `2b_estimate_zone_elasticity_manhattan.py` then `visualize_manhattan_elasticity_map.py` | `outputs/figures/step3_fig_manhattan_elasticity_map.pdf`; per-zone estimates in `outputs/step3_elasticity_estimates/step3_elasticity_zones_manhattan.csv` |
| Table 6 (elasticity by time window) | `3_estimate_temporal_elasticity.py` | `outputs/step3_elasticity_estimates/step3_elasticity_temporal_periods.csv` |
| Table A5 (zone-level distribution, 97/73 zones) | `2a_estimate_zone_elasticity_all_nyc.py` then `visualize_all_nyc_elasticity_map.py` | `outputs/figures/step3_fig_all_nyc_pre_winsorization_all_zones.csv`, `..._quality_zones.csv` |
| Winsorization statistics quoted in Section 5 | `visualize_manhattan_elasticity_map.py` | `outputs/figures/step3_fig_manhattan_winsorization_stats.csv` |
| Sample statistics in Section 5.1 | `1_identify_high_traffic_zones.py` | `outputs/step2_aggregated_markets/zone_selection/step2_restriction_summary.csv` |

All output files shipped in `outputs/` are pre-computed with this exact code and data; re-running the scripts overwrites them, with the published values on the reference machine and, for Table 6, slightly different values on other hardware (see Reproducibility).

## Reproducibility

Logs of the Table 6 runs described here are in `../logs/`. A fresh run overwrites the files in `outputs/step3_elasticity_estimates/`; the published values are also printed at the end of each log.

- **Figure 2, Table A5 and the in-text statistics are deterministic.** `2a` and `2b` use no random numbers. Run from the analysis data shipped in `outputs/step2_aggregated_markets/`, they reproduce the published per-zone estimates on both machines we tried, the reference machine and the Ubuntu server with an RTX 3090: every zone agrees with the published file to at least nine decimals.
- **Table 6 reproduces exactly on the reference machine and with slightly different values elsewhere.** The same code on the same data can give slightly different estimates on different GPUs. On the reference machine the script reproduces the published table to all digits (`table_6_reference_machine.log`); on the Ubuntu server with an RTX 3090 the estimates differ slightly (`other_hardware/table_6_rtx3090.log`).
- The subsampling size is `ZONE_TUNING_PARAMETER = 200` in `code_latest/step3_elasticity_estimation/config_quality_filters.py`; the estimator combines scales (m, 2m) for bias reduction.

Approximate runtimes on the reference machine: step 1 about 20 minutes (dominated by trip processing); step 2 about 20 minutes; 2b about 5 minutes; 3 about 15 minutes (100 bootstrap draws x 7 windows); 2a about 6 minutes; visualization scripts under 2 minutes. Each elasticity point estimate takes roughly 0.4-0.7 seconds on the GPU.
