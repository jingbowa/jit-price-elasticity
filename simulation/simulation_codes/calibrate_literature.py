"""
One-time calibration script for literature-based DGP configurations.

For each target paper, fixes alpha (and RC params) from the literature,
then calibrates gamma_0 and delta_kxi so the DGP matches:
  - target mean |own-price elasticity|
  - realistic share levels and elasticity spread

Run once; results are saved to simulation_results/calibrated_params.json.
The table_6_literature.py script loads these stored parameters.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import json

from DGP_models.model_module_blp import (
    BLP_corr_parameterized, BLP_corr_elasticities
)

NUM_PRODUCTS = 4
CALIBRATION_SEED = 99999
CALIBRATION_MARKETS = 5000
CALIBRATION_DRAWS = 500


def _compute_moments(params, seed=CALIBRATION_SEED):
    """Compute elasticities and shares for a given parameter vector."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    elas = BLP_corr_elasticities(
        num_markets=CALIBRATION_MARKETS,
        num_products=NUM_PRODUCTS,
        num_draws=CALIBRATION_DRAWS,
        params=params,
    )
    np.random.seed(seed)
    torch.manual_seed(seed)
    s_jt, p_jt, _, _, _, _, _ = BLP_corr_parameterized(
        CALIBRATION_MARKETS, NUM_PRODUCTS, 250, params
    )
    mean_shares = s_jt.mean(dim=0).numpy()
    mean_prices = p_jt.mean(dim=0).numpy()
    outside_share = 1.0 - mean_shares.sum()
    return elas, mean_shares, mean_prices, outside_share


def _calibrate_gamma0(params, target_elas, lo=-30.0, hi=40.0, steps=40):
    """Binary search for gamma_0 that matches target mean |elasticity|."""
    best_gamma, best_diff = None, float('inf')
    p = params.copy()
    for _ in range(steps):
        mid = (lo + hi) / 2
        p['gamma_0'] = mid
        np.random.seed(CALIBRATION_SEED)
        torch.manual_seed(CALIBRATION_SEED)
        elas = BLP_corr_elasticities(
            num_markets=CALIBRATION_MARKETS,
            num_products=NUM_PRODUCTS,
            num_draws=CALIBRATION_DRAWS,
            params=p,
        )
        mean_abs = np.abs(elas).mean()
        diff = abs(mean_abs - target_elas)
        if diff < best_diff:
            best_diff = diff
            best_gamma = mid
        if mean_abs > target_elas:
            lo = mid
        else:
            hi = mid
    return best_gamma


def _report(name, params, target_elas):
    """Print calibration diagnostics."""
    elas, shares, prices, outside = _compute_moments(params)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Target |elas|      : {target_elas}")
    print(f"  Achieved |elas|    : {np.abs(elas).mean():.4f}")
    print(f"  Per-product elas   : {np.round(elas, 4)}")
    print(f"  Per-product |elas| : {np.round(np.abs(elas), 4)}")
    print(f"  Mean shares        : {np.round(shares, 4)}")
    print(f"  Outside share      : {outside:.4f}")
    print(f"  Mean prices        : {np.round(prices, 2)}")
    print(f"  gamma_0            : {params['gamma_0']}")
    print(f"  delta_kxi          : {params.get('delta_kxi')}")
    return elas, shares, prices, outside


# =========================================================================
#  PER-PAPER CALIBRATION
# =========================================================================

def calibrate_wollmann():
    """
    Wollmann (2018) -- Trucks.
    Logit, alpha = -0.44 per $1000, prices ~$15K-$60K, target |elas| ~ 8.5.
    These parameters are already validated from prior runs.
    """
    params = {
        'alpha_0': -0.44, 'alpha_1': 0, 'alpha_2': 0,
        'gamma_0': 10.76, 'gamma_1': 0, 'gamma_2': 0,
        'delta_kxi': [-14.0, -4.0, 4.0, 14.0],
        'p_base': [19.0, 23.0, 28.0, 35.0],
        'p_z_scale': 3.0, 'p_kxi_scale': 0.5,
    }
    target_elas = 8.5
    elas, shares, prices, outside = _report("Logit_Wollmann", params, target_elas)

    return {
        "name": "Logit_Wollmann",
        "source": "Wollmann (2018) Table 2",
        "type": "logit",
        "alpha_0": params['alpha_0'],
        "target_elas": target_elas,
        "target_elas_range": [6.3, 10.9],
        "params": params,
        "achieved_mean_abs_elas": float(np.abs(elas).mean()),
        "achieved_elas": elas.tolist(),
        "achieved_shares": shares.tolist(),
        "achieved_prices": prices.tolist(),
        "achieved_outside_share": float(outside),
    }


def calibrate_nevo():
    """
    Nevo (2001) / Conlon & Gortmaker (2020) Table 7 -- Cereal.
    RC Logit. Prices in $/serving (~$0.10-$0.30).

    From Nevo (2001) Table VI / C&G (2020) Table 7 "Best Practices":
      - Price mean:  -27.198 (Nevo) / -27.489 (C&G)
      - Price SD:    2.453 (Nevo) / 2.910 (C&G)
      - Price x Income: 315.894 (Nevo) / 15.957 (C&G) -- poorly identified
      - Constant mean: 3.592 (Nevo)
      - Constant SD: 0.330 (Nevo) / 0.196 (C&G)
      - Constant x Income: 5.482 (Nevo) / 6.253 (C&G)
      - Mean own-price elasticity: -3.685 (C&G "Best Practices")

    Target: |elas| ≈ 3.7
    """
    target_elas = 3.7

    params = {
        'alpha_0': -27.5, 'alpha_1': 0.8, 'alpha_2': 2.9,
        'gamma_0': 0.0,
        'gamma_1': 0.5,
        'gamma_2': 0.2,
        'delta_kxi': [6.0, 2.0, -2.0, -6.0],
        'p_base': 0.15, 'p_z_scale': 0.05, 'p_kxi_scale': 0.01,
    }

    print("\n  Calibrating gamma_0 for RC_Nevo...")
    params['gamma_0'] = _calibrate_gamma0(params, target_elas, lo=-5.0, hi=15.0)
    print(f"  Found gamma_0 = {params['gamma_0']:.4f}")

    elas, shares, prices, outside = _report("RC_Nevo", params, target_elas)

    return {
        "name": "RC_Nevo",
        "source": "Nevo (2001) Table VI / Conlon & Gortmaker (2020) Table 7",
        "type": "rc",
        "alpha_0": params['alpha_0'],
        "target_elas": target_elas,
        "params": params,
        "achieved_mean_abs_elas": float(np.abs(elas).mean()),
        "achieved_elas": elas.tolist(),
        "achieved_shares": shares.tolist(),
        "achieved_prices": prices.tolist(),
        "achieved_outside_share": float(outside),
    }


def calibrate_blp():
    """
    BLP (1995) / Conlon & Gortmaker (2020) Table 8 -- Automobiles.
    RC Logit. Price enters as alpha*ln(y-p) with alpha=45.898.
    We approximate with alpha_0 = -0.30 (≈ alpha/mean_income) and
    alpha_1 = 0.05 (income interaction) in our additive alpha_i*p structure.

    From Table 8 "Best Practices" column:
      - Constant mean: -6.679, Constant SD: 2.962
      - ln(y-p) coefficient: 45.898
      - Mean own-price elasticity: -3.461
      - Mean markup: 0.346
      - Prices in $1000s (~$5K-$40K)

    Target: |elas| ≈ 3.5
    """
    target_elas = 3.5

    params = {
        'alpha_0': -0.30, 'alpha_1': 0.05, 'alpha_2': 0,
        'gamma_0': 0.0,
        'gamma_1': 0.5,
        'gamma_2': 3.0,
        'delta_kxi': [2.0, 0.7, -0.7, -2.0],
        'p_base': 15.0, 'p_z_scale': 5.0, 'p_kxi_scale': 1.0,
    }

    print("\n  Calibrating gamma_0 for RC_BLP...")
    params['gamma_0'] = _calibrate_gamma0(params, target_elas, lo=-10.0, hi=20.0)
    print(f"  Found gamma_0 = {params['gamma_0']:.4f}")

    elas, shares, prices, outside = _report("RC_BLP", params, target_elas)

    return {
        "name": "RC_BLP",
        "source": "BLP (1995) / Conlon & Gortmaker (2020) Table 8",
        "type": "rc",
        "alpha_0": params['alpha_0'],
        "target_elas": target_elas,
        "params": params,
        "achieved_mean_abs_elas": float(np.abs(elas).mean()),
        "achieved_elas": elas.tolist(),
        "achieved_shares": shares.tolist(),
        "achieved_prices": prices.tolist(),
        "achieved_outside_share": float(outside),
    }


# =========================================================================
#  MAIN
# =========================================================================

CALIBRATORS = {
    "Logit_Wollmann": calibrate_wollmann,
    "RC_Nevo": calibrate_nevo,
    "RC_BLP": calibrate_blp,
}


def run_calibration(only=None):
    output_path = "./simulation_results/calibrated_params.json"
    try:
        with open(output_path) as f:
            all_params = json.load(f)
        print(f"Loaded existing calibrations: {list(all_params.keys())}")
    except FileNotFoundError:
        all_params = {}

    for name, calibrator in CALIBRATORS.items():
        if only and name != only:
            continue
        if name in all_params:
            print(f"Skipping {name} (already calibrated)")
            continue
        result = calibrator()
        all_params[name] = result

    with open(output_path, "w") as f:
        json.dump(all_params, f, indent=2)
    print(f"\nSaved calibrated parameters to {output_path}")
    return all_params


if __name__ == "__main__":
    only_config = sys.argv[1] if len(sys.argv) > 1 else None
    run_calibration(only=only_config)
