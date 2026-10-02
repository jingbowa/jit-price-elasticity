"""
Table 6: BNN elasticity recovery and implied pricing under literature-calibrated DGPs.

For each DGP calibrated to a published paper's demand parameters:
  1. Computes true own-price elasticities from the structural model.
  2. Estimates BNN elasticities over Monte Carlo replications.
  3. Backs out marginal costs via Lerner's rule (using point-wise true elasticities).
  4. Solves the single-product pricing FOC iteratively using both true and
     BNN-estimated elasticities.  Each product iterates independently, holding
     other products' prices at their sample means.

Parameters are loaded from simulation_results/calibrated_params.json
(produced by calibrate_literature.py).
Output: simulation_results/table_6_literature.json
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import json
from tqdm import tqdm

from bnn_modules.bnn_torch import p_elas_2scale_torch
from DGP_models.model_module_blp import (
    BLP_corr_parameterized, BLP_corr_elasticities,
    get_mean_kxi, true_own_elasticity_at_price,
)

# ---------------------------------------------------------------------------
#  Simulation parameters
# ---------------------------------------------------------------------------
NUM_MARKETS = 50_000
NUM_PRODUCTS = 4
NUM_DRAWS = 250
BNN_TUNING = 25
MC_ITERATIONS = 10

# FOC iteration parameters
FOC_TOL = 1e-5
FOC_MAX_ITER = 100
FOC_DAMPING = 0.5
TRUE_ELAS_DRAWS = 5000
TRUE_ELAS_SEED = 42

PARAMS_PATH = "./simulation_results/calibrated_params.json"
RESULTS_PATH = "./simulation_results/table_6_literature.json"


# ---------------------------------------------------------------------------
#  Helper functions
# ---------------------------------------------------------------------------

def load_configs():
    with open(PARAMS_PATH) as f:
        return json.load(f)


def compute_true_elasticities(params):
    """Market-averaged true own-price elasticities (negative convention)."""
    np.random.seed(99999)
    torch.manual_seed(99999)
    elas = BLP_corr_elasticities(
        num_markets=50_000, num_products=NUM_PRODUCTS, num_draws=1000,
        params=params)
    return -elas


def back_out_costs_pointwise(mean_prices_t, kxi_vals, params):
    """Infer marginal costs from Lerner's rule using point-wise true elasticities.

    Using the same elasticity function as the FOC iteration avoids
    Jensen's-inequality mismatch from market-averaged elasticities.
    """
    J = len(mean_prices_t)
    costs = np.zeros(J)
    point_elas = np.zeros(J)
    for j in range(J):
        eps = true_own_elasticity_at_price(
            float(mean_prices_t[j]), j, mean_prices_t, kxi_vals, params,
            num_draws=TRUE_ELAS_DRAWS, seed=TRUE_ELAS_SEED)
        point_elas[j] = eps
        costs[j] = float(mean_prices_t[j]) * (1.0 + 1.0 / eps)
    return costs, point_elas


def _foc_step(c_j, eps):
    return float(c_j) * eps / (eps + 1.0)


def iterate_foc_true(c_j, j, mean_prices, kxi_vals, params, p_init,
                     tol=FOC_TOL, max_iter=FOC_MAX_ITER, damping=FOC_DAMPING):
    """Solve the single-product FOC with the true demand model."""
    p = float(p_init)
    for k in range(max_iter):
        eps = true_own_elasticity_at_price(
            p, j, mean_prices, kxi_vals, params,
            num_draws=TRUE_ELAS_DRAWS, seed=TRUE_ELAS_SEED)
        p_new = damping * _foc_step(c_j, eps) + (1 - damping) * p
        if abs(p_new - p) < tol:
            return p_new, k + 1
        p = p_new
    return p, max_iter


def estimate_bnn_elas_at_price(price_j, j, y_j, XZ_j, mean_xz, tuning):
    """BNN elasticity for product j evaluated at a specific price."""
    xz = mean_xz.clone()
    xz[j] = float(price_j)
    return p_elas_2scale_torch(
        y_j, XZ_j, xz,
        p1_col_idx=j,
        z_col_idx=XZ_j.shape[1] - 1,
        s=tuning,
    ).item()


def iterate_foc_bnn(c_j, j, y_j, XZ_j, mean_xz, tuning, p_init,
                    tol=FOC_TOL, max_iter=FOC_MAX_ITER, damping=FOC_DAMPING):
    """Solve the single-product FOC with BNN-estimated elasticities."""
    p = float(p_init)
    for k in range(max_iter):
        eps = estimate_bnn_elas_at_price(p, j, y_j, XZ_j, mean_xz, tuning)
        if eps >= -1.0:
            return p, k + 1
        p_new = damping * _foc_step(c_j, eps) + (1 - damping) * p
        if abs(p_new - p) < tol:
            return p_new, k + 1
        p = p_new
    return p, max_iter


# ---------------------------------------------------------------------------
#  Main simulation loop
# ---------------------------------------------------------------------------

def run_simulation(only_config=None):
    configs = load_configs()

    try:
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        print(f"Loaded existing results: {list(all_results.keys())}")
    except FileNotFoundError:
        all_results = {}

    for config_name, config in configs.items():
        if only_config and config_name != only_config:
            continue
        if config_name in all_results:
            print(f"\nSkipping {config_name} (already completed)")
            continue

        params = config["params"]
        source = config.get("source", "")
        model_type = config.get("type", "logit")

        print(f"\n{'='*60}")
        print(f"Configuration: {config_name}")
        print(f"  Source: {source}")
        print(f"  Type: {model_type}")
        print(f"{'='*60}")

        # --- True elasticities (market-averaged) ---
        true_elas = compute_true_elasticities(params)
        print(f"  True own-price elasticities: {np.round(true_elas, 4)}")

        # --- Reference data for mean shares / prices ---
        np.random.seed(12345)
        torch.manual_seed(12345)
        s_jt_ref, p_jt_ref, _, _, _, _, _ = BLP_corr_parameterized(
            NUM_MARKETS, NUM_PRODUCTS, NUM_DRAWS, params)
        mean_shares = s_jt_ref.mean(dim=0).numpy()
        mean_prices = p_jt_ref.mean(dim=0).numpy()
        mean_prices_t = torch.tensor(mean_prices, dtype=torch.float32)

        # --- Pricing setup: costs and true optimal prices ---
        kxi_vals = get_mean_kxi(params, NUM_PRODUCTS, seed=99999)
        costs, point_elas = back_out_costs_pointwise(
            mean_prices_t, kxi_vals, params)
        markups = (mean_prices - costs) / mean_prices
        print(f"  Point-wise elas: {np.round(point_elas, 3)}")
        print(f"  Costs:   {np.round(costs, 4)}")
        print(f"  Markups: {np.round(markups, 4)}")

        true_p_star = np.zeros(NUM_PRODUCTS)
        true_iters = np.zeros(NUM_PRODUCTS, dtype=int)
        for j in range(NUM_PRODUCTS):
            true_p_star[j], true_iters[j] = iterate_foc_true(
                costs[j], j, mean_prices_t, kxi_vals, params,
                p_init=mean_prices[j])
        print(f"  True p*: {np.round(true_p_star, 4)}")

        # --- Monte Carlo: BNN elasticity + BNN pricing ---
        mc_bnn_elas = []
        mc_bnn_p_star = []
        mc_bnn_iters = []

        for i in tqdm(range(MC_ITERATIONS), desc=config_name):
            seed = 12345 + i
            np.random.seed(seed)
            torch.manual_seed(seed)

            s_jt, p_jt, z_jt, _, _, _, _ = BLP_corr_parameterized(
                NUM_MARKETS, NUM_PRODUCTS, NUM_DRAWS, params)

            bnn_elas_i = np.zeros(NUM_PRODUCTS)
            bnn_p_star_i = np.zeros(NUM_PRODUCTS)
            bnn_iters_i = np.zeros(NUM_PRODUCTS, dtype=int)

            for j in range(NUM_PRODUCTS):
                y_j = s_jt[:, j].double()
                XZ_j = torch.cat((p_jt, z_jt[:, [j]]), dim=1).double()
                mean_xz_j = XZ_j.mean(dim=0)

                bnn_elas_i[j] = p_elas_2scale_torch(
                    y_j, XZ_j, mean_xz_j,
                    p1_col_idx=j, z_col_idx=NUM_PRODUCTS, s=BNN_TUNING,
                ).item()

                bnn_p_star_i[j], bnn_iters_i[j] = iterate_foc_bnn(
                    costs[j], j, y_j, XZ_j, mean_xz_j, BNN_TUNING,
                    p_init=mean_prices[j])

            mc_bnn_elas.append(bnn_elas_i.tolist())
            mc_bnn_p_star.append(bnn_p_star_i.tolist())
            mc_bnn_iters.append(bnn_iters_i.tolist())

        bnn_elas_mean = np.array(mc_bnn_elas).mean(axis=0)
        bnn_p_mean = np.array(mc_bnn_p_star).mean(axis=0)
        price_bias = bnn_p_mean - true_p_star

        print(f"\n  Summary for {config_name}:")
        print(f"    True elas:  {np.round(true_elas, 4)}")
        print(f"    BNN elas:   {np.round(bnn_elas_mean, 4)}")
        print(f"    Shares:     {np.round(mean_shares, 4)}")
        print(f"    Prices:     {np.round(mean_prices, 2)}")
        print(f"    True p*:    {np.round(true_p_star, 4)}")
        print(f"    BNN p*:     {np.round(bnn_p_mean, 4)}")
        print(f"    Price bias: {np.round(price_bias, 4)}")

        all_results[config_name] = {
            "source": source,
            "type": model_type,
            "alpha_0": params["alpha_0"],
            "true_elas": true_elas.tolist(),
            "bnn_elas_mean": bnn_elas_mean.tolist(),
            "bnn_elas_all": mc_bnn_elas,
            "mean_shares": mean_shares.tolist(),
            "mean_prices": mean_prices.tolist(),
            "mc_iterations": MC_ITERATIONS,
            "costs": costs.tolist(),
            "markups": markups.tolist(),
            "point_elas": point_elas.tolist(),
            "true_p_star": true_p_star.tolist(),
            "true_iters": true_iters.tolist(),
            "bnn_p_star_mean": bnn_p_mean.tolist(),
            "bnn_p_star_all": mc_bnn_p_star,
            "bnn_iters_all": mc_bnn_iters,
            "price_bias": price_bias.tolist(),
        }

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    print_summary(all_results)
    return all_results


# ---------------------------------------------------------------------------
#  Summary
# ---------------------------------------------------------------------------

def print_summary(all_results):
    print(f"\n{'='*70}")
    print("TABLE 6: Elasticity Recovery and Implied Pricing")
    print(f"{'='*70}")

    for name, r in all_results.items():
        true_e = np.array(r["true_elas"])
        bnn_e = np.array(r["bnn_elas_mean"])
        true_p = np.array(r["true_p_star"])
        bnn_p = np.array(r["bnn_p_star_mean"])
        shares = np.array(r["mean_shares"])
        prices = np.array(r["mean_prices"])

        print(f"\n--- {name} ({r['type']}, source: {r['source']}) ---")
        print(f"  {'Prod':>4}  {'True eps':>9}  {'BNN eps':>9}  {'eps Bias':>9}"
              f"  {'True p*':>9}  {'BNN p*':>9}  {'p* Bias':>9}")
        for j in range(len(true_e)):
            print(f"  {j+1:>4}  {true_e[j]:>9.3f}  {bnn_e[j]:>9.3f}"
                  f"  {bnn_e[j]-true_e[j]:>9.3f}"
                  f"  {true_p[j]:>9.3f}  {bnn_p[j]:>9.3f}"
                  f"  {bnn_p[j]-true_p[j]:>9.3f}")


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    run_simulation(only_config=only)
