"""
Table 6: Optimal prices from BNN and PyBLP under baseline DGPs.

For each of three baseline demand models (logit, independent RC logit,
correlated RC logit):
  1. Computes true optimal prices by iterating the pricing FOC with the
     true demand model.
  2. Computes BNN optimal prices by iterating the FOC with BNN-estimated
     elasticities.
  3. Computes PyBLP optimal prices via pyblp's compute_prices().

Each product is treated as a single-product firm (Bertrand-Nash).
Results are averaged across products and Monte Carlo iterations.

Output: simulation_results/table_6_baseline_pricing.json
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import json
import time
import pyblp
from tqdm import tqdm

from bnn_modules.bnn_torch import p_elas_2scale_torch
from DGP_models.model_module_blp import (
    BLP_corr_parameterized, BLP_corr_elasticities,
    get_mean_kxi, true_own_elasticity_at_price,
)

# ---------------------------------------------------------------------------
#  Simulation parameters
# ---------------------------------------------------------------------------
NUM_MARKETS = 20_000
NUM_PRODUCTS = 4
NUM_DRAWS = 250
BNN_TUNING = 7
MC_ITERATIONS = 10
PYBLP_MC_ITERATIONS = 2

FOC_TOL = 1e-5
FOC_MAX_ITER = 100
FOC_DAMPING = 0.5
TRUE_ELAS_DRAWS = 5000
TRUE_ELAS_SEED = 42

RESULTS_PATH = "./simulation_results/table_6_baseline_pricing.json"

COMMON_COSTS = np.array([0.762, 0.699, 0.687, 0.561])

# ---------------------------------------------------------------------------
#  Three baseline DGP configurations
# ---------------------------------------------------------------------------
CONFIGS = {
    "logit": {
        "label": "Logit",
        "params": {
            'alpha_0': -3, 'alpha_1': 0, 'alpha_2': 0,
            'gamma_0': 0.9, 'gamma_1': 0, 'gamma_2': 0,
            'p_base': 1.0, 'p_z_scale': 1.0, 'p_kxi_scale': 0.2,
        },
    },
    "ind_rc": {
        "label": "Independent RC logit",
        "params": {
            'alpha_0': -3, 'alpha_1': 0, 'alpha_2': 0.3,
            'gamma_0': 0.9, 'gamma_1': 0, 'gamma_2': 0,
            'p_base': 1.0, 'p_z_scale': 1.0, 'p_kxi_scale': 0.2,
        },
    },
    "corr_rc": {
        "label": "Correlated RC logit",
        "params": {
            'alpha_0': -3, 'alpha_1': 0.8, 'alpha_2': 0.3,
            'gamma_0': 0.9, 'gamma_1': 0.5, 'gamma_2': 0,
            'p_base': 1.0, 'p_z_scale': 1.0, 'p_kxi_scale': 0.2,
        },
    },
}


# ---------------------------------------------------------------------------
#  FOC helpers (same pattern as table_6_literature.py)
# ---------------------------------------------------------------------------

def back_out_costs_pointwise(mean_prices_t, kxi_vals, params):
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


def iterate_foc_true(c_j, j, mean_prices, kxi_vals, params, p_init):
    p = float(p_init)
    for k in range(FOC_MAX_ITER):
        eps = true_own_elasticity_at_price(
            p, j, mean_prices, kxi_vals, params,
            num_draws=TRUE_ELAS_DRAWS, seed=TRUE_ELAS_SEED)
        p_new = FOC_DAMPING * _foc_step(c_j, eps) + (1 - FOC_DAMPING) * p
        if abs(p_new - p) < FOC_TOL:
            return p_new, k + 1
        p = p_new
    return p, FOC_MAX_ITER


def iterate_foc_bnn(c_j, j, y_j, XZ_j, mean_xz, tuning, p_init):
    p = float(p_init)
    for k in range(FOC_MAX_ITER):
        xz = mean_xz.clone()
        xz[j] = float(p)
        eps = p_elas_2scale_torch(
            y_j, XZ_j, xz, p1_col_idx=j,
            z_col_idx=XZ_j.shape[1] - 1, s=tuning).item()
        if eps >= -1.0:
            return p, k + 1
        p_new = FOC_DAMPING * _foc_step(c_j, eps) + (1 - FOC_DAMPING) * p
        if abs(p_new - p) < FOC_TOL:
            return p_new, k + 1
        p = p_new
    return p, FOC_MAX_ITER


# ---------------------------------------------------------------------------
#  PyBLP estimation per model type
# ---------------------------------------------------------------------------

def _pyblp_logit(product_data):
    """Estimate logit demand via PyBLP (simple IV, no random coefficients)."""
    X1 = pyblp.Formulation('1 + prices')
    problem = pyblp.Problem(
        product_formulations=(X1,),
        product_data=product_data)
    result = problem.solve()
    return result


def _pyblp_ind_rc(product_data):
    """Estimate independent RC logit (RC on price only)."""
    X1 = pyblp.Formulation('1 + prices')
    X2 = pyblp.Formulation('0 + prices')
    problem = pyblp.Problem(
        product_formulations=(X1, X2),
        product_data=product_data,
        integration=pyblp.Integration('halton', size=200,
                                      specification_options={'seed': 0}))
    result = problem.solve(
        sigma=np.array([[0.1]]),
        optimization=pyblp.Optimization('l-bfgs-b', {'gtol': 1e-6}),
        iteration=pyblp.Iteration('squarem', {'atol': 1e-12}))
    return result


def _pyblp_corr_rc(product_data, agent_data, micro_statistics):
    """Estimate correlated RC logit with micro moments (full BLP)."""
    X1 = pyblp.Formulation('1 + prices')
    X2 = pyblp.Formulation('1 + prices + high_prices')
    problem = pyblp.Problem(
        product_formulations=(X1, X2),
        product_data=product_data,
        agent_formulation=pyblp.Formulation('1 + log_income + low + mid + high'),
        agent_data=agent_data,
        integration=pyblp.Integration('halton', size=250,
                                      specification_options={'seed': 0}))

    micro_dataset = pyblp.MicroDataset(
        name="Income Survey", observations=50_000,
        compute_weights=lambda t, p, a: np.ones((a.size, 1 + p.size)))

    parts = {
        'price_mid': pyblp.MicroPart(
            name="E[hp*mid]", dataset=micro_dataset,
            compute_values=lambda t, p, a: np.outer(
                a.demographics[:, 3], np.r_[0, p.X2[:, 2]])),
        'price_high': pyblp.MicroPart(
            name="E[hp*high]", dataset=micro_dataset,
            compute_values=lambda t, p, a: np.outer(
                a.demographics[:, 4], np.r_[0, p.X2[:, 2]])),
        'inside_mid': pyblp.MicroPart(
            name="E[1{j>0}*mid]", dataset=micro_dataset,
            compute_values=lambda t, p, a: np.outer(
                a.demographics[:, 3], np.r_[0, p.X2[:, 0]])),
        'inside_high': pyblp.MicroPart(
            name="E[1{j>0}*high]", dataset=micro_dataset,
            compute_values=lambda t, p, a: np.outer(
                a.demographics[:, 4], np.r_[0, p.X2[:, 0]])),
        'mid': pyblp.MicroPart(
            name="E[mid]", dataset=micro_dataset,
            compute_values=lambda t, p, a: np.outer(
                a.demographics[:, 3], np.r_[1, p.X2[:, 0]])),
        'high': pyblp.MicroPart(
            name="E[high]", dataset=micro_dataset,
            compute_values=lambda t, p, a: np.outer(
                a.demographics[:, 4], np.r_[1, p.X2[:, 0]])),
    }

    ratio = lambda v: v[0] / v[1]
    ratio_grad = lambda v: [1 / v[1], -v[0] / v[1]**2]

    micro_moments = [
        pyblp.MicroMoment(name="E[hp|mid]",
            value=micro_statistics.purchase_high_price_given_mid_income[0],
            parts=[parts['price_mid'], parts['mid']],
            compute_value=ratio, compute_gradient=ratio_grad),
        pyblp.MicroMoment(name="E[hp|high]",
            value=micro_statistics.purchase_high_price_given_high_income[0],
            parts=[parts['price_high'], parts['high']],
            compute_value=ratio, compute_gradient=ratio_grad),
        pyblp.MicroMoment(name="E[in|mid]",
            value=micro_statistics.purchase_prob_given_mid_income[0],
            parts=[parts['inside_mid'], parts['mid']],
            compute_value=ratio, compute_gradient=ratio_grad),
        pyblp.MicroMoment(name="E[in|high]",
            value=micro_statistics.purchase_prob_given_high_income[0],
            parts=[parts['inside_high'], parts['high']],
            compute_value=ratio, compute_gradient=ratio_grad),
    ]

    sigma0 = np.diag([0, 0.1, 0])
    pi0 = np.array([[0, 0.1, 0, 0, 0],
                     [0, 0.1, 0, 0, 0],
                     [0, 0, 0, 0, 0]])
    sigma_ub = np.diag([0, 2, 0])
    sigma_lb = np.diag([0, -2, 0])
    pi_ub = np.array([[0, 2, 0, 0, 0],
                      [0, 2, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    pi_lb = np.zeros_like(pi0)

    result = problem.solve(
        sigma=sigma0, pi=pi0,
        optimization=pyblp.Optimization('l-bfgs-b', {'gtol': 1e-8}),
        iteration=pyblp.Iteration('squarem', {'atol': 1e-8}),
        micro_moments=micro_moments,
        sigma_bounds=(sigma_lb, sigma_ub),
        pi_bounds=(pi_lb, pi_ub))
    return result


PYBLP_SOLVERS = {
    "logit": lambda pd, ad, ms: _pyblp_logit(pd),
    "ind_rc": lambda pd, ad, ms: _pyblp_ind_rc(pd),
    "corr_rc": lambda pd, ad, ms: _pyblp_corr_rc(pd, ad, ms),
}


# ---------------------------------------------------------------------------
#  Main simulation
# ---------------------------------------------------------------------------

def run_simulation(only_config=None, mc_iters=None):
    if mc_iters is None:
        mc_iters = MC_ITERATIONS

    try:
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        print(f"Loaded existing results: {list(all_results.keys())}")
    except FileNotFoundError:
        all_results = {}

    for config_name, cfg in CONFIGS.items():
        if only_config and config_name != only_config:
            continue
        if config_name in all_results:
            print(f"\nSkipping {config_name} (already completed)")
            continue

        params = cfg["params"]
        label = cfg["label"]

        print(f"\n{'='*60}")
        print(f"{label}  ({config_name})")
        print(f"{'='*60}")

        # --- Reference data for mean prices ---
        np.random.seed(12345)
        torch.manual_seed(12345)
        s_jt_ref, p_jt_ref, _, _, _, _, _ = BLP_corr_parameterized(
            NUM_MARKETS, NUM_PRODUCTS, NUM_DRAWS, params)
        mean_prices = p_jt_ref.mean(dim=0).numpy()
        mean_shares = s_jt_ref.mean(dim=0).numpy()
        mean_prices_t = torch.tensor(mean_prices, dtype=torch.float32)

        # --- True elasticities (market-averaged) ---
        np.random.seed(99999)
        torch.manual_seed(99999)
        true_elas = -BLP_corr_elasticities(
            num_markets=NUM_MARKETS, num_products=NUM_PRODUCTS,
            num_draws=1000, params=params)
        print(f"  True elas (mkt avg): {np.round(true_elas, 3)}")

        # --- Costs (common across all DGPs) ---
        kxi_vals = get_mean_kxi(params, NUM_PRODUCTS, seed=99999)
        costs = COMMON_COSTS
        print(f"  Costs (common): {np.round(costs, 4)}")

        # --- True optimal prices ---
        true_p_star = np.zeros(NUM_PRODUCTS)
        for j in range(NUM_PRODUCTS):
            true_p_star[j], _ = iterate_foc_true(
                costs[j], j, mean_prices_t, kxi_vals, params,
                p_init=mean_prices[j])
        print(f"  True p*: {np.round(true_p_star, 4)}")

        # --- Monte Carlo ---
        mc_bnn_p = []
        mc_pyblp_p = []
        mc_bnn_times = []
        mc_pyblp_times = []
        pyblp_solver = PYBLP_SOLVERS[config_name]

        for i in tqdm(range(mc_iters), desc=config_name):
            seed = 12345 + i
            np.random.seed(seed)
            torch.manual_seed(seed)

            s_jt, p_jt, z_jt, _, product_data, agent_data, micro_stats = \
                BLP_corr_parameterized(
                    NUM_MARKETS, NUM_PRODUCTS, NUM_DRAWS, params)

            # --- BNN pricing (timed: data ready -> optimal prices) ---
            t0_bnn = time.time()
            bnn_p_i = np.zeros(NUM_PRODUCTS)
            for j in range(NUM_PRODUCTS):
                y_j = s_jt[:, j].double()
                XZ_j = torch.cat((p_jt, z_jt[:, [j]]), dim=1).double()
                mean_xz_j = XZ_j.mean(dim=0)
                bnn_p_i[j], _ = iterate_foc_bnn(
                    costs[j], j, y_j, XZ_j, mean_xz_j, BNN_TUNING,
                    p_init=mean_prices[j])
            bnn_elapsed = time.time() - t0_bnn
            mc_bnn_p.append(bnn_p_i.tolist())
            mc_bnn_times.append(bnn_elapsed)

            # --- PyBLP pricing (timed: data ready -> optimal prices) ---
            if i < PYBLP_MC_ITERATIONS:
                try:
                    t0_pyblp = time.time()
                    result = pyblp_solver(product_data, agent_data, micro_stats)
                    costs_per_mkt = np.tile(costs, NUM_MARKETS)
                    pyblp_prices = result.compute_prices(
                        costs=costs_per_mkt.reshape(-1, 1))
                    pyblp_elapsed = time.time() - t0_pyblp
                    product_data_copy = product_data.copy()
                    product_data_copy['pyblp_prices'] = pyblp_prices.flatten()
                    avg_pyblp_p = product_data_copy.groupby(
                        'product_ids')['pyblp_prices'].mean().values
                    mc_pyblp_p.append(avg_pyblp_p.tolist())
                    mc_pyblp_times.append(pyblp_elapsed)
                    print(f"    MC {i+1}: BNN p={np.round(bnn_p_i, 4)} ({bnn_elapsed:.2f}s)  "
                          f"PyBLP p={np.round(avg_pyblp_p, 4)} ({pyblp_elapsed:.1f}s)")
                except Exception as e:
                    print(f"    MC {i+1}: BNN p={np.round(bnn_p_i, 4)} ({bnn_elapsed:.2f}s)  "
                          f"PyBLP FAILED -- {e}")
                    mc_pyblp_p.append([np.nan] * NUM_PRODUCTS)
            else:
                print(f"    MC {i+1}: BNN p={np.round(bnn_p_i, 4)} ({bnn_elapsed:.2f}s)")

        bnn_p_mean = np.array(mc_bnn_p).mean(axis=0)
        pyblp_p_mean = np.nanmean(mc_pyblp_p, axis=0) if mc_pyblp_p else np.full(NUM_PRODUCTS, np.nan)

        bnn_time_mean = float(np.mean(mc_bnn_times))
        pyblp_time_mean = float(np.mean(mc_pyblp_times)) if mc_pyblp_times else float('nan')

        print(f"\n  Summary for {label}:")
        print(f"    Mean prices: {np.round(mean_prices, 4)}")
        print(f"    True p*:     {np.round(true_p_star, 4)}")
        print(f"    BNN p*:      {np.round(bnn_p_mean, 4)}  "
              f"bias={np.round(bnn_p_mean - true_p_star, 4)}  "
              f"time={bnn_time_mean:.2f}s")
        print(f"    PyBLP p*:    {np.round(pyblp_p_mean, 4)}  "
              f"bias={np.round(pyblp_p_mean - true_p_star, 4)}  "
              f"time={pyblp_time_mean:.1f}s ({pyblp_time_mean/60:.1f}min)")

        all_results[config_name] = {
            "label": label,
            "mean_prices": mean_prices.tolist(),
            "mean_shares": mean_shares.tolist(),
            "true_elas": true_elas.tolist(),
            "costs": costs.tolist(),
            "true_p_star": true_p_star.tolist(),
            "bnn_p_star_mean": bnn_p_mean.tolist(),
            "bnn_p_star_all": mc_bnn_p,
            "bnn_bias": (bnn_p_mean - true_p_star).tolist(),
            "bnn_time_seconds": mc_bnn_times,
            "bnn_time_mean": bnn_time_mean,
            "pyblp_p_star_mean": pyblp_p_mean.tolist(),
            "pyblp_p_star_all": mc_pyblp_p,
            "pyblp_bias": (pyblp_p_mean - true_p_star).tolist(),
            "pyblp_time_seconds": mc_pyblp_times,
            "pyblp_time_mean": pyblp_time_mean,
            "mc_iterations": mc_iters,
            "pyblp_mc_iterations": len(mc_pyblp_p),
        }

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    print_summary(all_results)
    return all_results


def print_summary(all_results):
    print(f"\n{'='*70}")
    print("TABLE 6: Baseline Pricing")
    print(f"{'='*70}")
    print(f"  {'Model':<25}  {'True p*':>9}  {'BNN p*':>9}  {'BNN Bias':>9}  {'BNN Time':>9}"
          f"  {'PyBLP p*':>9}  {'PyBLP Bias':>10}  {'PyBLP Time':>11}")
    for name, r in all_results.items():
        tp = np.mean(r['true_p_star'])
        bp = np.mean(r['bnn_p_star_mean'])
        pp = np.nanmean(r.get('pyblp_p_star_mean', [np.nan]))
        bt = r.get('bnn_time_mean', float('nan'))
        pt = r.get('pyblp_time_mean', float('nan'))
        print(f"  {r['label']:<25}  {tp:>9.4f}  {bp:>9.4f}  {bp-tp:>9.4f}  {bt:>8.2f}s"
              f"  {pp:>9.4f}  {pp-tp:>10.4f}  {pt/60:>9.1f}min")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this config (logit / ind_rc / corr_rc)")
    parser.add_argument("--pilot", action="store_true",
                        help="Pilot mode: 2 MC iterations only")
    args = parser.parse_args()
    mc = 2 if args.pilot else None
    run_simulation(only_config=args.only, mc_iters=mc)
