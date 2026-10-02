import torch
import numpy as np
import pandas as pd

DEFAULT_PARAMS = {
    'alpha_0': -3, 'alpha_1': 0.8, 'alpha_2': 0.3,
    'gamma_0': -1.5, 'gamma_1': 0.5, 'gamma_2': 0,
    'p_base': 0.2, 'p_z_scale': 1.0, 'p_kxi_scale': 0.2,
}

def BLP_corr(num_markets, num_products, num_draws):
    """Original interface with hardcoded default parameters."""
    return BLP_corr_parameterized(num_markets, num_products, num_draws, DEFAULT_PARAMS)

def BLP_corr_parameterized(num_markets, num_products, num_draws, params):
    """Parameterized BLP DGP. Supports logit (all heterogeneity params = 0) through full RC."""
    alpha_0 = params['alpha_0']
    alpha_1 = params.get('alpha_1', 0)
    alpha_2 = params.get('alpha_2', 0)
    gamma_0 = params['gamma_0']
    gamma_1 = params.get('gamma_1', 0)
    gamma_2 = params.get('gamma_2', 0)

    p_base = torch.tensor(params.get('p_base', 0.2), dtype=torch.float32)
    p_z_scale = params.get('p_z_scale', 1.0)
    p_kxi_scale = params.get('p_kxi_scale', 0.2)

    delta_kxi = torch.tensor(params.get('delta_kxi', [0.5, 0.25, 0, -0.25]))
    kxi = torch.rand(num_products) - 0.5
    kxi_jt = kxi.unsqueeze(0).repeat(num_markets, 1) + 0.5*delta_kxi
    
    z_jt = torch.rand(num_markets, num_products) - 0.5
    
    p_jt = p_base + p_z_scale * z_jt + p_kxi_scale * kxi_jt
    high_price_jt = p_jt > p_jt.median()

    p_jt_pred = p_base + p_z_scale * z_jt

    inc = torch.exp(torch.randn(num_draws*5))
    log_income = torch.log(inc)
    nu1, nu2 = torch.randn(num_draws*5), torch.randn(num_draws*5)
    
    gamma_i = gamma_0 + gamma_1 * log_income + gamma_2 * nu1
    alpha_i = alpha_0 + alpha_1 * log_income + alpha_2 * nu2
    
    # Reshape beta_i and alpha_i to match the dimensions of x_jt and p_jt
    gamma_i_3d = gamma_i.unsqueeze(0).unsqueeze(0).repeat(num_markets, num_products, 1)
    alpha_i_3d = alpha_i.unsqueeze(0).unsqueeze(0).repeat(num_markets, num_products, 1)

    # Compute individual choice probabilities
    u_ijt = gamma_i_3d + alpha_i_3d * p_jt.unsqueeze(2) + kxi_jt.unsqueeze(2)

    exp_u_ijt = torch.exp(u_ijt)
    sum_exp_u_ijt = torch.sum(exp_u_ijt, dim=1, keepdim=True)
    s_ijt = exp_u_ijt / (1 + sum_exp_u_ijt)
    
    # Compute aggregate market shares
    s_jt = torch.mean(s_ijt, dim=2)

    # Count products within price ranges (adaptive to actual price variation)
    pairwise_diffs = p_jt_pred.unsqueeze(2) - p_jt_pred.unsqueeze(1)
    mask = ~torch.eye(num_products, dtype=torch.bool).unsqueeze(0)
    diff_std = pairwise_diffs[mask.expand_as(pairwise_diffs)].std().item()
    price_ranges = torch.tensor([-diff_std, 0, diff_std])
    counts = torch.zeros((num_markets, num_products, len(price_ranges)))
    
    for m in range(num_markets):
        for j in range(num_products):
            focal_price = p_jt_pred[m, j]
            price_diff = p_jt_pred[m, :] - focal_price
            for i, range_val in enumerate(price_ranges):
                if i == 0:
                    counts[m, j, i] = torch.sum((price_diff <= range_val) & (price_diff != 0))
                else:
                    counts[m, j, i] = torch.sum((price_diff > price_ranges[i-1]) & (price_diff <= range_val) & (price_diff != 0))

    # Calculate income terciles
    income_terciles = torch.quantile(log_income, torch.tensor([1/3, 2/3]))

    # Create income group indicators
    mid_income = (log_income > income_terciles[0]) & (log_income <= income_terciles[1])
    high_income = log_income > income_terciles[1]

    # Calculate purchase probabilities by income group
    purchase_prob_given_mid_income = s_ijt[:, :, mid_income].sum(dim=1).mean()
    purchase_prob_given_high_income = s_ijt[:, :, high_income].sum(dim=1).mean()

    # Share times price
    share_times_high_price = s_ijt * high_price_jt.unsqueeze(2)

     # Purchase price conditional on income groups
    purchase_high_price_given_mid_income = share_times_high_price[:, :, mid_income].sum(dim=1).mean()
    purchase_high_price_given_high_income = share_times_high_price[:, :, high_income].sum(dim=1).mean()

    # Create product data
    product_data = pd.DataFrame({
        'market_ids': np.repeat(range(num_markets), num_products),
        "firm_ids": np.tile(range(num_products), num_markets),
        "product_ids": np.tile(range(num_products), num_markets),
        'shares': s_jt.flatten().numpy(),
        'prices': p_jt.flatten().numpy(),
        'high_prices': high_price_jt.flatten().numpy(),
        'costs': z_jt.flatten().numpy(),
        'demand_instruments0': z_jt.flatten().numpy()
    })

    # Add count columns to product_data
    for i, range_val in enumerate(price_ranges):
        r = i + 1
        product_data[f'demand_instruments{r}'] = counts[:, :, i].flatten().numpy()

    # Create agent data
    inc2 = inc[:num_draws]
    log_income2 = log_income[:num_draws]
    agent_data = pd.DataFrame({
        'market_ids': np.repeat(range(num_markets), num_draws),
        'log_income': log_income2.repeat(num_markets).numpy(),
        'low': (log_income2.repeat(num_markets) <= income_terciles[0]).numpy().astype(int),
        'mid': ((log_income2.repeat(num_markets) > income_terciles[0]) & 
                (log_income2.repeat(num_markets) <= income_terciles[1])).numpy().astype(int),
        'high': (log_income2.repeat(num_markets) > income_terciles[1]).numpy().astype(int),
        'weights': np.ones(num_markets * num_draws) / num_draws
    })

    # Create micro statistics
    micro_statistics = pd.DataFrame({
        'purchase_prob_given_mid_income': [purchase_prob_given_mid_income.item()],
        'purchase_prob_given_high_income': [purchase_prob_given_high_income.item()],
        'purchase_high_price_given_mid_income': [purchase_high_price_given_mid_income.item()],
        'purchase_high_price_given_high_income': [purchase_high_price_given_high_income.item()]
    })

    return s_jt, p_jt, z_jt, inc, product_data, agent_data, micro_statistics

def BLP_corr_elasticities(num_markets=100, num_products=4, num_draws=200, params=None):
    """Compute true own-price elasticities from the DGP."""
    if params is None:
        params = DEFAULT_PARAMS

    alpha_0 = params['alpha_0']
    alpha_1 = params.get('alpha_1', 0)
    alpha_2 = params.get('alpha_2', 0)
    gamma_0 = params['gamma_0']
    gamma_1 = params.get('gamma_1', 0)
    gamma_2 = params.get('gamma_2', 0)

    p_base = torch.tensor(params.get('p_base', 0.2), dtype=torch.float32)
    p_z_scale = params.get('p_z_scale', 1.0)
    p_kxi_scale = params.get('p_kxi_scale', 0.2)

    delta_kxi = torch.tensor(params.get('delta_kxi', [0.5, 0.25, 0, -0.25]))
    kxi = torch.rand(num_products) - 0.5
    kxi_jt = kxi.unsqueeze(0).repeat(num_markets, 1) + 0.5*delta_kxi

    z_jt = torch.rand(num_markets, num_products) - 0.5
    p_jt = p_base + p_z_scale * z_jt + p_kxi_scale * kxi_jt

    inc = torch.exp(torch.randn(num_draws))
    log_income = torch.log(inc)
    nu1, nu2 = torch.randn(num_draws), torch.randn(num_draws)
    
    gamma_i = gamma_0 + gamma_1 * log_income + gamma_2 * nu1
    alpha_i = alpha_0 + alpha_1 * log_income + alpha_2 * nu2
    
    # Reshape gamma_i and alpha_i to match the dimensions of kxi_jt and p_jt
    gamma_i_3d = gamma_i.unsqueeze(0).unsqueeze(0).repeat(num_markets, num_products, 1)
    alpha_i_3d = alpha_i.unsqueeze(0).unsqueeze(0).repeat(num_markets, num_products, 1)

    # Compute individual choice probabilities
    u_ijt = gamma_i_3d + alpha_i_3d * p_jt.unsqueeze(2) + kxi_jt.unsqueeze(2)
    exp_u_ijt = torch.exp(u_ijt)
    sum_exp_u_ijt = torch.sum(exp_u_ijt, dim=1, keepdim=True)
    s_ijt = exp_u_ijt / (1 + sum_exp_u_ijt)
    
    # Compute aggregate market shares
    s_jt = torch.mean(s_ijt, dim=2)

    # Calculate elasticities
    own_elasticities = torch.zeros(num_products)
    dsdp = -alpha_i_3d * s_ijt * (1 - s_ijt)
    own_elasticities = torch.mean(dsdp * p_jt.unsqueeze(2) / s_jt.unsqueeze(2), dim=(0, 2))

    return own_elasticities.numpy()


def get_mean_kxi(params, num_products=4, seed=99999):
    """Return mean product-level demand shifters for a given parameterization and seed."""
    torch.manual_seed(seed)
    delta_kxi = torch.tensor(params.get('delta_kxi', [0.5, 0.25, 0, -0.25]),
                              dtype=torch.float32)
    kxi = torch.rand(num_products) - 0.5
    return kxi + 0.5 * delta_kxi


def true_own_elasticity_at_price(price_j, j, all_prices, kxi_vals, params,
                                  num_draws=5000, seed=42):
    """Compute true own-price elasticity for product j at a specific price_j.

    Evaluates the (RC) logit demand model at a single deterministic price vector,
    integrating over consumer heterogeneity.  Returns the conventional (negative)
    elasticity: eps_j = (ds_j/dp_j) * p_j / s_j.

    Args:
        price_j: scalar price for product j.
        j: product index (0-based).
        all_prices: (num_products,) tensor of prices for all products.
        kxi_vals: (num_products,) tensor of product-level demand shifters.
        params: DGP parameter dict.
        num_draws: consumer draws for Monte Carlo integration.
        seed: RNG seed for consumer draws.
    """
    alpha_0 = params['alpha_0']
    alpha_1 = params.get('alpha_1', 0)
    alpha_2 = params.get('alpha_2', 0)
    gamma_0 = params['gamma_0']
    gamma_1 = params.get('gamma_1', 0)
    gamma_2 = params.get('gamma_2', 0)

    p = all_prices.clone().float()
    p[j] = float(price_j)

    torch.manual_seed(seed)
    log_income = torch.randn(num_draws)
    nu1 = torch.randn(num_draws)
    nu2 = torch.randn(num_draws)

    gamma_i = gamma_0 + gamma_1 * log_income + gamma_2 * nu1
    alpha_i = alpha_0 + alpha_1 * log_income + alpha_2 * nu2

    # u_{ij} = gamma_i + alpha_i * p_j + kxi_j   shape: (J, num_draws)
    u_ij = (gamma_i.unsqueeze(0)
            + alpha_i.unsqueeze(0) * p.unsqueeze(1)
            + kxi_vals.unsqueeze(1))
    exp_u = torch.exp(u_ij)
    denom = 1.0 + exp_u.sum(dim=0, keepdim=True)
    s_ij = exp_u / denom                        # (J, num_draws)

    s_j = s_ij[j].mean()
    ds_dp_j = (alpha_i * s_ij[j] * (1 - s_ij[j])).mean()

    return (ds_dp_j * float(price_j) / s_j).item()
