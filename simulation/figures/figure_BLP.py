import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import matplotlib.pyplot as plt
import numpy as np

from bnn_modules.bnn_torch import p_elas_2scale_torch, p_elas_2scale_torch_boot
from DGP_models.model_module_torch import BLP_corr_data, BLP_corr_stats

def compute_elasticities(
    model_data=BLP_corr_data, 
    model_stats=BLP_corr_stats, 
    monte=10, 
    bins=41, 
    tuning_s=7,
    price_range=(0.3, 0.7)
):
    # Pre-compute price points
    prices = torch.linspace(price_range[0], price_range[1], bins)
    
    # Initialize tensors with proper dimensions
    results = {
        'e_own': torch.empty((bins, monte)),
        'var_own': torch.empty((bins, monte)),
        'e_cross': torch.empty((bins, monte)),
        'var_cross': torch.empty((bins, monte)),
        't_own': torch.empty((bins, monte)),
        't_cross': torch.empty((bins, monte))
    }
    
    # Monte Carlo iterations
    for j in range(monte):
        y1, y2, XZ, _, _ = model_data(m_price=torch.tensor([0.5, 0.5, 0.5]))
        y1, y2, XZ = y1.double(), y2.double(), XZ.double()
        
        # Vectorize price evaluations
        for i, p_eval in enumerate(prices):
            print(f"Processing iteration {j+1}/{monte}, price point {i+1}/{bins}")
            xz = torch.tensor([p_eval.item(), 0.5, 0.5, 0.0], dtype=torch.float64)
            
            # Compute elasticities and variances
            results['e_own'][i, j] = p_elas_2scale_torch(y1, XZ, xz, 0, 3, tuning_s).item()
            _, results['var_own'][i, j] = p_elas_2scale_torch_boot(y1, XZ, xz, 0, 3, tuning_s)
            
            results['e_cross'][i, j] = p_elas_2scale_torch(y2, XZ, xz, 0, 3, tuning_s)
            _, results['var_cross'][i, j] = p_elas_2scale_torch_boot(y2, XZ, xz, 0, 3, tuning_s)
            
            results['t_own'][i, j], results['t_cross'][i, j] = model_stats(
                m_price=torch.tensor([p_eval.item(), 0.5, 0.5])
            )

    # Compute summary statistics
    processed_results = {
        'p': prices.tolist(),
        'e_own': torch.mean(results['e_own'], dim=1).tolist(),
        't_own': torch.mean(results['t_own'], dim=1).tolist(),
        'std_own': torch.sqrt(torch.mean(results['var_own'], dim=1)).tolist(),
        'e_cross': torch.mean(results['e_cross'], dim=1).tolist(),
        't_cross': torch.mean(results['t_cross'], dim=1).tolist(),
        'std_cross': torch.sqrt(torch.mean(results['var_cross'], dim=1)).tolist()
    }

    # Save results
    with open('./simulation_results/fig_elas_curve_BLP.json', 'w') as f:
        json.dump(processed_results, f)

    return processed_results

if __name__ == "__main__":
    compute_elasticities()   # writes the result file; figures/all_figures.py draws the figure
