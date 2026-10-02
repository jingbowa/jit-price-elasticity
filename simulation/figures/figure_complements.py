import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib.pyplot as plt

from bnn_modules.bnn_numba import p_elas, p_elas_boot
from DGP_models.model_module_extra import Complements_data, Complements_stats

def compute_elasticities(
    model_data=Complements_data,
    model_stats=Complements_stats,
    monte=10,
    bins=41,
    tuning_s=7,
    price_range=(0.3, 0.7)
):
    # Pre-compute price points
    p = np.linspace(price_range[0], price_range[1], bins)
    
    # Initialize arrays
    results = {
        'e_own': np.zeros((bins, monte)),
        'var_own': np.zeros((bins, monte)),
        'e_cross': np.zeros((bins, monte)),
        'var_cross': np.zeros((bins, monte)),
        't_own': np.zeros((bins, monte)),
        't_cross': np.zeros((bins, monte))
    }
    
    # Monte Carlo iterations
    for j in range(monte):
        y1, y2, XZ, p1, z = model_data()
        
        for i, p_eval in enumerate(p):
            print(f"Processing iteration {j+1}/{monte}, price point {i+1}/{bins}")
            xz = np.array([p_eval, 0.5, 0.5, 0])
            
            # Compute elasticities and variances
            results['e_own'][i, j] = p_elas(y1, XZ, xz, tuning_s, p1, z)
            _, results['var_own'][i, j] = p_elas_boot(y1, XZ, xz, tuning_s, p1, z)
            
            results['e_cross'][i, j] = p_elas(y2, XZ, xz, tuning_s, p1, z)
            _, results['var_cross'][i, j] = p_elas_boot(y2, XZ, xz, tuning_s, p1, z)
            
            results['t_own'][i, j], results['t_cross'][i, j] = model_stats(np.array([p_eval, 0.5, 0.5]))

    # Compute summary statistics
    processed_results = {
        'p': p.tolist(),
        'e_own': np.mean(results['e_own'], axis=1).tolist(),
        't_own': np.mean(results['t_own'], axis=1).tolist(),
        'std_own': np.sqrt(np.mean(results['var_own'], axis=1)).tolist(),
        'e_cross': np.mean(results['e_cross'], axis=1).tolist(),
        't_cross': np.mean(results['t_cross'], axis=1).tolist(),
        'std_cross': np.sqrt(np.mean(results['var_cross'], axis=1)).tolist()
    }

    # Save results
    with open('./simulation_results/fig_elas_curve_complements.json', 'w') as f:
        json.dump(processed_results, f)

    return processed_results

if __name__ == "__main__":
    compute_elasticities()   # writes the result file; figures/all_figures.py draws the figure
