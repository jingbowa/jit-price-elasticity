import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm
import json

from bnn_modules.bnn_torch import p_elas_2scale_torch, p_elas_2scale_torch_boot
from DGP_models.model_module_extra import Complements_data, Complements_stats
from DGP_models.model_module_extra import Variety_data, Variety_stats  # slow: ~12 hours (see README runtime notes)

def monte_carlo(model_data, model_stats, num_products, tuning_s):
    monte_size = 50
    
    results = []
    
    xz = np.array([0.5]*num_products + [0])
    
    for i in tqdm(range(monte_size)):
        y1, y2, XZ, p1, z = model_data(np.array([0.5, 0.5, 0.5]))
        true_own, true_cross = model_stats(np.array([0.5, 0.5, 0.5]))
        
        # Convert to PyTorch tensors
        y1_tensor = torch.from_numpy(y1).double()
        y2_tensor = torch.from_numpy(y2).double()
        XZ_tensor = torch.from_numpy(XZ).double()
        xz_tensor = torch.from_numpy(xz).double()
        
        est_own = p_elas_2scale_torch(y1_tensor, XZ_tensor, xz_tensor, 0, num_products, tuning_s).item()
        _, var_own = p_elas_2scale_torch_boot(y1_tensor, XZ_tensor, xz_tensor, 0, num_products, tuning_s)
        se_own = torch.sqrt(var_own).item()
        
        est_cross = p_elas_2scale_torch(y2_tensor, XZ_tensor, xz_tensor, 0, num_products, tuning_s).item()
        _, var_cross = p_elas_2scale_torch_boot(y2_tensor, XZ_tensor, xz_tensor, 0, num_products, tuning_s)
        se_cross = torch.sqrt(var_cross).item()

        results.append({
            'iteration': i,
            'true_own': true_own,
            'true_cross': true_cross,
            'est_own': est_own,
            'est_cross': est_cross,
            'se_own': se_own,
            'se_cross': se_cross
        })
    
    return results

def run_simulations():
    num_products, tuning_s = 3, 7
    
    model_list = [
        ("Complements", Complements_data, Complements_stats),
        ("Variety", Variety_data, Variety_stats),
    ]
    
    all_results = {}
    for model_name, model_data, model_stats in model_list:
        print(f"Running simulations for {model_name} model...")
        results = monte_carlo(model_data, model_stats, num_products, tuning_s)
        all_results[model_name] = results
        print(f"Simulations for {model_name} completed.")
    
    # Save all results as JSON
    with open("./simulation_results/table_4_extra_models.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("All results saved to ./simulation_results/table_4_extra_models.json")
    
    return all_results

if __name__ == "__main__":
    run_simulations()