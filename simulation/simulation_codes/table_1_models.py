import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm
import json

from bnn_modules.bnn_torch import p_elas_2scale_torch, p_elas_2scale_torch_boot
from DGP_models.model_module_torch import Logit_data, Logit_stats
from DGP_models.model_module_torch import BLP_ind_data, BLP_ind_stats
from DGP_models.model_module_torch import BLP_corr_data, BLP_corr_stats

def monte_carlo(model_data, model_stats, num_obs, num_products, tuning_s):
    monte_size = 50
    
    results = []
    
    for i in tqdm(range(monte_size)):
        y1, y2, XZ, p1, z = model_data(num_obs=num_obs, num_products=num_products)
        true_own, true_cross = model_stats(num_obs=num_obs, num_products=num_products)
        
        # Ensure all tensors are float64
        y1_tensor = y1.double()
        y2_tensor = y2.double()
        XZ_tensor = XZ.double()
        xz_tensor = torch.tensor([0.5]*num_products + [0], dtype=torch.float64)
        
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
    num_obs, num_products, tuning_s = 20000, 4, 7
    
    model_list = [
        ("Logit", Logit_data, Logit_stats),
        ("BLP_ind", BLP_ind_data, BLP_ind_stats),
        ("BLP_corr", BLP_corr_data, BLP_corr_stats)
    ]
    
    all_results = {}
    for model_name, model_data, model_stats in model_list:
        print(f"Running simulations for {model_name} model...")
        results = monte_carlo(model_data, model_stats, num_obs, num_products, tuning_s)
        all_results[model_name] = results
        print(f"Simulations for {model_name} completed.")
    
    # Save all results as JSON
    with open("./simulation_results/table_1_models.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("All results saved to ./simulation_results/table_1_models.json")
    
    return all_results

if __name__ == "__main__":
    run_simulations()