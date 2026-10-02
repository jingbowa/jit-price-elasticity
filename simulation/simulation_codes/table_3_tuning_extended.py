import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm
from bnn_modules.bnn_torch import p_elas_2scale_torch
from DGP_models.model_module_torch import (
    Logit_data, Logit_stats,
    BLP_ind_data, BLP_ind_stats,
    BLP_corr_data, BLP_corr_stats
)

import json

def monte_carlo(model_data, model_stats, num_obs, num_products, device):
    monte_size = 1000
    tuning_range = torch.arange(2, 60, device=device)
    mse_own = torch.zeros(len(tuning_range), device=device)
    mse_cross = torch.zeros(len(tuning_range), device=device)

    for _ in tqdm(range(monte_size), desc="Monte Carlo Iterations"):
        y1, y2, XZ, _, _ = model_data(num_obs=num_obs, num_products=num_products)
        true_own, true_cross = model_stats(num_obs=num_obs, num_products=num_products)
        
        true_own = torch.tensor(true_own, device=device, dtype=torch.float64)
        true_cross = torch.tensor(true_cross, device=device, dtype=torch.float64)
        
        y1_tensor = y1.to(device).double()
        y2_tensor = y2.to(device).double()
        XZ_tensor = XZ.to(device).double()
        xz_tensor = torch.tensor([0.5]*num_products + [0], device=device, dtype=torch.float64)

        for idx, tuning_s in enumerate(tuning_range):
            p_elas_own = p_elas_2scale_torch(y1_tensor, XZ_tensor, xz_tensor, 0, num_products, int(tuning_s), dtype=torch.float64)
            p_elas_cross = p_elas_2scale_torch(y2_tensor, XZ_tensor, xz_tensor, 0, num_products, int(tuning_s), dtype=torch.float64)
            
            mse_own[idx] += (p_elas_own - true_own).pow(2).item()
            mse_cross[idx] += (p_elas_cross - true_cross).pow(2).item()

    return mse_own.cpu().numpy(), mse_cross.cpu().numpy(), tuning_range.cpu().numpy()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_obs_list = [10000, 20000, 40000, 60000]
    num_products_list = [6, 10, 12, 14, 18]
    model_list = [
        ("Logit", Logit_data, Logit_stats),
        ("BLP_ind", BLP_ind_data, BLP_ind_stats),
        ("BLP_corr", BLP_corr_data, BLP_corr_stats)
    ]

    results = {}

    for model_name, model_data, model_stats in model_list:
        for num_obs in num_obs_list:
            for num_products in num_products_list:
                mse_own, mse_cross, tuning_range = monte_carlo(model_data, model_stats, num_obs, num_products, device)
                
                key = f"{model_name}|{num_obs}|{num_products}"
                results[key] = {
                    "mse_own": mse_own.tolist(),
                    "mse_cross": mse_cross.tolist(),
                    "tuning_range": tuning_range.tolist(),
                    "optimal_own": int(tuning_range[mse_own.argmin()]),
                    "optimal_cross": int(tuning_range[mse_cross.argmin()])
                }

    with open('./simulation_results/table_3_tuning_full_extended.json', 'w') as f:
        json.dump(results, f)

    print("Results saved to ./simulation_results/table_3_tuning_full_extended.json")

if __name__ == "__main__":
    main()