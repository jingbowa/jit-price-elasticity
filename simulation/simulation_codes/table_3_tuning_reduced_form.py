import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm
from bnn_modules.bnn_torch import p_elas_2scale_torch
from DGP_models.model_module_rf import (
    RF_linear_data, RF_linear_stats,
    RF_quadratic_data, RF_quadratic_stats
)

import json

def monte_carlo(model_data, model_stats, num_obs, num_products, device):
    monte_size = 1000
    tuning_range = torch.arange(2, 30, device=device)
    mse_own = torch.zeros(len(tuning_range), device=device)

    for _ in tqdm(range(monte_size), desc="Monte Carlo Iterations"):
        y, XZ, _, _ = model_data(num_obs=num_obs, num_products=num_products)
        true_own = model_stats(num_obs=num_obs, num_products=num_products)
        
        true_own = torch.tensor(true_own, device=device, dtype=torch.float64)
        
        y_tensor = y.to(device).double()
        XZ_tensor = XZ.to(device).double()
        xz_tensor = torch.tensor([0.5]*num_products + [0], device=device, dtype=torch.float64)

        for idx, tuning_s in enumerate(tuning_range):
            p_elas_own = p_elas_2scale_torch(y_tensor, XZ_tensor, xz_tensor, 0, num_products, int(tuning_s), dtype=torch.float64)
            
            mse_own[idx] += (p_elas_own - true_own).pow(2).item()

    return mse_own.cpu().numpy(), tuning_range.cpu().numpy()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_obs_list = [10000, 20000, 40000, 60000]
    num_products_list = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    model_list = [
        ("RF_linear", RF_linear_data, RF_linear_stats),
        ("RF_quadratic", RF_quadratic_data, RF_quadratic_stats)
    ]

    results = {}

    for model_name, model_data, model_stats in model_list:
        for num_obs in num_obs_list:
            for num_products in num_products_list:
                mse_own, tuning_range = monte_carlo(model_data, model_stats, num_obs, num_products, device)
                
                key = f"{model_name}|{num_obs}|{num_products}"
                results[key] = {
                    "mse_own": mse_own.tolist(),
                    "tuning_range": tuning_range.tolist(),
                    "optimal_own": int(tuning_range[mse_own.argmin()]),
                }

    with open('./simulation_results/table_3_tuning_reduced_form.json', 'w') as f:
        json.dump(results, f)

    print("Results saved to ./simulation_results/table_3_tuning_reduced_form.json")

if __name__ == "__main__":
    main()