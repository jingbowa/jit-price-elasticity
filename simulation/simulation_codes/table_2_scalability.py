import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from tqdm import tqdm
import json
import numpy as np

from DGP_models.model_module_torch import BLP_corr_data

from bnn_modules.bnn_numba import p_elas as p_elas_numba, p_elas_boot as p_elas_boot_numba
from bnn_modules.bnn_torch import p_elas_2scale_torch, p_elas_2scale_torch_boot

def table_2():
    num_products = 4
    tuning_s = 7
    monte_size = 10
    sample_sizes = [5000, 20000, 80000, 320000, 1280000]
    boot_size = 100
    
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for num_obs in sample_sizes:
        print(f"Processing sample size: {num_obs}")
        
        # Initialize timing arrays
        time_numba_point = np.zeros(monte_size)
        time_numba_boot = np.zeros(monte_size)
        time_torch_cpu_point = np.zeros(monte_size)
        time_torch_cpu_boot = np.zeros(monte_size)
        time_torch_gpu_point = np.zeros(monte_size)
        time_torch_gpu_boot = np.zeros(monte_size)
        
        # Generate data
        y1, _, XZ, p1, z = BLP_corr_data(num_obs=num_obs, num_products=num_products)
    
        # Prepare all data versions upfront
        # Numba (numpy arrays)
        y1_numba = y1.numpy().astype(np.float64)
        XZ_numba = XZ.numpy().astype(np.float64)
        xz_numba = np.array([0.5]*num_products + [0], dtype=np.float64)
        p1_numba = p1.numpy().astype(np.float64)
        z_numba = z.numpy().astype(np.float64)
        
        # PyTorch CPU (double precision)
        y1_cpu = y1.double()
        XZ_cpu = XZ.double()
        xz_cpu = torch.tensor([0.5]*num_products + [0], dtype=torch.float64)
        
        # PyTorch GPU
        y1_gpu = y1.double().to(device)
        XZ_gpu = XZ.double().to(device)
        xz_gpu = torch.tensor([0.5]*num_products + [0], dtype=torch.float64).to(device)
        
        # Initial compilation/warmup
        _ = p_elas_numba(y1_numba, XZ_numba, xz_numba, tuning_s, p1_numba, z_numba)
        _ = p_elas_boot_numba(y1_numba, XZ_numba, xz_numba, tuning_s, p1_numba, z_numba, bsize=boot_size)
        _ = p_elas_2scale_torch(y1_cpu, XZ_cpu, xz_cpu, 0, num_products, tuning_s)
        _ = p_elas_2scale_torch_boot(y1_cpu, XZ_cpu, xz_cpu, 0, num_products, tuning_s, boot_size=boot_size)
        _ = p_elas_2scale_torch(y1_gpu, XZ_gpu, xz_gpu, 0, num_products, tuning_s)
        _ = p_elas_2scale_torch_boot(y1_gpu, XZ_gpu, xz_gpu, 0, num_products, tuning_s, boot_size=boot_size)
        
        for i in tqdm(range(monte_size)):
            # Numba point estimate
            start_time = time.time()
            _ = p_elas_numba(y1_numba, XZ_numba, xz_numba, tuning_s, p1_numba, z_numba)
            time_numba_point[i] = time.time() - start_time

            # Numba bootstrap
            start_time = time.time()
            _ = p_elas_boot_numba(y1_numba, XZ_numba, xz_numba, tuning_s, p1_numba, z_numba, bsize=boot_size)
            time_numba_boot[i] = time.time() - start_time
            
            # PyTorch CPU point estimate
            start_time = time.time()
            _ = p_elas_2scale_torch(y1_cpu, XZ_cpu, xz_cpu, 0, num_products, tuning_s)
            time_torch_cpu_point[i] = time.time() - start_time
            
            # PyTorch CPU bootstrap
            start_time = time.time()
            _, _ = p_elas_2scale_torch_boot(y1_cpu, XZ_cpu, xz_cpu, 0, num_products, tuning_s, boot_size=boot_size)
            time_torch_cpu_boot[i] = time.time() - start_time
            
            # PyTorch GPU point estimate
            start_time = time.time()
            _ = p_elas_2scale_torch(y1_gpu, XZ_gpu, xz_gpu, 0, num_products, tuning_s)
            time_torch_gpu_point[i] = time.time() - start_time
            
            # PyTorch GPU bootstrap
            start_time = time.time()
            _, _ = p_elas_2scale_torch_boot(y1_gpu, XZ_gpu, xz_gpu, 0, num_products, tuning_s, boot_size=boot_size)
            time_torch_gpu_boot[i] = time.time() - start_time
        
        results.append({
            'Sample Size': num_obs,
            'Numba Runtime (Point Estimate)': np.mean(time_numba_point),
            'Numba Runtime (100 Bootstrap)': np.mean(time_numba_boot),
            'PyTorch CPU Runtime (Point Estimate)': np.mean(time_torch_cpu_point),
            'PyTorch CPU Runtime (100 Bootstrap)': np.mean(time_torch_cpu_boot),
            'PyTorch GPU Runtime (Point Estimate)': np.mean(time_torch_gpu_point),
            'PyTorch GPU Runtime (100 Bootstrap)': np.mean(time_torch_gpu_boot)
        })
    
    # Save results as JSON
    with open("./simulation_results/table_2_scalability.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to ./simulation_results/table_2_scalability.json")
    
    return results

if __name__ == "__main__":
    table_2()