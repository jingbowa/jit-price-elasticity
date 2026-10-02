import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyblp
import numpy as np
import pandas as pd
import torch
import time
from DGP_models.model_module_blp import BLP_corr

def BLP_problem(product_data, agent_data):
    # Create formulations
    X1_formulation = pyblp.Formulation('1 + prices')
    X2_formulation = pyblp.Formulation('1 + prices + high_prices')

    # Create problem
    problem = pyblp.Problem(
        product_formulations=(X1_formulation, X2_formulation),
        product_data=product_data,
        agent_formulation=pyblp.Formulation('1 + log_income + low + mid + high'),
        agent_data=agent_data,
        integration=pyblp.Integration('halton', size=250, specification_options={'seed': 0})
    )

    return problem

def generate_micro_moments(micro_statistics):
    micro_dataset = pyblp.MicroDataset(
        name="Income Survey", 
        observations=50_000,
        compute_weights=lambda t, p, a: np.ones((a.size, 1 + p.size)),
    )

    price_mid_part = pyblp.MicroPart(
        name="E[high_price_j * mid_i]",
        dataset=micro_dataset,
        compute_values=lambda t, p, a: np.outer(a.demographics[:, 3], np.r_[0, p.X2[:, 2]]),
    )
    price_high_part = pyblp.MicroPart(
        name="E[high_price_j * high_i]",
        dataset=micro_dataset,
        compute_values=lambda t, p, a: np.outer(a.demographics[:, 4], np.r_[0, p.X2[:, 2]]),
    )
    inside_mid_part = pyblp.MicroPart(
        name="E[1{j > 0} * mid_i]",
        dataset=micro_dataset,
        compute_values=lambda t, p, a: np.outer(a.demographics[:, 3], np.r_[0, p.X2[:, 0]]),
    )
    inside_high_part = pyblp.MicroPart(
        name="E[1{j > 0} * high_i]",
        dataset=micro_dataset,
        compute_values=lambda t, p, a: np.outer(a.demographics[:, 4], np.r_[0, p.X2[:, 0]]),
    )
    mid_part = pyblp.MicroPart(
        name="E[mid_i]",
        dataset=micro_dataset,
        compute_values=lambda t, p, a: np.outer(a.demographics[:, 3], np.r_[1, p.X2[:, 0]]),
    )
    high_part = pyblp.MicroPart(
        name="E[high_i]",
        dataset=micro_dataset,
        compute_values=lambda t, p, a: np.outer(a.demographics[:, 4], np.r_[1, p.X2[:, 0]]),
    )

    compute_ratio = lambda v: v[0] / v[1]
    compute_ratio_gradient = lambda v: [1 / v[1], -v[0] / v[1]**2]

    micro_moments = [
        pyblp.MicroMoment(
            name="E[high_price_j | mid_i]",
            value=micro_statistics.purchase_high_price_given_mid_income[0],
            parts=[price_mid_part, mid_part],
            compute_value=compute_ratio,
            compute_gradient=compute_ratio_gradient,
        ),
        pyblp.MicroMoment(
            name="E[high_price_j | high_i]",
            value=micro_statistics.purchase_high_price_given_high_income[0],
            parts=[price_high_part, high_part],
            compute_value=compute_ratio,
            compute_gradient=compute_ratio_gradient,
        ),
        pyblp.MicroMoment(
            name="E[1{j > 0} | mid_i]",
            value=micro_statistics.purchase_prob_given_mid_income[0],
            parts=[inside_mid_part, mid_part],
            compute_value=compute_ratio,
            compute_gradient=compute_ratio_gradient,
        ),
        pyblp.MicroMoment(
            name="E[1{j > 0} | high_i]",
            value=micro_statistics.purchase_prob_given_high_income[0],
            parts=[inside_high_part, high_part],
            compute_value=compute_ratio,
            compute_gradient=compute_ratio_gradient,
        )
    ]
    
    return micro_moments

def BLP_solve(problem, micro_moments):
    # Initial values
    initial_sigma = np.diag([0, 0.1, 0])
    initial_pi = np.array([[0, 0.1, 0, 0, 0], 
                          [0, 0.1, 0, 0, 0], 
                          [0, 0, 0, 0, 0]])
    sigma_ub = np.diag([0, 2, 0])
    sigma_lb = np.diag([0, -2, 0])
    pi_ub = np.array([[0, 2, 0, 0, 0], 
                      [0, 2, 0, 0, 0], 
                      [0, 0, 0, 0, 0]])
    pi_lb = np.array([[0, 0, 0, 0, 0], 
                      [0, 0, 0, 0, 0], 
                      [0, 0, 0, 0, 0]])

    # Solve the problem
    result = problem.solve(
        sigma=initial_sigma,
        pi=initial_pi,
        optimization=pyblp.Optimization('l-bfgs-b', {'gtol': 1e-8}),
        iteration=pyblp.Iteration('squarem', {'atol': 1e-8}),
        micro_moments=micro_moments,
        sigma_bounds=(sigma_lb, sigma_ub),
        pi_bounds=(pi_lb, pi_ub)
    )
    
    return result

# Main timing analysis
def run_timing_analysis():
    num_products = 4
    num_draws = 250
    market_sizes = [5_000, 20_000, 80_000]
    num_simulations = 2
    
    all_results = []
    
    for num_markets in market_sizes:
        print(f"\nAnalyzing market size: {num_markets}")
        
        time_per_num_market = []
        for sim in range(num_simulations):
            print(f"Running simulation {sim + 1} of {num_simulations}")
            
            # Set seeds
            np.random.seed(12345 + sim)
            torch.manual_seed(12345 + sim)
            
            # Simulate data
            s_jt, p_jt, z_jt, inc, product_data, agent_data, micro_statistics = BLP_corr(num_markets, num_products, num_draws)
            # Generate micro-moments
            micro_moments = generate_micro_moments(micro_statistics)
                        
            # Start timer
            start_time = time.time()
            
            # Estimate
            problem = BLP_problem(product_data, agent_data)
            result = BLP_solve(problem, micro_moments)
            
            # Record time
            elapsed_time = time.time() - start_time
            time_per_num_market.append(elapsed_time)
        
        all_results.append({
            'num_markets': num_markets,
            'time': time_per_num_market
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results as json
    results_df.to_json("./simulation_results/table_appendix_pyblp.json")
    
    return results_df

if __name__ == "__main__":
    run_timing_analysis()
