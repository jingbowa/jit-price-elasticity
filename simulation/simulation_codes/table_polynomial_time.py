import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time
import pandas as pd
from bnn_modules.bnn_numba import bnn_2scale

def true_function(X):
    return (
        3 * np.sin(X[:, 0] + 1) + 
        2 * np.exp(-0.5 * (X[:, 0]*X[:, 2]) **2) + 
        1.5 * X[:, 0] * X[:, 1] +
        np.cos(X[:, 2]) ** 2 +
        np.sum(X[:, 0:], axis=1)
    )

def generate_data(n_samples, n_features=3):
    X = np.random.uniform(-1, 1, (n_samples, n_features))
    y = true_function(X)
    return X, y

def main():
    sample_sizes = [160_000]
    feature_sizes = [4, 8, 12]
    degree = 5
    n_simulations = 10
    
    # Create lists to store results
    results = []
    
    for n_samples in sample_sizes:
        for n_features in feature_sizes:
            # Generate data
            X, y = generate_data(n_samples, n_features=n_features)
            X_test = np.zeros((1, n_features))
            
            # Warm-up runs
            poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=True)
            X_poly = poly.fit_transform(X)
            linear_model = LinearRegression()
            linear_model.fit(X_poly, y)
            _ = bnn_2scale(y, X, X_test[0], s1=30)
            
            # Run multiple simulations and store times
            poly_times = []
            bnn_times = []
            for _ in range(n_simulations):
                # Time polynomial regression
                start_time = time.time()
                poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=True)
                X_poly = poly.fit_transform(X)
                linear_model = LinearRegression()
                linear_model.fit(X_poly, y)
                poly_times.append(time.time() - start_time)
                
                # Time BNN
                start_time = time.time()
                _ = bnn_2scale(y, X, X_test[0], s1=30)
                bnn_times.append(time.time() - start_time)
            
            # Calculate average execution times
            avg_poly_time = np.mean(poly_times)
            avg_bnn_time = np.mean(bnn_times)
            
            # Store results
            results.append({
                'N_samples': n_samples,
                'N_features': n_features,
                'Poly_Time': f"{avg_poly_time:.4f}",
                'BNN_Time': f"{avg_bnn_time:.4f}"
            })
    
    # Create pandas DataFrame
    df = pd.DataFrame(results)
    # Create a more intuitive column structure
    result_tables = []
    for feature in feature_sizes:
        # Filter data for this feature size
        feature_data = df[df['N_features'] == feature]
        # Create columns for both Poly and BNN times
        pivot = feature_data.pivot(index='N_samples', 
                                 columns=['N_features'],
                                 values=['Poly_Time', 'BNN_Time'])
        # Rename columns to be more intuitive
        pivot.columns = [f"{feature}features_{col[0].split('_')[0]}" for col in pivot.columns]
        result_tables.append(pivot)
    
    # Combine all feature results
    df_wide = pd.concat(result_tables, axis=1)
    
    # Sort columns to group features in desired order
    desired_order = []
    for feature in feature_sizes:
        desired_order.extend([f"{feature}features_BNN", f"{feature}features_Poly"])
    df_wide = df_wide.reindex(columns=desired_order)
    
    df_wide.to_csv("./tables/table_app_polynomial_time_160k.csv", index=True)
    df_wide.to_latex("./tables/table_app_polynomial_time_160k.tex", index=True)
    
if __name__ == "__main__":
    main()