import os
import sys
# Go up 3 levels: file → elasticity_estimation → code_latest → nyc_taxi
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_dir)
sys.path.append(project_dir)

import pandas as pd
import numpy as np
import torch
import time
from bnn_modules.bnn_torch import p_elas_2scale_torch, p_elas_2scale_torch_boot

# Import configuration
from code_latest.step3_elasticity_estimation.config_quality_filters import (
    FEATURE_COLUMNS, AGG_DIR, DERIVED_DIR, ZONE_ELAS_DIR,
    ZONE_TUNING_PARAMETER
)

# Quality filters applied directly in this script
MIN_ZONE_SAMPLE = 50000        # Minimum observations needed for stable estimates
MAX_ZERO_DEMAND_PCT = 0.65     # Maximum percentage of markets with zero rides (65% is the threshold for quality filtering)

# Confidence interval computation (set to True to compute bootstrap CIs)
COMPUTE_CI = False  # Set to True to compute bootstrap confidence intervals

# Bootstrap parameters (only used when COMPUTE_CI = True)
N_BOOTSTRAP = 100
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def load_and_prepare_data(months):
    """Load and prepare market data for analysis."""
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    all_data = []
    for month in months:
        year, month_num = month.split('-')
        file_path = f"{AGG_DIR}step2_markets_{year}-{month_num}.parquet"
        print(f"Loading {file_path}...")
        data = pd.read_parquet(file_path)
        data['month'] = month
        all_data.append(data)
    
    market_data = pd.concat(all_data, ignore_index=True)
    print(f"Total records: {len(market_data):,}")
    
    # Add zone coordinates
    zone_coords = pd.read_csv(f"{DERIVED_DIR}step1_zone_coordinates.csv")
    
    market_data = pd.merge(
        market_data,
        zone_coords[['LocationID', 'latitude', 'longitude']].rename(
            columns={'latitude': 'PU_latitude', 'longitude': 'PU_longitude'}
        ),
        left_on='PULocationID', right_on='LocationID', how='inner'
    ).drop('LocationID', axis=1)
    
    market_data = pd.merge(
        market_data,
        zone_coords[['LocationID', 'latitude', 'longitude']].rename(
            columns={'latitude': 'DO_latitude', 'longitude': 'DO_longitude'}
        ),
        left_on='DOLocationID', right_on='LocationID', how='inner'
    ).drop('LocationID', axis=1)
    
    # Create time features
    market_data['pickup_date'] = pd.to_datetime(market_data['pickup_date'])
    market_data['start_hour'] = market_data['time_window'].str.split('-').str[0].astype(int)
    market_data['hour_of_day'] = market_data['start_hour']
    start_date = market_data['pickup_date'].min()
    market_data['days_since_start'] = (market_data['pickup_date'] - start_date).dt.days
    
    # Create aggregated price features
    market_data['route_id'] = market_data['PULocationID'].astype(str) + '-' + market_data['DOLocationID'].astype(str)
    
    aggregations = {
        ('PULocationID', 'hour_of_day'): 'trip_price_pu_hour_avg',
        ('DOLocationID', 'hour_of_day'): 'trip_price_do_hour_avg',
        ('route_id', 'hour_of_day'): 'trip_price_route_hour_avg',
        ('PULocationID',): 'trip_price_pu_avg',
        ('DOLocationID',): 'trip_price_do_avg',
        ('route_id',): 'trip_price_route_avg',
        ('hour_of_day',): 'trip_price_hour_avg',
        ('pickup_date',): 'trip_price_date_avg',
    }
    
    for group_cols, col_name in aggregations.items():
        market_data[col_name] = market_data.groupby(list(group_cols))['trip_price'].transform('mean')
    
    return market_data

def prepare_tensors(market_data, feature_columns):
    """Convert data to PyTorch tensors and normalize all features."""
    check_columns = feature_columns + ['trip_price_iv', 'ride_count']
    market_data_filtered = market_data.dropna(subset=check_columns)
    print(f"Using {len(market_data_filtered):,} complete records")
    
    X_raw = torch.tensor(market_data_filtered[feature_columns].values, dtype=torch.float64, device=device)
    
    # Normalize X to [0,1] for all features (including trip_price)
    X_mins = X_raw.min(dim=0).values
    X_maxs = X_raw.max(dim=0).values
    denom_X = X_maxs - X_mins
    X_norm = torch.where(denom_X == 0, torch.zeros_like(X_raw), (X_raw - X_mins) / denom_X)
    
    # Store price normalization parameters for elasticity correction
    p_idx = feature_columns.index('trip_price')
    price_min = X_mins[p_idx].item()
    price_max = X_maxs[p_idx].item()
    
    # Normalize Z (instrument) to [0,1]
    Z = torch.tensor(market_data_filtered['trip_price_iv'].values.reshape(-1, 1), dtype=torch.float64, device=device)
    z_min, z_max = Z.min(), Z.max()
    denom_Z = z_max - z_min
    Z_norm = torch.zeros_like(Z) if denom_Z.item() == 0 else (Z - z_min) / denom_Z
    
    # Target variable
    y = torch.tensor(market_data_filtered['ride_count'].values, dtype=torch.float64, device=device)
    
    # Combine features and instrument
    XZ = torch.cat([X_norm, Z_norm], dim=1)
    
    return XZ, y, market_data_filtered, price_min, price_max

def rescale_elasticity(elas_normalized, p_norm, price_min, price_max):
    """Convert elasticity from normalized [0,1] price scale to original price scale."""
    if p_norm <= 0:
        return elas_normalized
    
    price_range = price_max - price_min
    p_original = price_min + p_norm * price_range
    scale_adjustment = p_original / (p_norm * price_range)
    
    return elas_normalized * scale_adjustment

def estimate_zone_elasticities(market_data_original, market_data_for_tensors, 
                                feature_columns, XZ_full, y_full, price_min, price_max):
    """Estimate elasticity for each zone (all boroughs) that passes quality filters."""
    print("\n" + "="*70)
    print("ESTIMATING ZONE-SPECIFIC ELASTICITIES")
    print("="*70)
    
    if COMPUTE_CI:
        print(f"Bootstrap CI: Enabled (N={N_BOOTSTRAP}, {int(CONFIDENCE_LEVEL*100)}% CI)")
    else:
        print(f"Bootstrap CI: Disabled (point estimates only)")
    
    print(f"Quality Filters:")
    print(f"  - Minimum Zone Sample: {MIN_ZONE_SAMPLE:,}")
    print(f"  - Maximum Zero-Demand: {MAX_ZERO_DEMAND_PCT*100:.0f}%")
    print(f"Price range: ${price_min:.2f} to ${price_max:.2f}")
    
    # Get all zones in the data (no borough filter - estimate for all zones)
    zones_in_data = market_data_for_tensors['PULocationID'].dropna().unique()
    processable_zones = sorted(zones_in_data)
    
    print(f"\nChecking quality criteria for {len(processable_zones)} zones (all boroughs)...")
    
    # Check quality for all zones using direct filters (for informational purposes)
    passed_zones = []
    failed_zones = []
    
    for zone_id in processable_zones:
        zone_data = market_data_original[market_data_original['PULocationID'] == zone_id]
        
        # Filter 1: Sample size
        if len(zone_data) < MIN_ZONE_SAMPLE:
            failed_zones.append((zone_id, f"Insufficient sample size: {len(zone_data)}"))
            continue
        
        # Filter 2: Zero demand percentage
        zero_pct = (zone_data['ride_count'] == 0).mean()
        if zero_pct > MAX_ZERO_DEMAND_PCT:
            failed_zones.append((zone_id, f"Too many zero-demand markets ({zero_pct*100:.1f}%)"))
            continue
        
        passed_zones.append(zone_id)
    
    print(f"\n{len(passed_zones)} zones passed quality checks")
    print(f"{len(failed_zones)} zones would be excluded by filters")
    print(f"Estimating for ALL {len(processable_zones)} zones regardless of quality filters")
    
    # Estimate elasticity for ALL zones
    print("\nEstimating elasticities (in original price units)...")
    print("-" * 70)
    
    p_col_idx = feature_columns.index('trip_price')
    z_col_idx = len(feature_columns)  # Z is last column
    
    zone_elasticities = []
    
    # Set random seed for reproducibility
    if COMPUTE_CI:
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
    
    # Variables for timing measurement
    estimation_times = []
    
    for i, zone_id in enumerate(processable_zones):
        zone_data_rows = market_data_original[market_data_original['PULocationID'] == zone_id]
        
        lat = zone_data_rows['PU_latitude'].mean()
        lng = zone_data_rows['PU_longitude'].mean()
        original_sample_size = len(zone_data_rows)
        
        # Get zone-specific mean features
        zone_mask = torch.tensor(
            (market_data_for_tensors['PULocationID'] == zone_id).values.astype(bool),
            dtype=torch.bool, device=XZ_full.device
        )
        
        XZ_zone = XZ_full[zone_mask]
        zone_xz_mean = XZ_zone.mean(dim=0)
        tensor_sample_size = XZ_zone.shape[0]
        p_norm = zone_xz_mean[p_col_idx].item()
        
        # Time the estimation
        start_time = time.time()
        
        if COMPUTE_CI:
            # Use bootstrap function to get mean and variance
            elas_mean_normalized, elas_var_normalized = p_elas_2scale_torch_boot(
                y_full, XZ_full, zone_xz_mean, 
                p_col_idx, z_col_idx, ZONE_TUNING_PARAMETER,
                boot_size=N_BOOTSTRAP, dtype=torch.float64
            )
            
            # Convert to original scale
            elas_mean_normalized = elas_mean_normalized.item()
            elas_var_normalized = elas_var_normalized.item()
            
            # Rescale mean and standard error
            elas_original = rescale_elasticity(elas_mean_normalized, p_norm, price_min, price_max)
            price_range = price_max - price_min
            if p_norm > 0 and price_range > 0:
                scale_adjustment = (price_min + p_norm * price_range) / (p_norm * price_range)
            else:
                scale_adjustment = 1.0
            std_error = np.sqrt(elas_var_normalized) * abs(scale_adjustment)
            
            # Calculate confidence intervals
            z_score = 1.96  # For 95% confidence
            ci_lower = elas_original - z_score * std_error
            ci_upper = elas_original + z_score * std_error
        else:
            # Use faster point estimate without bootstrap
            elas_normalized = p_elas_2scale_torch(
                y_full, XZ_full, zone_xz_mean, 
                p_col_idx, z_col_idx, ZONE_TUNING_PARAMETER,
                dtype=torch.float64
            )
            elas_original = rescale_elasticity(elas_normalized.item(), p_norm, price_min, price_max)
            ci_lower = np.nan
            ci_upper = np.nan
            std_error = np.nan
        
        elapsed_time = time.time() - start_time
        estimation_times.append(elapsed_time)
        
        # Print progress every 10 zones
        if (i + 1) % 10 == 0 or (i + 1) == len(processable_zones):
            avg_time = np.mean(estimation_times)
            if COMPUTE_CI:
                print(f"Progress: {i+1}/{len(processable_zones)} zones | Avg time: {avg_time:.3f}s per zone (with bootstrap)")
            else:
                print(f"Progress: {i+1}/{len(processable_zones)} zones | Avg time: {avg_time:.3f}s per zone (point estimate)")
        
        zone_elasticities.append({
            'zone_id': zone_id,
            'elasticity': elas_original,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std_error': std_error,
            'latitude': lat,
            'longitude': lng,
            'sample_size': original_sample_size,
            'tensor_sample_size': tensor_sample_size,
            'computation_time_seconds': elapsed_time
        })
    
    elas_df = pd.DataFrame(zone_elasticities)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total zones with elasticity estimates: {len(elas_df)}")
    print(f"  Zones passing quality filters: {len(passed_zones)}")
    print(f"  Zones with quality concerns: {len(failed_zones)}")
    
    # Print timing statistics
    if len(estimation_times) > 0:
        print(f"\nTiming Statistics:")
        print(f"  Average time per zone: {np.mean(estimation_times):.3f} seconds")
        print(f"  Median time per zone: {np.median(estimation_times):.3f} seconds")
        print(f"  Min time: {np.min(estimation_times):.3f} seconds")
        print(f"  Max time: {np.max(estimation_times):.3f} seconds")
        print(f"  Total estimation time: {np.sum(estimation_times):.1f} seconds ({np.sum(estimation_times)/60:.1f} minutes)")
        if COMPUTE_CI:
            print(f"  Bootstrap iterations: {N_BOOTSTRAP}")
    
    if len(elas_df) > 0:
        # Filter to quality-passing zones only for statistics
        elas_df_quality = elas_df[elas_df['zone_id'].isin(passed_zones)]
        
        print(f"\nElasticity statistics (quality-passing zones only, n={len(elas_df_quality)}):")
        print(elas_df_quality['elasticity'].describe())
        
        print(f"\nDetailed quantiles (quality-passing zones only):")
        quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        for q in quantiles:
            val = elas_df_quality['elasticity'].quantile(q)
            print(f"  {q*100:5.1f}th percentile: {val:7.4f}")
    
    return elas_df

if __name__ == "__main__":
    MONTHS = ["2024-01", "2024-02"]
    
    # Load and prepare data
    market_data = load_and_prepare_data(MONTHS)
    
    # Prepare tensors
    XZ, y, market_data_filtered, price_min, price_max = prepare_tensors(market_data, FEATURE_COLUMNS)
    
    # Estimate zone-specific elasticities
    zone_elasticities_df = estimate_zone_elasticities(
        market_data, market_data_filtered, 
        FEATURE_COLUMNS, XZ, y, price_min, price_max
    )
    
    # Save results
    os.makedirs(ZONE_ELAS_DIR, exist_ok=True)
    
    output_path = f"{ZONE_ELAS_DIR}step3_elasticity_zones_all_nyc.csv"
    zone_elasticities_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

