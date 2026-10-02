import os
import sys
# Go up 3 levels: file → elasticity_estimation → code_latest → nyc_taxi
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_dir)
sys.path.append(project_dir)

import pandas as pd
import numpy as np
import torch
from bnn_modules.bnn_torch import p_elas_2scale_torch, p_elas_2scale_torch_boot

# Import configuration
from code_latest.step3_elasticity_estimation.config_quality_filters import (
    FEATURE_COLUMNS as BASE_FEATURE_COLUMNS, AGG_DIR, DERIVED_DIR, ZONE_ELAS_DIR as ELASTICITY_OUTPUT_DIR,
    ZONE_TUNING_PARAMETER, ANALYZE_BOROUGHS
)

# Add hour_of_day to features for temporal analysis
# This is CRITICAL - without it, the model cannot distinguish between time periods
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ['hour_of_day']
print(f"Using {len(FEATURE_COLUMNS)} features (including hour_of_day for temporal analysis)")

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Confidence interval computation (off by default for speed)
COMPUTE_CI = True  # Set to True to compute bootstrap confidence intervals

# Bootstrap parameters (only used when COMPUTE_CI = True)
N_BOOTSTRAP = 100
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

# Time period definitions for analysis
# 3-hour blocks starting from hour 5
TIME_PERIODS = {
    '05:00-07:59': [5, 6, 7],      # Early morning
    '08:00-10:59': [8, 9, 10],     # Morning commute/business start
    '11:00-13:59': [11, 12, 13],   # Midday
    '14:00-16:59': [14, 15, 16],   # Afternoon
    '17:00-19:59': [17, 18, 19],   # Evening commute
    '20:00-22:59': [20, 21, 22],   # Evening/night
    '23:00-01:59': [23, 0, 1],     # Late night
    '02:00-04:59': [2, 3, 4]       # Very late night/early morning
}

def select_quality_zones(market_data):
    """
    Select zones for temporal analysis by borough (no quality filtering).
    """
    print("\n" + "="*70)
    print("ZONE SELECTION FOR TEMPORAL ANALYSIS")
    print("="*70)
    
    # Load zone lookup to filter by borough
    zone_lookup = pd.read_csv('./data/taxi_zone_lookup.csv')
    core_zones_ids = set(zone_lookup[zone_lookup['Borough'].isin(ANALYZE_BOROUGHS)]['LocationID'].tolist())
    
    zones_in_data = market_data['PULocationID'].dropna().unique()
    selected_zones = [z for z in zones_in_data if z in core_zones_ids]
    
    print(f"\nSelected {len(selected_zones)} zones in {', '.join(ANALYZE_BOROUGHS)}")
    print(f"Research question: Does elasticity vary by time of day?")
    
    return selected_zones

def load_and_prepare_data(months):
    """Load and prepare market data for temporal analysis."""
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
    print(f"Total records (all NYC): {len(market_data):,}")
    
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
    """Convert data to PyTorch tensors and normalize."""
    check_columns = feature_columns + ['trip_price_iv', 'ride_count']
    market_data_filtered = market_data.dropna(subset=check_columns)
    print(f"Using {len(market_data_filtered):,} complete records")
    
    X = torch.tensor(market_data_filtered[feature_columns].values, dtype=torch.float64, device=device)
    
    # Normalize X to [0,1] and store normalization parameters
    X_mins = X.min(dim=0).values
    X_maxs = X.max(dim=0).values
    denom_X = X_maxs - X_mins
    X = torch.where(denom_X == 0, torch.zeros_like(X), (X - X_mins) / denom_X)
    
    # Normalize Z (instrument) to [0,1]
    Z = torch.tensor(market_data_filtered['trip_price_iv'].values.reshape(-1, 1), dtype=torch.float64, device=device)
    z_min, z_max = Z.min(), Z.max()
    denom_Z = z_max - z_min
    Z_norm = torch.zeros_like(Z) if denom_Z.item() == 0 else (Z - z_min) / denom_Z
    
    # Target variable
    y = torch.tensor(market_data_filtered['ride_count'].values, dtype=torch.float64, device=device)
    
    # Combine features and instrument
    XZ = torch.cat([X, Z_norm], dim=1)
    
    # Return normalization parameters for later use
    norm_params = {'X_mins': X_mins, 'X_maxs': X_maxs, 'denom_X': denom_X}
    
    return XZ, y, market_data_filtered, norm_params

def estimate_time_period_subgroups(market_data_original, market_data_for_tensors, 
                                   feature_columns, XZ_full, y_full, time_periods_dict, norm_params):
    """
    Estimate elasticity with optional confidence intervals for time period subgroups.
    
    Method: Use ALL zone data for estimation, evaluate at each subgroup's mean features.
    Bootstrap confidence intervals computed for each period (if COMPUTE_CI = True).
    
    Args:
        time_periods_dict: Dictionary mapping period names to lists of hours
    """
    # Streamlined header
    if COMPUTE_CI:
        print(f"Estimating {len(time_periods_dict)} periods (boot={N_BOOTSTRAP}, CI={int(CONFIDENCE_LEVEL*100)}%)")
    else:
        print(f"Estimating {len(time_periods_dict)} periods (no CI)")
    
    p_col_idx = feature_columns.index('trip_price')
    z_col_idx = len(feature_columns)
    # Retrieve price normalization bounds
    price_min = norm_params['X_mins'][p_col_idx].item()
    price_max = norm_params['X_maxs'][p_col_idx].item()
    
    period_elasticities = []
    
    for i, (period_name, hours) in enumerate(time_periods_dict.items()):
        
        # Set random seed for this period (if using bootstrap)
        if COMPUTE_CI:
            np.random.seed(RANDOM_SEED + i)
            torch.manual_seed(RANDOM_SEED + i)
        
        # Get observations from this period
        period_mask = torch.tensor(
            market_data_for_tensors['hour_of_day'].isin(hours).values.astype(bool),
            dtype=torch.bool, device=XZ_full.device
        )
        
        period_obs_count = period_mask.sum().item()
        if period_obs_count == 0:
            continue
        
        XZ_period = XZ_full[period_mask]
        period_xz_mean = XZ_period.mean(dim=0)
        
        if COMPUTE_CI:
            # Use bootstrap function to get mean and variance
            elas_mean, elas_var = p_elas_2scale_torch_boot(
                y_full, XZ_full, period_xz_mean, 
                p_col_idx, z_col_idx, ZONE_TUNING_PARAMETER,
                boot_size=N_BOOTSTRAP, dtype=torch.float64
            )
            
            # Convert to Python scalars
            elas_norm = elas_mean.item()
            std_error_norm = np.sqrt(elas_var.item())
            
            # Rescale elasticity and std error to original price units
            p_norm = period_xz_mean[p_col_idx].item()
            price_range = price_max - price_min
            if p_norm > 0 and price_range > 0:
                scale_adjustment = (price_min + p_norm * price_range) / (p_norm * price_range)
            else:
                scale_adjustment = 1.0
            elas = elas_norm * scale_adjustment
            std_error = std_error_norm * abs(scale_adjustment)
            
            # Calculate confidence intervals using normal approximation on rescaled values
            z_score = 1.96  # For 95% confidence
            ci_lower = elas - z_score * std_error
            ci_upper = elas + z_score * std_error
            
            # No per-period print; summarized later
        else:
            # Use faster point estimate without bootstrap
            elas_tensor = p_elas_2scale_torch(
                y_full, XZ_full, period_xz_mean, 
                p_col_idx, z_col_idx, ZONE_TUNING_PARAMETER,
                dtype=torch.float64
            )
            
            # Rescale mean elasticity to original price units
            p_norm = period_xz_mean[p_col_idx].item()
            price_range = price_max - price_min
            if p_norm > 0 and price_range > 0:
                scale_adjustment = (price_min + p_norm * price_range) / (p_norm * price_range)
            else:
                scale_adjustment = 1.0
            elas = elas_tensor.item() * scale_adjustment
            ci_lower = np.nan
            ci_upper = np.nan
            std_error = np.nan
            
            # No per-period print; summarized later
        
        # Statistics from actual period data
        period_data_rows = market_data_original[market_data_original['hour_of_day'].isin(hours)]
        avg_price_raw = period_data_rows['trip_price'].mean()
        avg_demand = period_data_rows['ride_count'].mean()
        iv_corr = period_data_rows[['trip_price', 'trip_price_iv']].corr().iloc[0, 1]
        price_cv = period_data_rows['trip_price'].std() / avg_price_raw if avg_price_raw > 0 else 0
        iv_cv = period_data_rows['trip_price_iv'].std() / period_data_rows['trip_price_iv'].mean()
        
        period_elasticities.append({
            'period': period_name,
            'hours': str(hours),
            'elasticity': elas,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std_error': std_error,
            'period_obs': len(period_data_rows),
            'avg_price': avg_price_raw,
            'price_cv': price_cv,
            'avg_demand': avg_demand,
            'iv_correlation': iv_corr,
            'iv_cv': iv_cv
        })
    
    period_df = pd.DataFrame(period_elasticities)
    return period_df

if __name__ == "__main__":
    MONTHS = ["2024-01", "2024-02"]
    
    # Load and prepare data first
    market_data = load_and_prepare_data(MONTHS)
    
    # Apply quality filters to select zones (logic copied from 2b)
    zone_ids = select_quality_zones(market_data)
    
    # Filter data to selected zones
    print(f"\nFiltering data to {len(zone_ids)} quality-checked zones...")
    market_data = market_data[market_data['PULocationID'].isin(zone_ids)].copy()
    print(f"Records from selected zones: {len(market_data):,}")
    
    # Prepare tensors
    XZ, y, market_data_filtered, norm_params = prepare_tensors(market_data, FEATURE_COLUMNS)
    
    # Estimate elasticity by time period subgroups (Morning Peak, Midday, Evening Peak, etc.)
    period_df = estimate_time_period_subgroups(
        market_data, market_data_filtered,
        FEATURE_COLUMNS, XZ, y, TIME_PERIODS, norm_params
    )
    
    # Save results
    os.makedirs(ELASTICITY_OUTPUT_DIR, exist_ok=True)
    
    # Save time period subgroup results
    period_path = f"{ELASTICITY_OUTPUT_DIR}step3_elasticity_temporal_periods.csv"
    # Only keep required columns in outputs
    save_cols = ['period', 'hours', 'elasticity', 'ci_lower', 'ci_upper', 'std_error']
    period_df[save_cols].to_csv(period_path, index=False)
    print(f"\nTime period subgroup results saved to {period_path}")
    
    # Streamlined reporting: show core stored fields
    print("\nTemporal elasticity by period (rescaled):")
    period_cols = ['period', 'elasticity', 'ci_lower', 'ci_upper', 'std_error']
    period_show = period_df[period_cols].copy()
    for c in ['elasticity', 'ci_lower', 'ci_upper', 'std_error']:
        period_show[c] = pd.to_numeric(period_show[c], errors='coerce').round(3)
    print(period_show.to_string(index=False))