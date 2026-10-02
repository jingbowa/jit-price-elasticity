import os
import sys
# Go up 3 levels: file → subdirectory → code_latest → nyc_taxi
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_dir)
sys.path.append(project_dir)

import pandas as pd
import numpy as np
import json

# Configuration constants
ZONES_DIR = "./outputs/step1_processed_data/zones/"
GAS_PRICES_DIR = "./outputs/step1_processed_data/gas_prices/"
TRIP_DIR = "./outputs/step1_processed_data/trips/"
ZONE_SELECTION_DIR = "./outputs/step2_aggregated_markets/zone_selection/"
AGG_DIR = "./outputs/step2_aggregated_markets/markets/"
TAXI_ZONE_LOOKUP_PATH = "./data/taxi_zone_lookup.csv"
MONTHS = ["2024-01", "2024-02"]
MIN_RIDES_PER_PAIR = 100
HOURS = list(range(5, 24)) + [0, 1]

def process_trip_data(month, min_ride_count, high_traffic_zone_ids, borough_map):
    input_parquet_path = f"{TRIP_DIR}step1_trips_{month}.parquet"
    trips = pd.read_parquet(input_parquet_path).copy()
    print(f"  Loaded {len(trips)} raw trips for {month}.")

    trips = trips[
        trips['PULocationID'].isin(high_traffic_zone_ids) &
        trips['DOLocationID'].isin(high_traffic_zone_ids)
    ].copy()

    # Add PU_Borough and DO_Borough using the provided map
    trips['PU_Borough'] = trips['PULocationID'].map(borough_map)
    trips['DO_Borough'] = trips['DOLocationID'].map(borough_map)

    # Process datetime and create required columns
    trips['pickup_datetime'] = pd.to_datetime(trips['pickup_datetime'])
    trips['pickup_date'] = trips['pickup_datetime'].dt.date
    trips['weekday'] = trips['pickup_datetime'].dt.dayofweek
    trips['hour_of_day'] = trips['pickup_datetime'].dt.hour
    trips['time_window'] = trips['hour_of_day'].map(lambda h: f"{h:02d}-{(h+1)%24:02d}")
    trips['location_pair'] = trips['PULocationID'].astype(str) + '-' + trips['DOLocationID'].astype(str)
    
    # Add ride_count column (1 for each trip)
    trips['ride_count'] = 1
    
    # Filter valid hours and location pairs
    trips = trips[trips['hour_of_day'].isin(HOURS)].copy()
    pair_counts = trips.groupby('location_pair').size()
    valid_pairs = pair_counts[pair_counts >= min_ride_count].index

    # Apply all filters in one step
    trips = trips[
        (trips['location_pair'].isin(valid_pairs)) &
        (trips['trip_price'] > 0) & 
        (trips['trip_price'].notna()) & 
        (np.isfinite(trips['trip_price'])) &
        (trips['trip_time'] > 0) &
        (trips['trip_time'].notna()) &
        (np.isfinite(trips['trip_time'])) &
        (trips['trip_miles'] > 0) &
        (trips['trip_miles'].notna()) &
        (np.isfinite(trips['trip_miles']))
    ].copy()
    
    return trips

def create_balanced_panel(trips):
    """Create balanced panel with zero-ride records from trip data."""
    # Get reference info for location pairs, including all necessary location info
    pair_info = trips.groupby('location_pair').agg({
        'PULocationID': 'first',
        'DOLocationID': 'first',
        'PU_Borough': 'first',
        'DO_Borough': 'first'
    }).reset_index()
    
    # Create all possible combinations
    dates = pd.DataFrame({'pickup_date': sorted(trips['pickup_datetime'].dt.date.unique())})
    hours = pd.DataFrame({'hour_of_day': HOURS})
    hours['time_window'] = hours['hour_of_day'].map(lambda h: f"{h:02d}-{(h+1)%24:02d}")
    
    # Create cartesian product of pairs, dates, and hours
    all_markets = (pair_info
                    .merge(dates, how='cross')
                    .merge(hours, how='cross'))
    
    # Create market identifier
    all_markets['market'] = (all_markets['location_pair'] + '-' + 
                            all_markets['pickup_date'].astype(str) + '-' + 
                            all_markets['time_window'])
    
    # Add borough market identifier
    all_markets['bigger_market_borough'] = (all_markets['PU_Borough'] + '-' + 
                                            all_markets['pickup_date'].astype(str) + '-' + 
                                            all_markets['time_window'])

    return all_markets

def fill_missing_features(market_agg, features_to_fill):
    """Fill missing feature values using a hierarchical approach."""
    result = market_agg.copy()
    result['ride_count'] = result['ride_count'].fillna(0)
    
    # Add weekday information
    result['weekday'] = pd.to_datetime(result['pickup_date']).dt.dayofweek
    
    for feature in features_to_fill:
        missing_mask = result[feature].isna()
        missing_count = missing_mask.sum()
        
        if missing_count > 0:            
            # Step 1: Fill with PU-DO-date-hour averages
            date_hour_avg = (result[result[feature].notna()]
                            .groupby(['PULocationID', 'DOLocationID', 'pickup_date', 'time_window'])[feature]
                            .mean())
            
            result.loc[missing_mask, feature] = (
                result.loc[missing_mask]
                .set_index(['PULocationID', 'DOLocationID', 'pickup_date', 'time_window'])
                .index
                .map(date_hour_avg)
            )
            
            # Step 2: For remaining missing, fill with PU-DO-weekday-hour averages
            still_missing = result[feature].isna()
            if still_missing.any():
                weekday_hour_avg = (result[result[feature].notna()]
                                  .groupby(['PULocationID', 'DOLocationID', 'weekday', 'time_window'])[feature]
                                  .mean())
                
                result.loc[still_missing, feature] = (
                    result.loc[still_missing]
                    .set_index(['PULocationID', 'DOLocationID', 'weekday', 'time_window'])
                    .index
                    .map(weekday_hour_avg)
                )
            
            # Step 3: For remaining missing, fill with PU-DO-hour averages
            still_missing = result[feature].isna()
            if still_missing.any():
                location_hour_avg = (result[result[feature].notna()]
                                   .groupby(['PULocationID', 'DOLocationID', 'time_window'])[feature]
                                   .mean())
                
                result.loc[still_missing, feature] = (
                    result.loc[still_missing]
                    .set_index(['PULocationID', 'DOLocationID', 'time_window'])
                    .index
                    .map(location_hour_avg)
                )
            
            # Step 4: Drop any remaining missing values for this feature
            result = result.dropna(subset=[feature])
    
    # Clean up: remove weekday column as it was only needed for filling
    result = result.drop('weekday', axis=1)
    return result

def construct_market_ivs(market_agg):
    result = market_agg.copy()
    print("Constructing instrumental variables for markets")
    
    # Step 1: Calculate IVs using leave-one-out method
    # Add weighted price column for calculations
    result['trip_price_weighted'] = result['trip_price'] * result['ride_count']
    
    # Calculate borough-level totals once
    borough_totals = result.groupby('bigger_market_borough').agg({
        'ride_count': 'sum',
        'trip_price_weighted': 'sum'
    })
    
    # Join totals back to main DataFrame to compute leave-one-out values
    result = result.merge(
        borough_totals, 
        on='bigger_market_borough', 
        suffixes=('', '_total')
    )
    
    # Calculate leave-one-out IVs
    result['trip_price_iv'] = np.where(
        (result['ride_count_total'] - result['ride_count']) > 0,
        (result['trip_price_weighted_total'] - result['trip_price_weighted']) / 
        (result['ride_count_total'] - result['ride_count']),
        np.nan
    )
    
    # Clean up temporary columns
    result = result.drop(['trip_price_weighted', 'ride_count_total', 
                         'trip_price_weighted_total'], axis=1)
    
    # Step 2: Fill missing IV values using daily borough averages
    missing_iv_mask = result['trip_price_iv'].isna()
    missing_count = missing_iv_mask.sum()
    
    if missing_count > 0:
        print(f" Filling {missing_count} missing IV values using borough averages")
        
        # Calculate daily averages in one step
        daily_avgs = (
            result
            .groupby('bigger_market_borough')
            .apply(lambda x: 
                   np.sum(x['trip_price'] * x['ride_count']) / x['ride_count'].sum() 
                   if x['ride_count'].sum() > 0 else np.nan,
                   include_groups=False
            )
            .to_dict()
        )
        
        # Fill missing values directly
        result.loc[missing_iv_mask, 'trip_price_iv'] = (
            result.loc[missing_iv_mask, 'bigger_market_borough'].map(daily_avgs)
        )
        
        filled_count = missing_iv_mask.sum() - result['trip_price_iv'].isna().sum()
        print(f"  Successfully filled {filled_count} out of {missing_count} missing values")
    
    # Step 3: Clean data by removing infinite values and NaNs
    result = result.replace([np.inf, -np.inf], np.nan)
    
    # Count how many we're dropping
    nan_count = result['trip_price_iv'].isna().sum()
    if nan_count > 0:
        print(f"  Dropping {nan_count} rows with missing or infinite IV values")
    
    # Drop rows with NaN values
    result = result.dropna()
    
    return result

def process_month(month, distance_lookup, high_traffic_zone_ids, borough_map):
    """Process data for a single month."""
    print(f"\nProcessing {month}")
    
    # Step 1: Load and filter trip data
    trips = process_trip_data(month, MIN_RIDES_PER_PAIR, high_traffic_zone_ids, borough_map)

    # Step 2: Aggregate by market
    market_agg = (trips.groupby(['PULocationID', 'DOLocationID', 'pickup_date', 'time_window'])
                 .agg({'ride_count': 'sum',
                      'trip_price': 'mean',
                      'trip_time': 'mean',
                      'trip_miles': 'mean',
                      'passenger_wait_time': 'mean'})
                 .reset_index())
    
    # Step 3: Create balanced panel and merge with aggregated data
    all_markets = create_balanced_panel(trips)
    all_market_agg = pd.merge(
        all_markets, market_agg, 
        on=['PULocationID', 'DOLocationID', 'pickup_date', 'time_window'], 
        how='left'
    )
    
    # Step 4: Fill missing price values
    all_market_agg = fill_missing_features(all_market_agg, ['trip_price', 'trip_time', 'trip_miles', 'passenger_wait_time'])
    
    # Step 5: Compute and fill instrumental variables
    all_market_agg = construct_market_ivs(all_market_agg)
    
    # Step 6: Add distance information
    # Create a series of tuples (PULocationID, DOLocationID) to map
    pu_do_pairs = list(zip(all_market_agg['PULocationID'], all_market_agg['DOLocationID']))
    all_market_agg['geo_miles'] = pd.Series(pu_do_pairs).map(distance_lookup)
    
    # Step 7: Save results
    month_num = month.split('-')[1]
    
    # Save full parquet
    all_market_agg.to_parquet(f"{AGG_DIR}step2_markets_2024-{month_num}.parquet", index=False)
    
    # Save sample to samples subdirectory
    samples_dir = f"{AGG_DIR}samples/"
    os.makedirs(samples_dir, exist_ok=True)
    all_market_agg.head(100).to_csv(f"{samples_dir}step2_markets_2024-{month_num}_sample.csv", index=False)

    return {
        "routes": set(trips['location_pair'].unique()),
        "balanced_market_count": len(all_markets),
        "final_market_count": len(all_market_agg),
        "zero_market_count": int((all_market_agg['ride_count'] == 0).sum())
    }

def main():
    os.makedirs(AGG_DIR, exist_ok=True)
    
    # Load taxi zone lookup for borough information
    print("Loading taxi zone lookup data...")
    zone_lookup_df = pd.read_csv(TAXI_ZONE_LOOKUP_PATH)
    # Ensure LocationID is int for mapping
    zone_lookup_df['LocationID'] = zone_lookup_df['LocationID'].astype(int)
    borough_map = zone_lookup_df.set_index('LocationID')['Borough'].to_dict()

    # Load high-traffic zone IDs
    high_traffic_zones_full_path = os.path.join(ZONE_SELECTION_DIR, "step2_zones_selected.csv")
    high_traffic_df = pd.read_csv(high_traffic_zones_full_path)
    high_traffic_zone_ids = set(high_traffic_df['LocationID'].unique())
    high_traffic_df['LocationID'] = high_traffic_df['LocationID'].astype(int)
    high_traffic_with_borough = high_traffic_df.merge(
        zone_lookup_df[['LocationID', 'Borough']],
        on='LocationID',
        how='left'
    )
    zone_counts_by_borough = (high_traffic_with_borough['Borough']
                              .fillna('Unknown')
                              .value_counts()
                              .sort_index())
    total_high_traffic_zones = len(high_traffic_zone_ids)
    print("\nHigh-traffic zone summary:")
    print(f"  Total zones meeting pickup & dropoff thresholds: {total_high_traffic_zones}")
    for borough, count in zone_counts_by_borough.items():
        print(f"  {borough}: {count}")
    print()

    # Load distance lookup just once
    with open(f"{ZONES_DIR}step1_zone_distances.json", 'r') as f:
        distance_lookup_json = json.load(f)
    
    distance_lookup = {
        tuple(map(lambda x: int(float(x)), k.split(','))): v 
        for k, v in distance_lookup_json.items()
    }
    
    all_routes = set()
    total_balanced_markets = 0
    total_final_markets = 0
    total_zero_markets = 0
    
    for month in MONTHS:
        month_stats = process_month(month, distance_lookup, high_traffic_zone_ids, borough_map)
        all_routes |= month_stats["routes"]
        total_balanced_markets += month_stats["balanced_market_count"]
        total_final_markets += month_stats["final_market_count"]
        total_zero_markets += month_stats["zero_market_count"]

    print("\nAggregate market construction summary:")
    print(f"  Distinct routes across months: {len(all_routes):,}")
    print(f"  Balanced panel market cells before imputation: {total_balanced_markets:,}")
    print(f"  Final market cells after imputation & cleaning: {total_final_markets:,}")
    print(f"  Markets with zero observed trips retained: {total_zero_markets:,}")

if __name__ == "__main__":
    main()