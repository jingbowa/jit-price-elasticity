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
DATA_DIR = "./data/main_data/"
ZONES_DIR = "./outputs/step1_processed_data/zones/"
GAS_PRICES_DIR = "./outputs/step1_processed_data/gas_prices/"
OUTPUT_DIR = "./outputs/step1_processed_data/trips/"

MPG_ESTIMATE = 35  # Miles per gallon estimate
DEPRECIATION_RATE = 0.2  # Per mile
UBER_LICENSE_CODE = "HV0003" # Centralized Uber license code

def load_reference_data():
    """Load preprocessed reference data."""
    print("\nLoading preprocessed data...")
    zones = pd.read_csv(f"{ZONES_DIR}step1_zones.csv")
    zone_coords = pd.read_csv(f"{ZONES_DIR}step1_zone_coordinates.csv")
    
    # Load comprehensive month-to-price dictionary directly from JSON
    with open(f"{GAS_PRICES_DIR}step1_gas_prices_lookup.json", 'r') as f:
        month_to_gas_price = json.load(f)
    
    return zones, zone_coords, month_to_gas_price

def standardize_dataframe(df):        
    # Convert datetime columns to pandas datetime objects
    datetime_columns = ['request_datetime', 'on_scene_datetime', 'pickup_datetime', 'dropoff_datetime']
    for col in datetime_columns:
        df[col] = pd.to_datetime(df[col])
    return df

def calculate_time_metrics(df):
    """Calculate various time-related metrics."""
    # we should use the offical trip_time, it is already in seconds    
    # Calculate wait time (time between request and pickup)
    df['passenger_wait_time'] = (df['pickup_datetime'] - df['request_datetime']).dt.total_seconds()
    
    # Calculate driver arrival time (time between request and on_scene)
    df['driver_arrival_time'] = (df['on_scene_datetime'] - df['request_datetime']).dt.total_seconds()
    df['driver_wait_time'] = (df['pickup_datetime'] - df['on_scene_datetime']).dt.total_seconds()
    
    # Calculate total time (arrival + wait + trip)
    df['driver_total_time'] = df['driver_arrival_time'] + df['driver_wait_time'] + df['trip_time']
    
    # Add basic time fields
    df['hour_of_day'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.day_name()
    
    return df

def calculate_financial_metrics(df, gas_price):
    """Calculate financial metrics including fares, costs, and earnings."""
    # Add gas price to dataframe
    df['gas_price_per_gallon'] = gas_price
    
    # total fare
    df['total_fare'] = df["base_passenger_fare"] + df["tolls"] + df["bcf"] + df["congestion_surcharge"] + df["tips"]
    
    # Calculate base fare
    df['trip_price'] = df["base_passenger_fare"]
    
    # Calculate platform fees
    df['platform_fees'] = df['airport_fee'] + df["tolls"] + df["bcf"] + df["congestion_surcharge"]
    
    # Calculate profit margin
    df['platform_profit'] = df["base_passenger_fare"] + df["tips"] - df['driver_pay']
    df['profit_margin'] = df['platform_profit'] / df['total_fare']
    
    # Calculate distance and cost metrics
    # Estimate arrival miles
    # Handle potential division by zero if trip_time is 0 (though filter_trips should prevent this)
    df['arrival_miles'] = np.where(
        df['trip_time'] > 0, 
        df['trip_miles'] * (df['driver_arrival_time'] / df['trip_time']), 
        0 
    )
    
    # Calculate total driving distance
    df['total_miles'] = df['trip_miles'] + df['arrival_miles']

    # Calculate gas cost
    df['gas_cost'] = (df['total_miles'] / MPG_ESTIMATE) * df['gas_price_per_gallon']
    
    # Calculate depreciation cost
    df['depreciation_cost'] = df['total_miles'] * DEPRECIATION_RATE
    
    # Calculate total operating cost
    df['operating_cost'] = df['gas_cost'] + df['depreciation_cost']
    
    # Calculate driver labor metrics
    df['net_driver_pay'] = df['driver_pay'] - df['operating_cost']
    
    # Calculate hourly earnings
    df['driver_hourly_earnings'] = df['net_driver_pay'] / (df['driver_total_time'] / 3600)
    
    return df

def filter_trips(df):
    """Filters trips based on specified criteria."""
    # Filter for Uber trips, non-WAV, non-shared, no airport fee, 
    # with positive trip miles and positive trip time.
    df_filtered = df[
        (df['hvfhs_license_num'] == UBER_LICENSE_CODE) &
        (df['wav_match_flag'] != 'Y') &
        (df['shared_match_flag'] != 'Y') &
        (df['airport_fee'] == 0) & # Remove trips to/from airports
        (df['trip_miles'] > 0) & # Ensures trips have actual distance
        (df['trip_time'] > 0) # Ensures trips have actual duration and prevents division by zero
    ].copy() # Apply .copy() once at the end
    return df_filtered

def process_location_data(df, zones, zone_coords):
    """Process location IDs and merge zone information."""
    # Convert trip data location IDs to numeric
    df['PULocationID'] = pd.to_numeric(df['PULocationID'], errors='coerce').astype('Int64')
    df['DOLocationID'] = pd.to_numeric(df['DOLocationID'], errors='coerce').astype('Int64')
        
    # Filter out rows with missing location IDs
    df = df.dropna(subset=['PULocationID', 'DOLocationID'])
    print(f"Rows with valid location IDs: {len(df)}")
    
    # Merge zone information with the trip data - use inner join to drop unmatched locations
    print("Merging zone information with trip data...")
    df = pd.merge(df, zones[['LocationID', 'Zone', 'Borough']], 
                 left_on='PULocationID', right_on='LocationID', how='inner')
    df = df.rename(columns={'Zone': 'PU_Zone', 'Borough': 'PU_Borough'}).drop('LocationID', axis=1)
    
    df = pd.merge(df, zones[['LocationID', 'Zone', 'Borough']], 
                 left_on='DOLocationID', right_on='LocationID', how='inner')
    df = df.rename(columns={'Zone': 'DO_Zone', 'Borough': 'DO_Borough'}).drop('LocationID', axis=1)
    
    # Merge coordinates for pickup location
    print("Adding coordinate information...")
    df = pd.merge(df, zone_coords[['LocationID', 'longitude', 'latitude']], 
                 left_on='PULocationID', right_on='LocationID', how='inner')
    df = df.rename(columns={'longitude': 'PU_longitude', 'latitude': 'PU_latitude'}).drop('LocationID', axis=1)
    
    # Merge coordinates for dropoff location
    df = pd.merge(df, zone_coords[['LocationID', 'longitude', 'latitude']], 
                 left_on='DOLocationID', right_on='LocationID', how='inner')
    df = df.rename(columns={'longitude': 'DO_longitude', 'latitude': 'DO_latitude'}).drop('LocationID', axis=1)
        
    return df

def engineer_features(df, zones, zone_coords, gas_price, year_month):
    df = standardize_dataframe(df) # Standardize datetime columns
    df = calculate_time_metrics(df) # Calculate time-based metrics
    df = filter_trips(df) # Filter trips
    df = process_location_data(df, zones, zone_coords) # Process location data
    df = calculate_financial_metrics(df, gas_price) # Calculate financial metrics
    df['year_month'] = year_month # Add year-month field for tracking
    return df

def process_file(file_path, zones, zone_coords, month_to_gas_price, year_month): 
    print(f"\nProcessing file: {file_path}")
    df = pd.read_parquet(file_path)
    gas_price = month_to_gas_price[year_month]
    
    df_processed = engineer_features(df, zones, zone_coords, gas_price, year_month)
    return df_processed
    
if __name__ == "__main__":
    # Load reference data
    zones, zone_coords, month_to_gas_price = load_reference_data()

    for month in range(1, 3): # Process the sample period: January-February 2024
        year_month = f"2024-{month:02d}"
        file_path = f"{DATA_DIR}fhvhv_tripdata_{year_month}.parquet"
        processed_df = process_file(file_path, zones, zone_coords, month_to_gas_price, year_month)
                    
        # Save processed trip data as Parquet
        output_path = f"{OUTPUT_DIR}step1_trips_{year_month}.parquet"
        processed_df.to_parquet(output_path, index=False)
        print(f"Saved processed trip data to {output_path} with {len(processed_df)} rows")

        # Save a sample of the processed data as CSV
        samples_dir = f"{OUTPUT_DIR}samples/"
        os.makedirs(samples_dir, exist_ok=True)
        sample_path = f"{samples_dir}step1_trips_{year_month}_sample.csv"
        processed_df.sample(n=1000).to_csv(sample_path, index=False)