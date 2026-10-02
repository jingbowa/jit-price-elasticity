import os
import sys
# Go up 3 levels: file → subdirectory → code_latest → nyc_taxi
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_dir)
sys.path.append(project_dir)
import pandas as pd
import json
from datetime import datetime

def process_gas_prices():
    """
    Process gasoline price data and create a comprehensive lookup dictionary
    that handles all possible months in the range, with fallback to the most
    recent available price for months without data.
    """
    # Create output directories if they don't exist
    gas_output_dir = "./outputs/step1_processed_data/gas_prices/"
    os.makedirs(gas_output_dir, exist_ok=True)

    print("Loading and processing gasoline price data...")
    
    # Read gasoline price data
    gas_df = pd.read_csv("./data/New_York_City_Regular_All_Formulations_Retail_Gasoline_Prices.csv")
    print(f"Loaded gasoline data shape: {gas_df.shape}")

    # Rename the long column name
    gas_df.rename(columns={
        'New York City Regular All Formulations Retail Gasoline Prices Dollars per Gallon': 'gas_price_per_gallon'
    }, inplace=True)

    # Convert month column to datetime
    gas_df['Month'] = pd.to_datetime(gas_df['Month'], format="%b-%y")

    # Create a standardized year-month column
    gas_df['year_month'] = gas_df['Month'].dt.strftime('%Y-%m')

    # Sort by date
    gas_df = gas_df.sort_values('Month')

    # Save processed data as CSV for reference
    gas_df.to_csv(f"{gas_output_dir}step1_gas_prices.csv", index=False)
    
    # Create base month-to-price dictionary
    base_month_to_gas_price = dict(zip(gas_df['year_month'], gas_df['gas_price_per_gallon']))
    
    # Create a comprehensive lookup that handles missing months
    # First, get all possible months in the range
    start_date = gas_df['Month'].min()
    end_date = datetime.now()
    
    # Generate all year-month combinations in the range
    all_months = []
    current_date = start_date
    while current_date <= end_date:
        all_months.append(current_date.strftime('%Y-%m'))
        # Move to next month
        year = current_date.year + (current_date.month == 12)
        month = (current_date.month % 12) + 1
        current_date = datetime(year, month, 1)
    
    # Create comprehensive lookup with fallback to most recent available price
    comprehensive_lookup = {}
    for month in all_months:
        if month in base_month_to_gas_price:
            comprehensive_lookup[month] = base_month_to_gas_price[month]
        else:
            # Find most recent available price
            available_months = [m for m in base_month_to_gas_price.keys() if m <= month]
            if available_months:
                most_recent_month = max(available_months)
                comprehensive_lookup[month] = base_month_to_gas_price[most_recent_month]
                print(f"No exact gas price for {month}. Using price from {most_recent_month}")
            else:
                # If no earlier price available, use the earliest available
                earliest_month = min(base_month_to_gas_price.keys())
                comprehensive_lookup[month] = base_month_to_gas_price[earliest_month]
                print(f"No earlier gas price for {month}. Using earliest available from {earliest_month}")
    
    # Save the comprehensive lookup as JSON
    with open(f"{gas_output_dir}step1_gas_prices_lookup.json", 'w') as f:
        json.dump(comprehensive_lookup, f)
    
    print("Gas price data processing complete.")
    print(f"Created comprehensive lookup with {len(comprehensive_lookup)} month entries.")
    print(f"Saved to {gas_output_dir}step1_gas_prices_lookup.json")

if __name__ == "__main__":
    process_gas_prices()