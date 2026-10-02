import os
import sys
# Go up 3 levels: file → subdirectory → code_latest → nyc_taxi
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_dir)
sys.path.append(project_dir)
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

INPUT_DIR = "./outputs/step1_processed_data/trips"
OUTPUT_DIR = "./outputs/step2_aggregated_markets/zone_selection"
MIN_TRIP_COUNT_THRESHOLD = 100_000
VALID_HOURS = list(range(5, 24)) + [0, 1]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_processed_data(months):
    """
    Loads processed trip data for specified months and concatenates them.

    Args:
        months (list[str]): A list of month strings (e.g., ["2024-01", "2024-02"]).

    Returns:
        pd.DataFrame: A DataFrame containing combined trip data for the specified months.
    """
    all_dfs = []
    for month_str in months:
        file_path = f"{INPUT_DIR}/step1_trips_{month_str}.parquet"
        df = pd.read_parquet(file_path)
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)

def save_df_to_csv(df, filename, output_dir, description):
    """Helper function to save a DataFrame to a CSV file and print a message."""
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"Saved {description} to {output_path}")

def save_selected_popular_zones(popular_pu_zones, popular_do_zones, output_dir):
    """
    Filters zones that have more than MIN_TRIP_COUNT_THRESHOLD trips for both pickups and dropoffs
    and saves the list (Zone, LocationID) to a CSV file.

    Args:
        popular_pu_zones (pd.DataFrame): DataFrame of popular pickup zones with 'PU_Zone', 'LocationID', and 'trip_count'.
        popular_do_zones (pd.DataFrame): DataFrame of popular dropoff zones with 'DO_Zone', 'LocationID', and 'trip_count'.
        output_dir (str): The directory to save the output CSV file.
    """
    # Filter zones with more than MIN_TRIP_COUNT_THRESHOLD trips for both pickup and dropoff
    popular_pu_filtered = popular_pu_zones[popular_pu_zones['trip_count'] > MIN_TRIP_COUNT_THRESHOLD]
    popular_do_filtered = popular_do_zones[popular_do_zones['trip_count'] > MIN_TRIP_COUNT_THRESHOLD]

    common_zones = pd.merge(popular_pu_filtered, popular_do_filtered, 
                              left_on=['PU_Zone', 'LocationID'], 
                              right_on=['DO_Zone', 'LocationID'], 
                              how='inner',
                              suffixes=('_pu', '_do'))
    
    selected_zones = common_zones[['PU_Zone', 'LocationID']].copy()
    selected_zones.rename(columns={'PU_Zone': 'Zone'}, inplace=True)

    save_df_to_csv(selected_zones, "step2_zones_selected.csv", output_dir,
                      f"selected popular zones with LocationID (both PU and DO > {MIN_TRIP_COUNT_THRESHOLD} trips)")
    return selected_zones

def summarize_restricted_trips(df, selected_zones, output_dir):
    """
    Summarizes the impact of data restrictions on trip counts and key statistics.

    Args:
        df (pd.DataFrame): Combined trip data after initial cleaning.
        selected_zones (pd.DataFrame): DataFrame with zones passing the popularity threshold.
        output_dir (str): Directory to save the summary output.
    """
    if selected_zones.empty:
        print("No zones met the popularity threshold; skipping restriction summary.")
        return

    summary_df = df.copy()
    if not is_datetime64_any_dtype(summary_df['pickup_datetime']):
        summary_df['pickup_datetime'] = pd.to_datetime(summary_df['pickup_datetime'])

    selected_ids = set(selected_zones['LocationID'].unique())
    restricted_df = summary_df[
        summary_df['PULocationID'].isin(selected_ids) &
        summary_df['DOLocationID'].isin(selected_ids) &
        summary_df['pickup_datetime'].dt.hour.isin(VALID_HOURS)
    ]

    total_trips = len(restricted_df)
    if total_trips == 0:
        print("No trips remain after applying restrictions; summary not generated.")
        return

    total_trips_millions = total_trips / 1_000_000
    total_trips_before = len(summary_df)
    share_of_all_trips = (total_trips / total_trips_before * 100) if total_trips_before > 0 else 0
    mean_trip_price = restricted_df['trip_price'].mean()
    mean_trip_distance = restricted_df['trip_miles'].mean()
    mean_trip_duration_minutes = restricted_df['trip_time'].mean() / 60
    restricted_zone_count = len(
        set(restricted_df['PULocationID']).union(set(restricted_df['DOLocationID']))
    )

    summary_output = pd.DataFrame([{
        "total_trips": total_trips,
        "total_trips_millions": total_trips_millions,
        "share_of_all_trips_percent": share_of_all_trips,
        "restricted_zone_count": restricted_zone_count,
        "mean_trip_price": mean_trip_price,
        "mean_trip_distance_miles": mean_trip_distance,
        "mean_trip_duration_minutes": mean_trip_duration_minutes
    }])

    summary_path = os.path.join(output_dir, "step2_restriction_summary.csv")
    summary_output.to_csv(summary_path, index=False)

    print(
        "\nThese restrictions yield "
        f"{total_trips_millions:.2f} million trips over {restricted_zone_count} zones "
        f"({share_of_all_trips:.1f}% of all trips), with mean trip price ${mean_trip_price:.2f}, "
        f"mean distance {mean_trip_distance:.2f} miles, and mean duration {mean_trip_duration_minutes:.1f} minutes."
    )

def analyze_zone_popularity(df):
    """
    Analyzes zone and route popularity from trip data, saving results to CSV files.
    Assumes df contains 'PULocationID' and 'DOLocationID'.

    Args:
        df (pd.DataFrame): DataFrame containing trip data with 'PU_Zone', 'PULocationID', 'DO_Zone', 'DOLocationID' columns.
    """
    # Popular Pickup Zones
    # Assuming df has 'PULocationID' corresponding to 'PU_Zone'
    popular_pu_zones = df.groupby('PU_Zone').agg(
        LocationID=('PULocationID', 'first'),  # Get the LocationID for the zone
        trip_count=('PU_Zone', 'size')        # Count trips for the zone
    ).reset_index()
    popular_pu_zones = popular_pu_zones.sort_values(by='trip_count', ascending=False)
    popular_pu_zones = popular_pu_zones[['PU_Zone', 'LocationID', 'trip_count']] # Ensure column order
    save_df_to_csv(popular_pu_zones, "step2_zones_pickup_popular.csv", OUTPUT_DIR, "popular pickup zones with LocationID")

    # Popular Dropoff Zones
    # Assuming df has 'DOLocationID' corresponding to 'DO_Zone'
    popular_do_zones = df.groupby('DO_Zone').agg(
        LocationID=('DOLocationID', 'first'), # Get the LocationID for the zone
        trip_count=('DO_Zone', 'size')       # Count trips for the zone
    ).reset_index()
    popular_do_zones = popular_do_zones.sort_values(by='trip_count', ascending=False)
    popular_do_zones = popular_do_zones[['DO_Zone', 'LocationID', 'trip_count']] # Ensure column order
    save_df_to_csv(popular_do_zones, "step2_zones_dropoff_popular.csv", OUTPUT_DIR, "popular dropoff zones with LocationID")

    # Popular Routes (Pickup-Dropoff pairs)
    popular_routes = df.groupby(['PU_Zone', 'DO_Zone']).size().reset_index(name='trip_count')
    popular_routes = popular_routes.sort_values(by='trip_count', ascending=False)
    save_df_to_csv(popular_routes, "step2_routes_popular.csv", OUTPUT_DIR, "popular routes")

    selected_zones = save_selected_popular_zones(popular_pu_zones, popular_do_zones, OUTPUT_DIR)
    return selected_zones

if __name__ == "__main__":
    months_to_process = ["2024-01", "2024-02"]
    combined_trip_data = load_processed_data(months_to_process)

    selected_zones_df = analyze_zone_popularity(combined_trip_data)
    summarize_restricted_trips(combined_trip_data, selected_zones_df, OUTPUT_DIR)
    print("\nAnalysis complete. Results saved in:", OUTPUT_DIR)