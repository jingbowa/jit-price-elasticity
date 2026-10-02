import os
import sys
# Go up 3 levels: file → subdirectory → code_latest → nyc_taxi
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_dir)
sys.path.append(project_dir)

import pandas as pd
import geopandas as gpd
import os

def process_zone_data():
    # Create output directory if it doesn't exist
    zones_output_dir = "./outputs/step1_processed_data/zones/"
    os.makedirs(zones_output_dir, exist_ok=True)

    print("Loading and processing zone data...")
    
    # Load taxi zone lookup table
    zones = pd.read_csv("./data/taxi_zone_lookup.csv")
    print(f"Number of zones: {len(zones)}")

    # Load taxi zone shapefile
    taxi_zones_gdf = gpd.read_file("./data/taxi_zones/")
    print(f"Number of zone shapes: {len(taxi_zones_gdf)}")

    # Convert to WGS84 coordinates if needed
    if taxi_zones_gdf.crs and 'EPSG:4326' not in str(taxi_zones_gdf.crs):
        print("Converting shapefile to WGS84 coordinates...")
        taxi_zones_gdf = taxi_zones_gdf.to_crs(epsg=4326)
    
    # Project to a suitable projected CRS for New York City (NAD83 / New York Long Island)
    taxi_zones_projected = taxi_zones_gdf.to_crs(epsg=2263)
    
    # Calculate centroids on the projected data
    centroids = taxi_zones_projected.geometry.centroid
    
    # Convert centroids back to WGS84 for lat/long values
    centroids_wgs84 = gpd.GeoSeries(centroids, crs=2263).to_crs(epsg=4326)
    
    # Add coordinates to the original dataframe
    taxi_zones_gdf['longitude'] = centroids_wgs84.x
    taxi_zones_gdf['latitude'] = centroids_wgs84.y

    # Clean string values
    for column in zones.select_dtypes(include=['object']).columns:
        zones[column] = zones[column].astype(str).str.strip("'").str.strip()

    # Filter out "Outside of NYC" and "Unknown" locations
    zones_filtered = zones[~zones['Zone'].str.contains('Outside of NYC|Unknown', na=False, regex=True)].copy()
    print(f"Removed {len(zones) - len(zones_filtered)} locations")
    
    # Ensure LocationID is integer - using .loc to avoid SettingWithCopyWarning
    zones_filtered.loc[:, 'LocationID'] = pd.to_numeric(zones_filtered['LocationID'], errors='coerce').astype(int)

    # Extract coordinate data for zones
    zone_coords = taxi_zones_gdf[['LocationID', 'longitude', 'latitude']].copy()
    zone_coords['LocationID'] = pd.to_numeric(zone_coords['LocationID'], errors='coerce')
    zone_coords = zone_coords.drop_duplicates('LocationID')

    # Save processed data
    zones_filtered.to_csv(f"{zones_output_dir}step1_zones.csv", index=False)
    zone_coords.to_csv(f"{zones_output_dir}step1_zone_coordinates.csv", index=False)
    
    print(f"Zone data processing complete. Files saved to {zones_output_dir}")

if __name__ == "__main__":
    process_zone_data()