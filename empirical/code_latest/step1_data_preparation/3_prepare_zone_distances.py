import os
import sys
# Go up 3 levels: file → subdirectory → code_latest → nyc_taxi
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_dir)
sys.path.append(project_dir)

import pandas as pd
from geopy.distance import geodesic
import pickle
import json

ZONES_DIR = "./outputs/step1_processed_data/zones/"

def create_distance_lookup():
    # Load zone coordinates
    zone_coords = pd.read_csv(f"{ZONES_DIR}step1_zone_coordinates.csv")
    
    # Create a dictionary to store distances
    distance_lookup = {}
    
    # Iterate through all possible zone pairs
    for _, origin in zone_coords.iterrows():
        origin_id = origin['LocationID']
        origin_coords = (origin['latitude'], origin['longitude'])
        
        for _, dest in zone_coords.iterrows():
            dest_id = dest['LocationID']
            dest_coords = (dest['latitude'], dest['longitude'])
            
            # Calculate distance and store in dictionary
            distance = geodesic(origin_coords, dest_coords).miles
            distance_lookup[(origin_id, dest_id)] = distance
    
    # Save as pickle
    with open(f"{ZONES_DIR}step1_zone_distances.pkl", 'wb') as f:
        pickle.dump(distance_lookup, f)
    
    # Convert tuple keys to strings for JSON compatibility
    json_lookup = {f"{o},{d}": dist for (o, d), dist in distance_lookup.items()}
    
    # Save as JSON
    with open(f"{ZONES_DIR}step1_zone_distances.json", 'w') as f:
        json.dump(json_lookup, f)
    
    print(f"Distance lookup table created with {len(distance_lookup)} pairs")
    print(f"Saved as both .pkl and .json files in {ZONES_DIR}")
    return distance_lookup

if __name__ == "__main__":
    create_distance_lookup()