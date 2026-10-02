"""
Configuration for the NYC ride-sharing elasticity estimation.

All results reported in the paper were produced with the values below.
"""

# ============= ESTIMATION PARAMETERS =============

# Subsampling size m for the debiased two-scale BNN estimator.
# The estimator internally combines scales (m, 2m); see bnn_modules/bnn_torch.py.
ZONE_TUNING_PARAMETER = 200

# ============= DATA CONFIGURATION =============

# Feature columns used in elasticity estimation
FEATURE_COLUMNS = [
    'trip_price',                    # Endogenous variable (instrumented)
    'trip_miles',                    # Trip distance
    'trip_time',                     # Trip duration
    'passenger_wait_time',           # Passenger wait time (market tightness)
    'trip_price_pu_hour_avg',        # Pickup zone-hour average price
    'trip_price_do_hour_avg',        # Dropoff zone-hour average price
    'trip_price_route_hour_avg',     # Route-hour average price
    'trip_price_pu_avg',             # Pickup zone average price
    'trip_price_do_avg',             # Dropoff zone average price
    'trip_price_route_avg',          # Route average price
    'PU_latitude',                   # Pickup location latitude
    'PU_longitude',                  # Pickup location longitude
    'DO_latitude',                   # Dropoff location latitude
    'DO_longitude',                  # Dropoff location longitude
    'trip_price_date_avg',           # Daily average price
    'trip_price_hour_avg'            # Hourly average price
]

# Borough filter for the zone-specific analysis in the main text
ANALYZE_BOROUGHS = ['Manhattan']

# ============= DIRECTORY PATHS =============
# Relative to the package root (empirical/)

ZONES_DIR = "./outputs/step1_processed_data/zones/"
MARKETS_DIR = "./outputs/step2_aggregated_markets/markets/"
ELASTICITY_OUTPUT_DIR = "./outputs/step3_elasticity_estimates/"

# Aliases used by the estimation scripts
AGG_DIR = MARKETS_DIR
DERIVED_DIR = ZONES_DIR
ZONE_ELAS_DIR = ELASTICITY_OUTPUT_DIR
