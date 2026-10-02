import os
import sys
# Go up 3 levels to project root
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_dir)
sys.path.append(project_dir)

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

# Configuration
ZONE_SHAPEFILE = './data/taxi_zones/taxi_zones.shp'
ZONE_LOOKUP = './data/taxi_zone_lookup.csv'
ELASTICITY_FILE = './outputs/step3_elasticity_estimates/step3_elasticity_zones_all_nyc.csv'
OUTPUT_DIR = './outputs/figures/'

# Figure styling - Professional publication quality
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300
# Enable LaTeX for mathematical symbols
plt.rcParams['text.usetex'] = False  # Use mathtext instead of full LaTeX for compatibility
plt.rcParams['mathtext.fontset'] = 'dejavusans'

def load_and_prepare_data():
    """Load zone elasticities and geographic boundaries for all NYC."""
    print("Loading data...")
    
    # Load elasticity estimates
    elasticity_df = pd.read_csv(ELASTICITY_FILE)
    print(f"Loaded {len(elasticity_df)} zone elasticity estimates (all NYC)")
    
    # Determine which zones passed quality filters
    # Load market data to check quality criteria
    print("Checking quality filters...")
    MIN_ZONE_SAMPLE = 50000
    MAX_ZERO_DEMAND_PCT = 0.65
    
    try:
        markets1 = pd.read_parquet('./outputs/step2_aggregated_markets/markets/step2_markets_2024-01.parquet')
        markets2 = pd.read_parquet('./outputs/step2_aggregated_markets/markets/step2_markets_2024-02.parquet')
        all_markets = pd.concat([markets1, markets2])
        
        passed_zones = []
        for zone_id in elasticity_df['zone_id'].unique():
            zone_data = all_markets[all_markets['PULocationID'] == zone_id]
            if len(zone_data) >= MIN_ZONE_SAMPLE and (zone_data['ride_count'] == 0).mean() <= MAX_ZERO_DEMAND_PCT:
                passed_zones.append(zone_id)
        
        elasticity_df['passed_quality_filter'] = elasticity_df['zone_id'].isin(passed_zones)
        print(f"Quality filters: {len(passed_zones)} zones passed, {len(elasticity_df) - len(passed_zones)} zones failed")
    except Exception as e:
        print(f"Warning: Could not apply quality filters: {e}")
        elasticity_df['passed_quality_filter'] = True  # Default to including all
    
    # Load zone shapefile
    zones_gdf = gpd.read_file(ZONE_SHAPEFILE)
    zones_gdf['LocationID'] = zones_gdf['LocationID'].astype(int)
    
    # Load zone lookup for borough info
    zone_lookup = pd.read_csv(ZONE_LOOKUP)
    zones_gdf = zones_gdf.merge(zone_lookup[['LocationID', 'Borough']], on='LocationID', how='left')
    
    # Merge elasticity data with geographic data
    zones_with_elas = zones_gdf.merge(
        elasticity_df,
        left_on='LocationID',
        right_on='zone_id',
        how='left'
    )
    
    # Filter out zones outside the main NYC area (e.g., airports, remote areas)
    # Keep the 5 main boroughs
    main_boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
    nyc_zones = zones_with_elas[zones_with_elas['Borough'].isin(main_boroughs)].copy()
    
    print(f"NYC zones with elasticity estimates: {nyc_zones['elasticity'].notna().sum()}")
    print(f"Zones by borough:")
    for borough in main_boroughs:
        borough_data = nyc_zones[nyc_zones['Borough'] == borough]
        n_with_data = borough_data['elasticity'].notna().sum()
        n_total = len(borough_data)
        print(f"  {borough:15s}: {n_with_data:3d}/{n_total:3d} zones with data")
    
    # Winsorize elasticity values at 10% on each end to handle outliers
    valid_elasticity_all = nyc_zones['elasticity'].dropna()
    
    # Separate quality-filtered zones
    if 'passed_quality_filter' in nyc_zones.columns:
        valid_elasticity_quality = nyc_zones[nyc_zones['passed_quality_filter'] == True]['elasticity'].dropna()
    else:
        valid_elasticity_quality = valid_elasticity_all
    
    winsorization_stats = {}
    
    if len(valid_elasticity_all) > 0:
        # ========================================================================
        # DETAILED PRE-WINSORIZATION STATISTICS - ALL 97 ZONES
        # ========================================================================
        print("\n" + "="*70)
        print("PRE-WINSORIZATION STATISTICS - ALL ZONES (n=97)")
        print("="*70)
        print(f"Number of zones with elasticity estimates: {len(valid_elasticity_all)}")
        print(f"\nBasic Statistics:")
        print(f"  Mean:   {valid_elasticity_all.mean():7.4f}")
        print(f"  Median: {valid_elasticity_all.median():7.4f}")
        print(f"  Std:    {valid_elasticity_all.std():7.4f}")
        print(f"  Min:    {valid_elasticity_all.min():7.4f}")
        print(f"  Max:    {valid_elasticity_all.max():7.4f}")
        print(f"  Range:  {valid_elasticity_all.max() - valid_elasticity_all.min():7.4f}")
        
        print(f"\nDetailed Quantiles (Pre-Winsorization):")
        quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        quantile_values_all = {}
        for q in quantiles:
            val = valid_elasticity_all.quantile(q)
            quantile_values_all[f'q{int(q*100):02d}'] = val
            print(f"  {q*100:5.1f}th percentile: {val:7.4f}")
        
        print(f"\nDistribution Characteristics:")
        print(f"  Skewness: {valid_elasticity_all.skew():7.4f}")
        print(f"  Kurtosis: {valid_elasticity_all.kurtosis():7.4f}")
        
        # Count zones in different elasticity ranges
        print(f"\nElasticity Distribution:")
        print(f"  Zones with elasticity < -2.0: {(valid_elasticity_all < -2.0).sum()}")
        print(f"  Zones with elasticity < -1.0: {(valid_elasticity_all < -1.0).sum()}")
        print(f"  Zones with elasticity < -0.5: {(valid_elasticity_all < -0.5).sum()}")
        print(f"  Zones with elasticity >= -0.5: {(valid_elasticity_all >= -0.5).sum()}")
        print(f"  Zones with elasticity >= 0.0: {(valid_elasticity_all >= 0.0).sum()}")
        
        # ========================================================================
        # DETAILED PRE-WINSORIZATION STATISTICS - QUALITY-FILTERED 73 ZONES
        # ========================================================================
        print("\n" + "="*70)
        print("PRE-WINSORIZATION STATISTICS - QUALITY-FILTERED ZONES (n=73)")
        print("="*70)
        print(f"Number of zones that passed quality filters: {len(valid_elasticity_quality)}")
        print(f"\nBasic Statistics:")
        print(f"  Mean:   {valid_elasticity_quality.mean():7.4f}")
        print(f"  Median: {valid_elasticity_quality.median():7.4f}")
        print(f"  Std:    {valid_elasticity_quality.std():7.4f}")
        print(f"  Min:    {valid_elasticity_quality.min():7.4f}")
        print(f"  Max:    {valid_elasticity_quality.max():7.4f}")
        print(f"  Range:  {valid_elasticity_quality.max() - valid_elasticity_quality.min():7.4f}")
        
        print(f"\nDetailed Quantiles (Pre-Winsorization):")
        quantile_values_quality = {}
        for q in quantiles:
            val = valid_elasticity_quality.quantile(q)
            quantile_values_quality[f'q{int(q*100):02d}'] = val
            print(f"  {q*100:5.1f}th percentile: {val:7.4f}")
        
        print(f"\nDistribution Characteristics:")
        print(f"  Skewness: {valid_elasticity_quality.skew():7.4f}")
        print(f"  Kurtosis: {valid_elasticity_quality.kurtosis():7.4f}")
        
        # Count zones in different elasticity ranges
        print(f"\nElasticity Distribution:")
        print(f"  Zones with elasticity < -2.0: {(valid_elasticity_quality < -2.0).sum()}")
        print(f"  Zones with elasticity < -1.0: {(valid_elasticity_quality < -1.0).sum()}")
        print(f"  Zones with elasticity < -0.5: {(valid_elasticity_quality < -0.5).sum()}")
        print(f"  Zones with elasticity >= -0.5: {(valid_elasticity_quality >= -0.5).sum()}")
        print(f"  Zones with elasticity >= 0.0: {(valid_elasticity_quality >= 0.0).sum()}")
        
        # ========================================================================
        # WINSORIZATION
        # ========================================================================
        # Note: Winsorization is based on ALL zones for consistency with existing code
        lower_bound = valid_elasticity_all.quantile(0.10)
        upper_bound = valid_elasticity_all.quantile(0.90)
        
        print("\n" + "="*70)
        print("APPLYING WINSORIZATION (based on all zones)")
        print("="*70)
        print(f"  Original range: [{valid_elasticity_all.min():.4f}, {valid_elasticity_all.max():.4f}]")
        print(f"  Winsorization bounds (10th-90th percentiles): [{lower_bound:.4f}, {upper_bound:.4f}]")
        
        # Store original values for reference
        nyc_zones['elasticity_original'] = nyc_zones['elasticity'].copy()
        
        # Apply winsorization
        nyc_zones['elasticity_winsorized'] = nyc_zones['elasticity'].clip(lower=lower_bound, upper=upper_bound)
        
        n_lower = (nyc_zones['elasticity_original'] < lower_bound).sum()
        n_upper = (nyc_zones['elasticity_original'] > upper_bound).sum()
        print(f"  Zones winsorized: {n_lower} at lower bound, {n_upper} at upper bound")
        print(f"  Winsorized range: [{nyc_zones['elasticity_winsorized'].min():.4f}, {nyc_zones['elasticity_winsorized'].max():.4f}]")
        
        # Print post-winsorized statistics
        q25_wins = nyc_zones['elasticity_winsorized'].quantile(0.25)
        q75_wins = nyc_zones['elasticity_winsorized'].quantile(0.75)
        print(f"\nPost-Winsorization Statistics:")
        print(f"  Mean:   {nyc_zones['elasticity_winsorized'].mean():7.4f}")
        print(f"  Median: {nyc_zones['elasticity_winsorized'].median():7.4f}")
        print(f"  Std:    {nyc_zones['elasticity_winsorized'].std():7.4f}")
        print(f"  Q25:    {q25_wins:7.4f}")
        print(f"  Q75:    {q75_wins:7.4f}")
        
        # Store comprehensive winsorization statistics
        winsorization_stats = {
            # Pre-winsorization statistics - ALL ZONES
            'n_zones_all': len(valid_elasticity_all),
            'pre_all_mean': valid_elasticity_all.mean(),
            'pre_all_median': valid_elasticity_all.median(),
            'pre_all_std': valid_elasticity_all.std(),
            'pre_all_min': valid_elasticity_all.min(),
            'pre_all_max': valid_elasticity_all.max(),
            'pre_all_range': valid_elasticity_all.max() - valid_elasticity_all.min(),
            'pre_all_skewness': valid_elasticity_all.skew(),
            'pre_all_kurtosis': valid_elasticity_all.kurtosis(),
            # Pre-winsorization quantiles - ALL ZONES
            **{f'pre_all_{k}': v for k, v in quantile_values_all.items()},
            # Pre-winsorization distribution counts - ALL ZONES
            'pre_all_count_below_minus_2': (valid_elasticity_all < -2.0).sum(),
            'pre_all_count_below_minus_1': (valid_elasticity_all < -1.0).sum(),
            'pre_all_count_below_minus_0.5': (valid_elasticity_all < -0.5).sum(),
            'pre_all_count_above_minus_0.5': (valid_elasticity_all >= -0.5).sum(),
            'pre_all_count_above_0': (valid_elasticity_all >= 0.0).sum(),
            # Pre-winsorization statistics - QUALITY-FILTERED ZONES
            'n_zones_quality': len(valid_elasticity_quality),
            'pre_quality_mean': valid_elasticity_quality.mean(),
            'pre_quality_median': valid_elasticity_quality.median(),
            'pre_quality_std': valid_elasticity_quality.std(),
            'pre_quality_min': valid_elasticity_quality.min(),
            'pre_quality_max': valid_elasticity_quality.max(),
            'pre_quality_range': valid_elasticity_quality.max() - valid_elasticity_quality.min(),
            'pre_quality_skewness': valid_elasticity_quality.skew(),
            'pre_quality_kurtosis': valid_elasticity_quality.kurtosis(),
            # Pre-winsorization quantiles - QUALITY-FILTERED ZONES
            **{f'pre_quality_{k}': v for k, v in quantile_values_quality.items()},
            # Pre-winsorization distribution counts - QUALITY-FILTERED ZONES
            'pre_quality_count_below_minus_2': (valid_elasticity_quality < -2.0).sum(),
            'pre_quality_count_below_minus_1': (valid_elasticity_quality < -1.0).sum(),
            'pre_quality_count_below_minus_0.5': (valid_elasticity_quality < -0.5).sum(),
            'pre_quality_count_above_minus_0.5': (valid_elasticity_quality >= -0.5).sum(),
            'pre_quality_count_above_0': (valid_elasticity_quality >= 0.0).sum(),
            # Winsorization bounds
            'lower_bound_percentile': 10,
            'upper_bound_percentile': 90,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'zones_winsorized_lower': n_lower,
            'zones_winsorized_upper': n_upper,
            # Post-winsorization statistics
            'post_mean': nyc_zones['elasticity_winsorized'].mean(),
            'post_median': nyc_zones['elasticity_winsorized'].median(),
            'post_std': nyc_zones['elasticity_winsorized'].std(),
            'post_min': nyc_zones['elasticity_winsorized'].min(),
            'post_max': nyc_zones['elasticity_winsorized'].max(),
            'post_q25': q25_wins,
            'post_q75': q75_wins,
        }
    
    return nyc_zones, winsorization_stats

def create_nyc_elasticity_map(nyc_gdf, output_path):
    """
    Create a publication-quality map of all NYC showing zone elasticities.
    Uses winsorized elasticity values for better visualization.
    """
    # Create figure with appropriate aspect ratio for NYC (wider than Manhattan alone)
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Use sequential colormap - clean gradient without confusing middle colors
    # Purple (high elasticity) to Yellow-Orange (low elasticity)
    import matplotlib
    cmap = matplotlib.colormaps['plasma_r']  # Clean purple to yellow gradient (reversed)
    
    # Use winsorized elasticity values for better visualization
    if 'elasticity_winsorized' in nyc_gdf.columns:
        plot_column = 'elasticity_winsorized'
        valid_elas = nyc_gdf['elasticity_winsorized'].dropna()
    else:
        plot_column = 'elasticity'
        valid_elas = nyc_gdf['elasticity'].dropna()
    
    if len(valid_elas) == 0:
        print("No valid elasticity values found!")
        return
    
    # Set colorbar limits based on winsorized data
    # Force the scale to focus on negative values (most elasticities are negative)
    vmin = valid_elas.min()
    vmax = min(valid_elas.max(), 0)  # Cap at 0 to avoid having too much positive range
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Plot zones with elasticity estimates
    nyc_with_elas = nyc_gdf[nyc_gdf[plot_column].notna()]
    nyc_without_elas = nyc_gdf[nyc_gdf[plot_column].isna()]
    
    # Plot zones with data
    nyc_with_elas.plot(
        ax=ax,
        column=plot_column,
        cmap=cmap,
        norm=norm,
        edgecolor='white',
        linewidth=0.5,
        legend=False
    )
    
    # Plot zones without data in light gray
    nyc_without_elas.plot(
        ax=ax,
        color='#ebebeb',
        edgecolor='white',
        linewidth=0.5,
        alpha=0.65
    )
    
    # Add clean colorbar without label
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.036, pad=0.04, aspect=20, 
                       shrink=0.7)
    cbar.ax.tick_params(labelsize=11, width=0)
    cbar.outline.set_visible(False)
    
    # Clean styling - no title
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nMap saved to: {output_path}")
    plt.close()

def create_borough_summary_stats(nyc_gdf, output_path):
    """Create a summary statistics table by borough with both original and winsorized values."""
    main_boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
    
    summary_stats = []
    for borough in main_boroughs:
        borough_data = nyc_gdf[nyc_gdf['Borough'] == borough]
        valid_elas = borough_data['elasticity'].dropna()
        
        if len(valid_elas) > 0:
            # Get winsorized data if available
            if 'elasticity_winsorized' in borough_data.columns:
                valid_elas_wins = borough_data['elasticity_winsorized'].dropna()
            else:
                valid_elas_wins = valid_elas
            
            summary_stats.append({
                'Borough': borough,
                'Zones_with_data': len(valid_elas),
                'Total_zones': len(borough_data),
                'Mean_elasticity': valid_elas.mean(),
                'Median_elasticity': valid_elas.median(),
                'Std_elasticity': valid_elas.std(),
                'Min_elasticity': valid_elas.min(),
                'Max_elasticity': valid_elas.max(),
                'Mean_elasticity_winsorized': valid_elas_wins.mean(),
                'Median_elasticity_winsorized': valid_elas_wins.median(),
                'Std_elasticity_winsorized': valid_elas_wins.std(),
                'Min_elasticity_winsorized': valid_elas_wins.min(),
                'Max_elasticity_winsorized': valid_elas_wins.max()
            })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_path, index=False)
    
    # Format numeric columns for better display (round to 3 decimal places)
    display_df = summary_df.copy()
    numeric_cols = [col for col in display_df.columns if col not in ['Borough', 'Zones_with_data', 'Total_zones']]
    for col in numeric_cols:
        display_df[col] = display_df[col].round(3)
    
    print("\n" + "="*100)
    print("BOROUGH SUMMARY STATISTICS (ORIGINAL VALUES)")
    print("="*100)
    original_cols = ['Borough', 'Zones_with_data', 'Total_zones', 
                     'Mean_elasticity', 'Median_elasticity', 'Std_elasticity',
                     'Min_elasticity', 'Max_elasticity']
    # Set pandas display options for better formatting
    with pd.option_context('display.max_columns', None, 'display.width', 120, 'display.max_colwidth', 15):
        print(display_df[original_cols].to_string(index=False))
    
    print("\n" + "="*100)
    print("BOROUGH SUMMARY STATISTICS (WINSORIZED VALUES)")
    print("="*100)
    winsorized_cols = ['Borough', 'Zones_with_data', 'Total_zones',
                       'Mean_elasticity_winsorized', 'Median_elasticity_winsorized', 
                       'Std_elasticity_winsorized', 'Min_elasticity_winsorized', 
                       'Max_elasticity_winsorized']
    with pd.option_context('display.max_columns', None, 'display.width', 120, 'display.max_colwidth', 15):
        print(display_df[winsorized_cols].to_string(index=False))
    
    print(f"\nSummary saved to: {output_path}")

if __name__ == "__main__":
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    nyc_gdf, winsorization_stats = load_and_prepare_data()
    
    # Save comprehensive winsorization statistics (includes pre and post)
    if winsorization_stats:
        wins_stats_file = os.path.join(OUTPUT_DIR, 'step3_fig_all_nyc_winsorization_stats.csv')
        wins_df = pd.DataFrame([winsorization_stats])
        wins_df.to_csv(wins_stats_file, index=False)
        print(f"\nComprehensive winsorization statistics saved to: {wins_stats_file}")
        
        # Save pre-winsorization statistics for ALL zones
        pre_all_file = os.path.join(OUTPUT_DIR, 'step3_fig_all_nyc_pre_winsorization_all_zones.csv')
        pre_all_cols = [k for k in winsorization_stats.keys() if k.startswith('pre_all_') or k == 'n_zones_all']
        pre_all_df = wins_df[pre_all_cols]
        pre_all_df.to_csv(pre_all_file, index=False)
        print(f"Pre-winsorization statistics (all 97 zones) saved to: {pre_all_file}")
        
        # Save pre-winsorization statistics for QUALITY-FILTERED zones
        pre_quality_file = os.path.join(OUTPUT_DIR, 'step3_fig_all_nyc_pre_winsorization_quality_zones.csv')
        pre_quality_cols = [k for k in winsorization_stats.keys() if k.startswith('pre_quality_') or k == 'n_zones_quality']
        pre_quality_df = wins_df[pre_quality_cols]
        pre_quality_df.to_csv(pre_quality_file, index=False)
        print(f"Pre-winsorization statistics (quality-filtered 73 zones) saved to: {pre_quality_file}")
    
    # Create professional elasticity map for all NYC
    output_file = os.path.join(OUTPUT_DIR, 'step3_fig_all_nyc_elasticity_map.png')
    create_nyc_elasticity_map(nyc_gdf, output_file)
    
    # Create borough summary statistics
    summary_file = os.path.join(OUTPUT_DIR, 'step3_fig_all_nyc_borough_summary.csv')
    create_borough_summary_stats(nyc_gdf, summary_file)
    
    print("\n" + "="*100)
    print("VISUALIZATION COMPLETE")
    print("="*100)
    print(f"All NYC elasticity map saved to:")
    print(f"  {output_file}")
    print(f"Borough summary statistics saved to:")
    print(f"  {summary_file}")
    print(f"Winsorization statistics saved to:")
    print(f"  {wins_stats_file}")

