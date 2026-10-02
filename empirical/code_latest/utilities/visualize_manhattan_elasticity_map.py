import os
import sys
# Go up 3 levels to project root
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_dir)
sys.path.append(project_dir)

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
import numpy as np
from matplotlib.gridspec import GridSpec

# Configuration
ZONE_SHAPEFILE = './data/taxi_zones/taxi_zones.shp'
ZONE_LOOKUP = './data/taxi_zone_lookup.csv'
ELASTICITY_FILE = './outputs/step3_elasticity_estimates/step3_elasticity_zones_manhattan.csv'
OUTPUT_DIR = './outputs/figures/'

# Publication settings
COLORMAP_STYLE = 'viridis'  # Primary colormap for publication
PANEL_LABEL = None  # Set to 'A', 'B', etc. for multi-panel figures
EXPORT_VECTOR = True  # Export PDF in addition to PNG

# Figure styling - Professional publication quality
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts for better PDF compatibility
plt.rcParams['ps.fonttype'] = 42
# Enable mathtext for Greek symbols
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'

def load_and_prepare_data():
    """Load zone elasticities and geographic boundaries."""
    print("Loading data...")
    
    # Load elasticity estimates
    elasticity_df = pd.read_csv(ELASTICITY_FILE)
    print(f"Loaded {len(elasticity_df)} zone elasticity estimates with CI")
    
    # Load zone shapefile
    zones_gdf = gpd.read_file(ZONE_SHAPEFILE)
    zones_gdf['LocationID'] = zones_gdf['LocationID'].astype(int)
    
    # Load zone lookup for borough info
    zone_lookup = pd.read_csv(ZONE_LOOKUP)
    zones_gdf = zones_gdf.merge(zone_lookup[['LocationID', 'Borough']], on='LocationID', how='left')
    
    # Merge elasticity data with geographic data (exclude duplicate Borough column)
    elasticity_cols = [col for col in elasticity_df.columns if col != 'Borough']
    zones_with_elas = zones_gdf.merge(
        elasticity_df[elasticity_cols],
        left_on='LocationID',
        right_on='zone_id',
        how='left'
    )
    
    # Filter to Manhattan
    manhattan = zones_with_elas[zones_with_elas['Borough'] == 'Manhattan'].copy()
    
    print(f"Manhattan zones with elasticity estimates: {manhattan['elasticity'].notna().sum()}")
    
    # Winsorize elasticity values at 5% on each end to handle outliers
    valid_elasticity = manhattan['elasticity'].dropna()
    winsorization_stats = {}
    
    if len(valid_elasticity) > 0:
        # ========================================================================
        # DETAILED PRE-WINSORIZATION STATISTICS
        # ========================================================================
        print("\n" + "="*70)
        print("PRE-WINSORIZATION STATISTICS")
        print("="*70)
        print(f"Number of zones with elasticity estimates: {len(valid_elasticity)}")
        print(f"\nBasic Statistics:")
        print(f"  Mean:   {valid_elasticity.mean():7.4f}")
        print(f"  Median: {valid_elasticity.median():7.4f}")
        print(f"  Std:    {valid_elasticity.std():7.4f}")
        print(f"  Min:    {valid_elasticity.min():7.4f}")
        print(f"  Max:    {valid_elasticity.max():7.4f}")
        print(f"  Range:  {valid_elasticity.max() - valid_elasticity.min():7.4f}")
        
        print(f"\nDetailed Quantiles (Pre-Winsorization):")
        quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        quantile_values = {}
        for q in quantiles:
            val = valid_elasticity.quantile(q)
            quantile_values[f'q{int(q*100):02d}'] = val
            print(f"  {q*100:5.1f}th percentile: {val:7.4f}")
        
        print(f"\nDistribution Characteristics:")
        print(f"  Skewness: {valid_elasticity.skew():7.4f}")
        print(f"  Kurtosis: {valid_elasticity.kurtosis():7.4f}")
        
        # Count zones in different elasticity ranges
        print(f"\nElasticity Distribution:")
        print(f"  Zones with elasticity < -2.0: {(valid_elasticity < -2.0).sum()}")
        print(f"  Zones with elasticity < -1.0: {(valid_elasticity < -1.0).sum()}")
        print(f"  Zones with elasticity < -0.5: {(valid_elasticity < -0.5).sum()}")
        print(f"  Zones with elasticity >= -0.5: {(valid_elasticity >= -0.5).sum()}")
        print(f"  Zones with elasticity >= 0.0: {(valid_elasticity >= 0.0).sum()}")
        
        # ========================================================================
        # WINSORIZATION
        # ========================================================================
        lower_bound = valid_elasticity.quantile(0.05)
        upper_bound = valid_elasticity.quantile(0.95)
        
        print("\n" + "="*70)
        print("APPLYING WINSORIZATION")
        print("="*70)
        print(f"  Original range: [{valid_elasticity.min():.4f}, {valid_elasticity.max():.4f}]")
        print(f"  Winsorization bounds (5th-95th percentiles): [{lower_bound:.4f}, {upper_bound:.4f}]")
        
        # Store original values for reference
        manhattan['elasticity_original'] = manhattan['elasticity'].copy()
        
        # Apply winsorization
        manhattan['elasticity_winsorized'] = manhattan['elasticity'].clip(lower=lower_bound, upper=upper_bound)
        
        n_lower = (manhattan['elasticity_original'] < lower_bound).sum()
        n_upper = (manhattan['elasticity_original'] > upper_bound).sum()
        print(f"  Zones winsorized: {n_lower} at lower bound, {n_upper} at upper bound")
        print(f"  Winsorized range: [{manhattan['elasticity_winsorized'].min():.4f}, {manhattan['elasticity_winsorized'].max():.4f}]")
        
        # Print post-winsorized statistics
        q25_wins = manhattan['elasticity_winsorized'].quantile(0.25)
        q75_wins = manhattan['elasticity_winsorized'].quantile(0.75)
        print(f"\nPost-Winsorization Statistics:")
        print(f"  Mean:   {manhattan['elasticity_winsorized'].mean():7.4f}")
        print(f"  Median: {manhattan['elasticity_winsorized'].median():7.4f}")
        print(f"  Std:    {manhattan['elasticity_winsorized'].std():7.4f}")
        print(f"  Q25:    {q25_wins:7.4f}")
        print(f"  Q75:    {q75_wins:7.4f}")
        
        # Store comprehensive winsorization statistics
        winsorization_stats = {
            # Pre-winsorization statistics
            'n_zones': len(valid_elasticity),
            'pre_mean': valid_elasticity.mean(),
            'pre_median': valid_elasticity.median(),
            'pre_std': valid_elasticity.std(),
            'pre_min': valid_elasticity.min(),
            'pre_max': valid_elasticity.max(),
            'pre_range': valid_elasticity.max() - valid_elasticity.min(),
            'pre_skewness': valid_elasticity.skew(),
            'pre_kurtosis': valid_elasticity.kurtosis(),
            # Pre-winsorization quantiles
            **{f'pre_{k}': v for k, v in quantile_values.items()},
            # Pre-winsorization distribution counts
            'pre_count_below_minus_2': (valid_elasticity < -2.0).sum(),
            'pre_count_below_minus_1': (valid_elasticity < -1.0).sum(),
            'pre_count_below_minus_0.5': (valid_elasticity < -0.5).sum(),
            'pre_count_above_minus_0.5': (valid_elasticity >= -0.5).sum(),
            'pre_count_above_0': (valid_elasticity >= 0.0).sum(),
            # Winsorization bounds
            'lower_bound_percentile': 5,
            'upper_bound_percentile': 95,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'zones_winsorized_lower': n_lower,
            'zones_winsorized_upper': n_upper,
            # Post-winsorization statistics
            'post_mean': manhattan['elasticity_winsorized'].mean(),
            'post_median': manhattan['elasticity_winsorized'].median(),
            'post_std': manhattan['elasticity_winsorized'].std(),
            'post_min': manhattan['elasticity_winsorized'].min(),
            'post_max': manhattan['elasticity_winsorized'].max(),
            'post_q25': q25_wins,
            'post_q75': q75_wins,
        }
    
    return manhattan, winsorization_stats


def get_colormap_and_norm(colormap_style, vmin, vmax):
    """Get appropriate colormap and normalization based on style."""
    import matplotlib
    
    if colormap_style == 'viridis':
        cmap = matplotlib.colormaps['viridis']
        norm = Normalize(vmin=vmin, vmax=vmax)
        label = 'Price Elasticity of Demand'
        
    elif colormap_style == 'cividis':
        cmap = matplotlib.colormaps['cividis']
        norm = Normalize(vmin=vmin, vmax=vmax)
        label = 'Price Elasticity of Demand'
        
    elif colormap_style == 'plasma_r':
        cmap = matplotlib.colormaps['plasma_r']
        norm = Normalize(vmin=vmin, vmax=vmax)
        label = 'Price Elasticity of Demand'
        
    elif colormap_style == 'diverging':
        # Use diverging colormap centered at -1.0 (unit elastic)
        cmap = matplotlib.colormaps['RdBu_r']
        center = -1.0
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        label = 'Price Elasticity of Demand'
        
    elif colormap_style == 'grayscale':
        # Grayscale for B&W printing
        cmap = matplotlib.colormaps['gray_r']
        norm = Normalize(vmin=vmin, vmax=vmax)
        label = 'Price Elasticity of Demand'
        
    else:
        # Default to viridis
        cmap = matplotlib.colormaps['viridis']
        norm = Normalize(vmin=vmin, vmax=vmax)
        label = 'Price Elasticity of Demand'
    
    return cmap, norm, label

def create_manhattan_elasticity_map(manhattan_gdf, output_path, colormap_style='viridis'):
    """
    Journal-style map: single panel with map + vertical colorbar.

    - Uses winsorized elasticities if available.
    - No stats box or explanatory paragraph on the figure.
    - Legend for missing zones is minimal and unobtrusive.
    - Layout is closer to a typical Management Science map.
    """
    # --------------------------------------------------------------
    # 1. Filter to main Manhattan island and choose plotting column
    # --------------------------------------------------------------
    excluded_zone_ids = [194, 202, 103, 104, 105]  # Randalls, Roosevelt, small islands
    manhattan_gdf = manhattan_gdf[~manhattan_gdf['LocationID'].isin(excluded_zone_ids)].copy()

    if 'elasticity_winsorized' in manhattan_gdf.columns:
        plot_column = 'elasticity_winsorized'
        valid_elas = manhattan_gdf['elasticity_winsorized'].dropna()
    else:
        plot_column = 'elasticity'
        valid_elas = manhattan_gdf['elasticity'].dropna()

    if len(valid_elas) == 0:
        print("No valid elasticity values found!")
        return

    # Color scale based on winsorized data
    vmin = valid_elas.min()
    vmax = min(valid_elas.max(), 0)  # cap at 0, since all elasticities should be ≤ 0

    cmap, norm, _ = get_colormap_and_norm(colormap_style, vmin, vmax)

    # --------------------------------------------------------------
    # 2. Figure and axis: single panel for map + colorbar
    #    Size is roughly "tall single-column"; tweak as needed
    # --------------------------------------------------------------
    fig, ax_map = plt.subplots(figsize=(6.5, 6.8))  # good for eventual 1–2 column scaling

    manhattan_with_elas = manhattan_gdf[manhattan_gdf[plot_column].notna()]
    manhattan_without_elas = manhattan_gdf[manhattan_gdf[plot_column].isna()]

    # Plot zones with data
    manhattan_with_elas.plot(
        ax=ax_map,
        column=plot_column,
        cmap=cmap,
        norm=norm,
        edgecolor='white',
        linewidth=0.4,
        legend=False
    )

    # Plot zones without data in subtle gray
    if len(manhattan_without_elas) > 0:
        manhattan_without_elas.plot(
            ax=ax_map,
            color='#d9d9d9',
            edgecolor='white',
            linewidth=0.4,
            alpha=0.8
        )

    # Tight bounds around Manhattan with a tiny margin
    bounds = manhattan_gdf.total_bounds
    x_range = bounds[2] - bounds[0]
    y_range = bounds[3] - bounds[1]
    ax_map.set_xlim(bounds[0] - 0.02 * x_range, bounds[2] + 0.02 * x_range)
    ax_map.set_ylim(bounds[1] - 0.02 * y_range, bounds[3] + 0.02 * y_range)

    # --------------------------------------------------------------
    # 3. Title and clean axis
    # --------------------------------------------------------------

    ax_map.axis('off')

    # --------------------------------------------------------------
    # 4. Colorbar: slim, attached to map axis
    # --------------------------------------------------------------
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Attach to map axis; fraction/pad tune thickness & spacing
    cbar = fig.colorbar(sm, ax=ax_map, fraction=0.035, pad=0.01)

    # Explicit integer tick labels (e.g. -6, -5, ... 0)
    tick_values = np.arange(np.ceil(vmin), np.floor(vmax) + 1)
    if len(tick_values) > 1:
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f'{int(v)}' for v in tick_values])

    cbar.set_label(r'Estimated price elasticity of Uber rides',
                   fontsize=10, labelpad=6)
    cbar.ax.tick_params(labelsize=8, width=0.4, length=3)
    cbar.outline.set_linewidth(0.4)
    cbar.outline.set_edgecolor('#888888')

    # --------------------------------------------------------------
    # 5. Save as PNG + optional vector PDF
    # --------------------------------------------------------------
    fig.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nMap saved to: {output_path}")
    
    if EXPORT_VECTOR:
        pdf_path = output_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Vector version saved to: {pdf_path}")
    
    plt.close()

def create_manhattan_summary_stats(manhattan_gdf, output_path):
    """Create a summary statistics table for Manhattan with both original and winsorized values."""
    valid_elas = manhattan_gdf['elasticity'].dropna()
    
    if len(valid_elas) > 0:
        # Get winsorized data if available
        if 'elasticity_winsorized' in manhattan_gdf.columns:
            valid_elas_wins = manhattan_gdf['elasticity_winsorized'].dropna()
        else:
            valid_elas_wins = valid_elas
        
        summary_stats = {
            'Borough': 'Manhattan',
            'Zones_with_data': len(valid_elas),
            'Total_zones': len(manhattan_gdf),
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
        }
        
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv(output_path, index=False)
        
        # Format numeric columns for better display (round to 3 decimal places)
        display_df = summary_df.copy()
        numeric_cols = [col for col in display_df.columns if col not in ['Borough', 'Zones_with_data', 'Total_zones']]
        for col in numeric_cols:
            display_df[col] = display_df[col].round(3)
        
        print("\n" + "="*100)
        print("MANHATTAN SUMMARY STATISTICS (ORIGINAL VALUES)")
        print("="*100)
        original_cols = ['Borough', 'Zones_with_data', 'Total_zones', 
                         'Mean_elasticity', 'Median_elasticity', 'Std_elasticity',
                         'Min_elasticity', 'Max_elasticity']
        # Set pandas display options for better formatting
        with pd.option_context('display.max_columns', None, 'display.width', 120, 'display.max_colwidth', 15):
            print(display_df[original_cols].to_string(index=False))
        
        print("\n" + "="*100)
        print("MANHATTAN SUMMARY STATISTICS (WINSORIZED VALUES)")
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
    manhattan_gdf, winsorization_stats = load_and_prepare_data()
    
    # Save comprehensive winsorization statistics (includes pre and post)
    if winsorization_stats:
        wins_stats_file = os.path.join(OUTPUT_DIR, 'step3_fig_manhattan_winsorization_stats.csv')
        wins_df = pd.DataFrame([winsorization_stats])
        wins_df.to_csv(wins_stats_file, index=False)
        print(f"\nComprehensive winsorization statistics saved to: {wins_stats_file}")
        
        # Also save a separate file with just pre-winsorization statistics for easy reference
        pre_wins_file = os.path.join(OUTPUT_DIR, 'step3_fig_manhattan_pre_winsorization_stats.csv')
        pre_wins_cols = [k for k in winsorization_stats.keys() if k.startswith('pre_') or k == 'n_zones']
        pre_wins_df = wins_df[pre_wins_cols]
        pre_wins_df.to_csv(pre_wins_file, index=False)
        print(f"Pre-winsorization statistics saved to: {pre_wins_file}")
    
    # Create professional elasticity map
    print(f"\nCreating publication-quality map with '{COLORMAP_STYLE}' colormap...")
    output_file = os.path.join(OUTPUT_DIR, 'step3_fig_manhattan_elasticity_map.png')
    create_manhattan_elasticity_map(manhattan_gdf, output_file, colormap_style=COLORMAP_STYLE)
    
    # Create Manhattan summary statistics
    summary_file = os.path.join(OUTPUT_DIR, 'step3_fig_manhattan_summary.csv')
    create_manhattan_summary_stats(manhattan_gdf, summary_file)
    
    print("\n" + "="*100)
    print("VISUALIZATION COMPLETE")
    print("="*100)
    print(f"\nJournal-style map created:")
    print(f"  {output_file}")
    if EXPORT_VECTOR:
        print(f"  {output_file.replace('.png', '.pdf')} (vector)")
    print(f"\nSummary statistics:")
    print(f"  {summary_file}")
    print(f"  {wins_stats_file}")
    print(f"\nMinimalist journal-style design:")
    print(f"  [x] Single-panel layout (6.5\" x 6.8\") suitable for 1-2 column width")
    print(f"  [x] Simple title (left-aligned)")
    print(f"  [x] Slim colorbar with epsilon notation")
    print(f"  [x] Explicit negative tick labels (-6 to -1)")
    print(f"  [x] Vector format (PDF)" if EXPORT_VECTOR else "  [ ] Vector format")
    print(f"  [x] Viridis colormap (colorblind-friendly)")
    print(f"  [x] Thin zone boundaries (0.4pt)")
    print(f"  [x] Cropped whitespace")
    print(f"  [x] No legend, scale bar, or north arrow (cleaner for journals)")
    print(f"\n  Note: Statistics, methodological details, and missing zone explanation in caption.")

