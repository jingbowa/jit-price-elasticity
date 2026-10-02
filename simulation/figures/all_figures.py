"""Draw the simulation figures of the paper from the result files.

    python figures/all_figures.py

    plot_elasticity_curves()    Figure 1 (a)-(c): figures/fig_BLP.png, fig_complements.png, fig_variety.png
                                from simulation_results/fig_elas_curve_{BLP,complements,variety}.json
    plot_optimal_tuning_own()   Online Appendix Figure A2: figures/optimal_tuning_own.png
                                from tables/table_3_opt_tuning_own.csv (written by tables/all_tables.py)

This is the script that drew the published figures. With the shipped result files it writes the same
four PNG files as in the paper (on the reference machine, matplotlib 3.9.2: byte-identical).
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_optimal_tuning_own():
    # Read the data
    df = pd.read_csv('./tables/table_3_opt_tuning_own.csv')

    # Convert wide to long format for better plotting
    id_vars = ['model', 'observations']
    value_vars = [str(x) for x in range(4, 22, 2)]
    df_long = pd.melt(df, 
                    id_vars=id_vars,
                    value_vars=value_vars,
                    var_name='tuning_parameter',
                    value_name='value')
    df_long['tuning_parameter'] = df_long['tuning_parameter'].astype(int)

    # Set style
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    # Create figure with custom size
    fig, ax = plt.subplots(figsize=(10, 7))  # Slightly smaller figure since legend is inside

    # Define colors that are distinguishable in both color and grayscale
    colors = {
        10000: '#E69F00',  # Orange
        20000: '#56B4E9',  # Light blue
        40000: '#009E73',  # Green
        60000: '#CC79A7'   # Pink
    }

    # Define distinctive markers for better black/white printing
    markers = {
        10000: 'o',    # Circle
        20000: 's',    # Square
        40000: '^',    # Triangle up
        60000: 'D'     # Diamond
    }

    # Plot each combination of sample size
    for obs in sorted(df_long['observations'].unique()):
        model = 'BLP_corr'
        data = df_long[(df_long['model'] == model) & (df_long['observations'] == obs)]
        plt.plot(data['tuning_parameter'], data['value'],
                label=f'n={obs:,}',
                color=colors[obs],
                marker=markers[obs],
                linestyle='-',
                markersize=6,
                linewidth=1.5,
                alpha=0.8,
                markerfacecolor='white',
                markeredgewidth=1.5,
                markeredgecolor=colors[obs])

    # Update axis labels to reflect actual meanings
    plt.xlabel('Number of Products', fontsize=16, fontweight='bold')
    plt.ylabel('Optimal Tuning Value', fontsize=16, fontweight='bold')
    
    # Update title
    plt.title('DGP: Correlated random-coefficients BLP', 
            fontsize=16, pad=20)

    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.7, color='gray', linewidth=0.5)
    ax.set_axisbelow(True)

    # Create a more compact legend
    legend = ax.legend(loc='upper left',
                    bbox_to_anchor=(0.02, 0.98),
                    borderaxespad=0.,
                    fontsize=16,
                    frameon=True,
                    edgecolor='black',
                    fancybox=True,
                    ncol=2)  # Use 2 columns for more compact layout
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_facecolor('white')

    # Set background color
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Customize spines
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(0.5)

    # Set x-axis ticks to match actual tuning parameter values with larger font
    plt.xticks(df_long['tuning_parameter'].unique(), fontsize=12)
    plt.yticks(fontsize=12)

    # Update secondary y-axis ticks with larger font
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylim(ax.get_ylim())  # match the limits of the primary axis
    ax2.tick_params(axis='y', labelright=True, labelleft=False, labelsize=16)  # larger font for right ticks
    ax.tick_params(axis='y', labelright=False, labelleft=True, labelsize=16)   # larger font for left ticks

    # Add subtle box around plot area
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # Adjust layout
    plt.tight_layout()

    # Save the figure with high DPI
    plt.savefig('./figures/optimal_tuning_own.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    return None

def plot_elasticity_curves():
    # Set common style parameters
    sns.set_context("paper", font_scale=1.2)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    # Define configurations for each model
    configs = [
        {
            'file': './simulation_results/fig_elas_curve_BLP.json',
            'output': './figures/fig_BLP.png',
            'ylim_own': (-1.8, -0.5),
            'ylim_cross': (0.2, 1.0),
            'xlim': (0.4, 0.7)
        },
        {
            'file': './simulation_results/fig_elas_curve_variety.json',
            'output': './figures/fig_variety.png',
            'ylim_own': (-3, -1.2),
            'ylim_cross': (-0.3, 1.0),
            'xlim': (0.4, 0.7)
        },
        {
            'file': './simulation_results/fig_elas_curve_complements.json',
            'output': './figures/fig_complements.png',
            'ylim_own': (-1.8, -0.5),
            'ylim_cross': (-0.45, 0.1),
            'xlim': (0.4, 0.7)
        }
    ]

    # Create plots for each configuration
    for config in configs:
        # Read data
        with open(config['file'], 'r') as file:
            data = json.load(file)

        fig = plt.figure(figsize=(12, 5))
        gs = GridSpec(1, 2, figure=fig)
        fig.subplots_adjust(wspace=0.25)

        # Plot 1: Own Price Elasticity
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(data['p'], data['e_own'], color='navy', linewidth=2, 
                marker='o', markersize=5, alpha=0.5, label='Point Estimate')
        ax1.scatter(data['p'], data['t_own'], color='navy', marker='*', s=100, 
                   zorder=6, label='True Value', alpha=0.8)

        # Calculate and plot confidence intervals
        ci_lower = np.array(data['e_own']) - 1.96 * np.array(data['std_own'])
        ci_upper = np.array(data['e_own']) + 1.96 * np.array(data['std_own'])
        ax1.fill_between(data['p'], ci_lower, ci_upper, color='navy', alpha=0.2, label='95% CI')

        ax1.set_xlabel('Price', fontsize=12)
        ax1.set_ylabel('Own Price Elasticity', fontsize=12)
        ax1.legend(frameon=True, loc='upper right', fontsize=16)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_ylim(*config['ylim_own'])

        # Set x-axis limits if specified in config
        if 'xlim' in config:
            ax1.set_xlim(*config['xlim'])

        # Plot 2: Cross Price Elasticity
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(data['p'], data['e_cross'], color='darkred', linewidth=2,
                marker='o', markersize=5, alpha=0.5, label='Point Estimate')
        ax2.scatter(data['p'], data['t_cross'], color='darkred', marker='*', s=100, 
                   zorder=6, label='True Value', alpha=0.8)

        # Calculate and plot confidence intervals
        ci_lower = np.array(data['e_cross']) - 1.96 * np.array(data['std_cross'])
        ci_upper = np.array(data['e_cross']) + 1.96 * np.array(data['std_cross'])
        ax2.fill_between(data['p'], ci_lower, ci_upper, color='darkred', alpha=0.2, label='95% CI')

        ax2.set_xlabel('Price', fontsize=12)
        ax2.set_ylabel('Cross Price Elasticity', fontsize=12)
        ax2.legend(frameon=True, loc='upper right', fontsize=16)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_ylim(*config['ylim_cross'])

        # Set x-axis limits if specified in config
        if 'xlim' in config:
            ax2.set_xlim(*config['xlim'])

        plt.tight_layout()

        # Save the figure
        plt.savefig(config['output'], format='png', dpi=300, bbox_inches='tight')
        plt.close()

    return None

if __name__ == '__main__':
    plot_elasticity_curves()
    plot_optimal_tuning_own()
