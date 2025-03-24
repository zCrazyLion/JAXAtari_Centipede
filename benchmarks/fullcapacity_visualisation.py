import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from pathlib import Path

# Set a clean, simple style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# Constants
OUTPUT_DIR = "visualizations"
GAMES = ["pong", "breakout", "kangaroo", "freeway", "seaquest", "skiing", "tennis"]
GAME_DISPLAY_NAMES = {
    "pong": "Pong",
    "breakout": "Breakout",
    "kangaroo": "Kangaroo",
    "freeway": "Freeway",
    "seaquest": "Seaquest",
    "skiing": "Skiing",
    "tennis": "Tennis"
}

# Hardcoded hardware info
CPU_INFO = "Intel(R) Core(TM) i9-9900KF CPU @ 3.60GHz"
GPU_INFO = "NVIDIA RTX 2070"
STEPS_PER_ENV = 250_000

# Environment counts from benchmark runs
JAX_ENVS = 16384
OC_ENVS = 16


def format_number(num):
    """Format large numbers as K or M for display"""
    if num >= 1000000:
        return f"{num / 1000000:.1f}M"
    elif num >= 1000:
        return f"{num / 1000:.0f}K"
    else:
        return f"{num:.0f}"


def load_benchmark_data():
    """
    Load benchmark data from the results directory using the format from the previous script.
    Returns a nested dictionary with data organized by game and implementation.
    """
    data = {}

    for game in GAMES:
        base_path = Path(f"./results/{game}/raw")
        if not base_path.exists():
            print(f"Skipping {game}: directory not found")
            continue

        data[game] = {"jax": {}, "oc": {}}

        # Load JAX data
        try:
            # Load throughput (steps per second) from JAX
            jax_throughput_file = base_path / "jax_values_throughput.npy"
            if jax_throughput_file.exists():
                jax_throughput = np.load(jax_throughput_file)
                # Take the highest throughput value (likely from the largest number of environments)
                jax_max_throughput = np.max(jax_throughput)

                data[game]["jax"] = {
                    "steps_per_second": float(jax_max_throughput),
                    "num_envs": JAX_ENVS
                }
                print(f"Loaded JAX data for {game}: {float(jax_max_throughput):,.0f} steps/sec with {JAX_ENVS} envs")
            else:
                print(f"Warning: JAX throughput data missing for {game}")
        except Exception as e:
            print(f"Error loading JAX data for {game}: {e}")

        # Load OCAtari data
        try:
            # Load throughput (steps per second) from OCAtari
            oc_throughput_file = base_path / "atari_values_throughput.npy"
            if oc_throughput_file.exists():
                oc_throughput = np.load(oc_throughput_file)
                # Take the highest throughput value
                oc_max_throughput = np.max(oc_throughput)

                data[game]["oc"] = {
                    "steps_per_second": float(oc_max_throughput),
                    "num_envs": OC_ENVS
                }
                print(f"Loaded OCAtari data for {game}: {float(oc_max_throughput):,.0f} steps/sec with {OC_ENVS} envs")
            else:
                print(f"Warning: OCAtari throughput data missing for {game}")
        except Exception as e:
            print(f"Error loading OCAtari data for {game}: {e}")

    # Print summary
    if not data:
        print("No benchmark data found.")
    else:
        print(f"Successfully loaded data for {len(data)} games")

    return data


def create_dual_horizontal_bar_chart(data, export_formats=['png', 'svg']):
    """
    Create a dual horizontal bar chart with JAX and OC bars stacked beneath each other.
    This is the new approach replacing the simple ratio chart.
    """
    # Prepare the data
    games = []
    jax_steps = []
    oc_steps = []
    ratios = []

    for game in GAMES:
        if game not in data or not data[game]["jax"] or not data[game]["oc"]:
            continue

        games.append(GAME_DISPLAY_NAMES.get(game, game))
        jax_steps.append(data[game]["jax"]["steps_per_second"])
        oc_steps.append(data[game]["oc"]["steps_per_second"])
        ratios.append(data[game]["jax"]["steps_per_second"] / data[game]["oc"]["steps_per_second"])

    # Create DataFrame
    df = pd.DataFrame({
        'Game': games,
        'JAXtari Throughput': jax_steps,
        'OCAtari Throughput': oc_steps,
        'Speedup Ratio': ratios
    })

    # Sort by speedup ratio for more impact
    df = df.sort_values('Speedup Ratio', ascending=False)

    # Create figure and axes with a clean style
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')

    # Remove unnecessary spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create positions for the bars
    # For each game, we need two positions: one for JAX and one for OC
    positions = np.arange(len(df['Game'])) * 3  # Spacing between game groups

    # Width for the bars
    bar_height = 0.8

    # Create horizontal bars for JAX (upper)
    jax_bars = ax.barh(positions, df['JAXtari Throughput'], height=bar_height,
                       color='#4477AA', edgecolor='black', linewidth=0.5,
                       label=f'JAXtari (GPU, {JAX_ENVS:,} parallel envs)')

    # Create visible OCAtari bars - minimum width for visibility
    min_visible_width = max(df['JAXtari Throughput']) * 0.005  # Make at least 2% of the max width
    visible_oc_widths = np.maximum(df['OCAtari Throughput'], min_visible_width)

    oc_bars = ax.barh(positions + bar_height + 0.2, visible_oc_widths, height=bar_height,
                      color='#EE6677', edgecolor='black', linewidth=0.5,
                      label=f'OCAtari (CPU, {OC_ENVS} parallel envs)')

    # Add game labels on the y-axis (centered between the two bars)
    ax.set_yticks(positions + (bar_height + 0.2) / 2)
    ax.set_yticklabels(df['Game'], fontsize=13, fontweight='bold')

    # Add value labels on the bars
    for i, (jax_bar, oc_bar) in enumerate(zip(jax_bars, oc_bars)):
        # Add JAX value
        ax.text(jax_bar.get_width() * 0.5, jax_bar.get_y() + jax_bar.get_height() / 2,
                f"{format_number(df['JAXtari Throughput'].iloc[i])}",
                va='center', ha='center', color='white', fontweight='bold', fontsize=11)

        '''
        # Add OC value
        ax.text(oc_bar.get_width() * 0.5, oc_bar.get_y() + oc_bar.get_height() / 2,
                f"{format_number(df['OCAtari Throughput'].iloc[i])}",
                va='center', ha='center', color='white', fontweight='bold', fontsize=11)
        '''

        # Add speedup ratio between the bars
        ax.text(jax_bar.get_width() + jax_bar.get_width() * 0.02,
                jax_bar.get_y() + jax_bar.get_height() + 0.1,
                f"{df['Speedup Ratio'].iloc[i]:.1f}× faster",
                va='center', ha='left', color='black', fontweight='bold', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='lightgray', alpha=0.9))

    # Set axis labels and title
    ax.set_xlabel('Steps per Second', fontsize=14, fontweight='bold')
    ax.set_title('JAXtari Performance Advantage over OCAtari', fontsize=18, fontweight='bold', pad=20)

    # Set x-axis limit to ensure all bars are visible with some padding
    max_throughput = max(df['JAXtari Throughput']) * 1.15
    ax.set_xlim(0, max_throughput)

    # Format x-axis with commas for thousands
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Add grid lines only on the x-axis
    ax.grid(axis='x', linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(title="Implementation",
              title_fontsize=12, fontsize=11, loc='upper right')

    # Add annotation about hardware and steps per env
    plt.figtext(0.5, 0.01,
                f"Benchmark testing was conducted using {STEPS_PER_ENV:,} steps per environment on an {CPU_INFO} processor with an {GPU_INFO} graphics card.",
                ha='center', fontsize=11, color='#333333', style='italic')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.07, 1, 0.97])

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save the figure
    for fmt in export_formats:
        out_path = f"{OUTPUT_DIR}/JAXtari_dual_horizontal_bar.{fmt}"
        plt.savefig(out_path, dpi=300, bbox_inches='tight', format=fmt)
        print(f"Saved dual horizontal bar chart as {fmt.upper()} file: {out_path}")

    plt.close()


def create_dual_bar_chart(data, export_formats=['png', 'svg']):
    """
    Create a vertical dual bar chart with absolute values.
    """
    # Prepare the data
    games = []
    jax_steps = []
    oc_steps = []
    ratios = []

    for game in GAMES:
        if game not in data or not data[game]["jax"] or not data[game]["oc"]:
            continue

        games.append(GAME_DISPLAY_NAMES.get(game, game))
        # Convert to thousands to make rendering easier
        jax_steps.append(data[game]["jax"]["steps_per_second"] / 1000)
        oc_steps.append(data[game]["oc"]["steps_per_second"] / 1000)
        ratios.append(data[game]["jax"]["steps_per_second"] / data[game]["oc"]["steps_per_second"])

    # Create DataFrame
    df = pd.DataFrame({
        'Game': games,
        'JAXtari Throughput (K)': jax_steps,
        'OCAtari Throughput (K)': oc_steps,
        'Speedup Ratio': ratios
    })

    # Sort by speedup ratio for more impact
    df = df.sort_values('Speedup Ratio', ascending=False)

    # Create figure and axes with a clean style
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')

    # Remove unnecessary spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Calculate positions
    x = np.arange(len(df['Game'])) * 1.5  # More spacing
    width = 0.6  # Wider bars

    # Create bars with a simpler approach - no log scale
    jax_bars = ax.bar(x - width / 2, df['JAXtari Throughput (K)'], width,
                      label=f'JAXtari (GPU, {JAX_ENVS:,} parallel envs)',
                      color='#4477AA', edgecolor='black', linewidth=0.5)

    oc_bars = ax.bar(x + width / 2, df['OCAtari Throughput (K)'], width,
                     label=f'OCAtari (CPU, {OC_ENVS} parallel envs)',
                     color='#EE6677', edgecolor='black', linewidth=0.5)

    # Add OC value - position label in the center of the visible section
    for i, oc_bar in enumerate(oc_bars):
        # Get the real value instead of the displayed width
        real_value = df['OCAtari Throughput'].iloc[i]

        ax.text(min(real_value * 1.2, oc_bar.get_width() * 0.5),
                oc_bar.get_y() + oc_bar.get_height() / 2,
                f"{format_number(real_value)}",
                va='center', ha='center', color='white', fontweight='bold', fontsize=11)

    # Add ratio annotations above the JAXtari bars
    for i, bar in enumerate(jax_bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{df['Speedup Ratio'].iloc[i]:.1f}×",
                ha='center', va='bottom', fontweight='bold', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray', alpha=0.9))

        # Add value label on each bar
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"{int(df['JAXtari Throughput (K)'].iloc[i])}K",
                ha='center', va='center', fontweight='bold', color='white', fontsize=11)

        # Add value label on OCAtari bars
        ax.text(oc_bars[i].get_x() + oc_bars[i].get_width() / 2, oc_bars[i].get_height() / 2,
                f"{int(df['OCAtari Throughput (K)'].iloc[i])}K",
                ha='center', va='center', fontweight='bold', color='white', fontsize=11)

    # Set axis labels and title
    ax.set_ylabel('Steps per Second (thousands)', fontsize=14, fontweight='bold')
    ax.set_title('JAXtari vs OCAtari Performance Comparison', fontsize=18, fontweight='bold', pad=20)

    # Configure x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(df['Game'], fontsize=13, fontweight='bold')

    # Set y-axis limit to ensure all bars are visible
    ax.set_ylim(0, max(df['JAXtari Throughput (K)']) * 1.15)

    # Add grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(title="Implementation",
              title_fontsize=12, fontsize=11, loc='upper right')

    # Add annotation about hardware and steps per env
    plt.figtext(0.5, 0.01,
                f"Benchmark testing was conducted using {STEPS_PER_ENV:,} steps per environment on an {CPU_INFO} processor with an {GPU_INFO} graphics card.",
                ha='center', fontsize=11, color='#333333', style='italic')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.07, 1, 0.97])

    # Save the figure
    for fmt in export_formats:
        out_path = f"{OUTPUT_DIR}/JAXtari_dual_bar.{fmt}"
        plt.savefig(out_path, dpi=300, bbox_inches='tight', format=fmt)
        print(f"Saved dual bar chart as {fmt.upper()} file: {out_path}")

    plt.close()


def create_lollipop_chart(data, export_formats=['png', 'svg']):
    """
    Create a lollipop chart which reliably renders well in SVG.
    """
    # Prepare the data
    games = []
    jax_steps = []
    oc_steps = []
    ratios = []

    for game in GAMES:
        if game not in data or not data[game]["jax"] or not data[game]["oc"]:
            continue

        games.append(GAME_DISPLAY_NAMES.get(game, game))
        jax_steps.append(data[game]["jax"]["steps_per_second"])
        oc_steps.append(data[game]["oc"]["steps_per_second"])
        ratios.append(data[game]["jax"]["steps_per_second"] / data[game]["oc"]["steps_per_second"])

    # Create DataFrame
    df = pd.DataFrame({
        'Game': games,
        'JAXtari Throughput': jax_steps,
        'OCAtari Throughput': oc_steps,
        'Speedup Ratio': ratios
    })

    # Sort by speedup ratio for more impact
    df = df.sort_values('Speedup Ratio', ascending=False)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')

    # Remove unnecessary spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set up positions with more spacing
    y_pos = np.arange(len(df['Game'])) * 1.2

    # Create horizontal lines for each game
    for i, game in enumerate(df['Game']):
        # Plot lines from OC to JAX values
        ax.plot([df['OCAtari Throughput'].iloc[i], df['JAXtari Throughput'].iloc[i]],
                [y_pos[i], y_pos[i]],
                color='gray', linestyle='-', linewidth=1.5, zorder=1)

        # Plot JAX dots
        ax.plot(df['JAXtari Throughput'].iloc[i], y_pos[i],
                marker='o', markersize=12, color='#4477AA',
                markeredgecolor='black', markeredgewidth=0.5, zorder=3)

        # Plot OC dots
        ax.plot(df['OCAtari Throughput'].iloc[i], y_pos[i],
                marker='o', markersize=12, color='#EE6677',
                markeredgecolor='black', markeredgewidth=0.5, zorder=3)

        # Add ratio as text
        midpoint = np.sqrt(
            df['OCAtari Throughput'].iloc[i] * df['JAXtari Throughput'].iloc[i])  # Geometric mean for log scale
        ax.text(midpoint, y_pos[i] + 0.15,
                f"{df['Speedup Ratio'].iloc[i]:.1f}× faster",
                ha='center', va='bottom', fontweight='bold', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                          edgecolor='lightgray', alpha=0.9))

    # Set x-axis to log scale
    ax.set_xscale('log')

    # Format x-axis with formatted numbers
    def log_axis_formatter(x, pos):
        if x >= 1000000:
            return f"{x / 1000000:.1f}M"
        elif x >= 1000:
            return f"{x / 1000:.0f}K"
        else:
            return f"{x:.0f}"

    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(log_axis_formatter))

    # Add labels for each dot
    for i, (jax_val, oc_val) in enumerate(zip(df['JAXtari Throughput'], df['OCAtari Throughput'])):
        # JAX label
        ax.text(jax_val, y_pos[i] - 0.15, format_number(jax_val),
                ha='center', va='top', fontsize=11, color='#4477AA', fontweight='bold')

        # OC label
        ax.text(oc_val, y_pos[i] - 0.15, format_number(oc_val),
                ha='center', va='top', fontsize=11, color='#EE6677', fontweight='bold')

    # Set axis labels and title
    ax.set_xlabel('Steps per Second (log scale)', fontsize=14, fontweight='bold')
    ax.set_title('JAXtari vs OCAtari Performance Comparison', fontsize=18, fontweight='bold', pad=20)

    # Configure y-axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Game'], fontsize=13, fontweight='bold')

    # Add grid lines on the x-axis
    ax.grid(axis='x', linestyle='--', alpha=0.3, color='gray', zorder=0)
    ax.set_axisbelow(True)

    # Add legend
    jax_line = mpl.lines.Line2D([], [], color='#4477AA', marker='o', linestyle='None',
                                markersize=12, label=f'JAXtari (GPU, {JAX_ENVS:,} parallel envs)')
    oc_line = mpl.lines.Line2D([], [], color='#EE6677', marker='o', linestyle='None',
                               markersize=12, label=f'OCAtari (CPU, {OC_ENVS} parallel envs)')

    ax.legend(handles=[jax_line, oc_line], loc='upper right',
              title="Implementation",
              title_fontsize=12, fontsize=11)

    # Add annotation about hardware and steps per env
    plt.figtext(0.5, 0.01,
                f"Benchmark testing was conducted using {STEPS_PER_ENV:,} steps per environment on an {CPU_INFO} processor with an {GPU_INFO} graphics card.",
                ha='center', fontsize=11, color='#333333', style='italic')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.07, 1, 0.97])

    # Save the figure
    for fmt in export_formats:
        out_path = f"{OUTPUT_DIR}/JAXtari_lollipop_chart.{fmt}"
        plt.savefig(out_path, dpi=300, bbox_inches='tight', format=fmt)
        print(f"Saved lollipop chart as {fmt.upper()} file: {out_path}")

    plt.close()


def main():
    """Main function to create the visualizations."""
    # Load benchmark data
    data = load_benchmark_data()

    print("Creating dual horizontal bar chart (new ratio chart)...")
    create_dual_horizontal_bar_chart(data, export_formats=['png', 'svg'])

    print("Creating dual bar chart with absolute values...")
    create_dual_bar_chart(data, export_formats=['png', 'svg'])

    print("Creating lollipop chart...")
    create_lollipop_chart(data, export_formats=['png', 'svg'])

    print(f"Visualizations have been saved to the '{OUTPUT_DIR}' directory.")


if __name__ == "__main__":
    main()