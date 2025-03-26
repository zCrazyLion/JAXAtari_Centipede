import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from pathlib import Path
import os

# Set a clean, simple style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# Constants for hardware and configuration
CPU_INFO = "Intel(R) Core(TM) i9-9900KF CPU @ 3.60GHz"
GPU_INFO = "NVIDIA RTX 2070"
STEPS_PER_ENV = 250_000
OUTPUT_DIR = "visualizations"


def load_game_data(game_name):
    """
    Load data for a specific game. Raises RuntimeError if files not found.
    """
    base_path = Path(f"./results/{game_name}/raw")

    if not base_path.exists():
        raise RuntimeError(f"Data directory not found for game '{game_name}'. Expected: {base_path}")

    required_files = [
        "jax_workers.npy",
        "jax_values_throughput.npy",
        "jax_values_time.npy",
        "atari_workers.npy",
        "atari_values_throughput.npy",
        "atari_values_time.npy"
    ]

    # Check if all files exist
    missing_files = [f for f in required_files if not (base_path / f).exists()]
    if missing_files:
        raise RuntimeError(f"Missing data files for game '{game_name}': {', '.join(missing_files)}")

    # Load scaling data
    jax_workers = np.load(base_path / "jax_workers.npy")
    jax_throughput = np.load(base_path / "jax_values_throughput.npy")
    jax_time = np.load(base_path / "jax_values_time.npy")

    atari_workers = np.load(base_path / "atari_workers.npy")
    atari_throughput = np.load(base_path / "atari_values_throughput.npy")
    atari_time = np.load(base_path / "atari_values_time.npy")

    return {
        'jax_workers': jax_workers,
        'jax_throughput': jax_throughput,
        'jax_time': jax_time,
        'atari_workers': atari_workers,
        'atari_throughput': atari_throughput,
        'atari_time': atari_time
    }


def create_jax_vs_ocatari_comparison(games, display_names=None, output_file="jax_vs_ocatari_comparison.svg"):
    """
    Create a visualization comparing JAX vs OCAtari performance across all games.
    Ensures proper spacing for legend and descriptive text.

    Args:
        games: List of game names matching the data directory structure
        display_names: Optional display names for the games (defaults to games list)
        output_file: Filename for the SVG output
    """
    if display_names is None:
        display_names = [g.capitalize() for g in games]

    # Create a larger figure for the comparison
    plt.figure(figsize=(20, 12))  # Change from (20, 14) to (20, 12)

    # Define marker styles for better distinction
    jax_markers = ['o', 's', '^', 'D', 'v', 'p']
    ocatari_markers = ['o', 's', '^', 'D', 'v', 'p']

    # Define colors - different color for each game
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    # Define line styles
    jax_linestyle = 'solid'
    ocatari_linestyle = 'dashed'

    # Load and plot data for each game
    for i, (game, display_name) in enumerate(zip(games, display_names)):
        try:
            data = load_game_data(game)

            # Plot JAX data (solid line)
            plt.plot(
                data['jax_workers'],
                data['jax_throughput'],
                marker=jax_markers[i % len(jax_markers)],
                linestyle=jax_linestyle,
                color=colors[i % len(colors)],
                linewidth=2.5,
                markersize=8,
                label=f"JAXtari - {display_name}"
            )

            # Plot OCAtari data (dotted line)
            plt.plot(
                data['atari_workers'],
                data['atari_throughput'],
                marker=ocatari_markers[i % len(ocatari_markers)],
                linestyle=ocatari_linestyle,
                color=colors[i % len(colors)],
                linewidth=2.0,
                markersize=8,
                label=f"OCAtari - {display_name}"
            )

            # Annotate the highest throughput values for each implementation with improved positioning
            max_jax_idx = np.argmax(data['jax_throughput'])
            plt.annotate(
                f"{data['jax_throughput'][max_jax_idx]:.1f}",
                (data['jax_workers'][max_jax_idx], data['jax_throughput'][max_jax_idx]),
                textcoords="offset points",
                xytext=(5, 10),  # Offset slightly right to avoid overlap with line
                ha='left',  # Left-align text
                fontsize=11,  # Appropriate font size
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8)  # Add background
            )

            max_atari_idx = np.argmax(data['atari_throughput'])
            plt.annotate(
                f"{data['atari_throughput'][max_atari_idx]:.1f}",
                (data['atari_workers'][max_atari_idx], data['atari_throughput'][max_atari_idx]),
                textcoords="offset points",
                xytext=(-5, -15),  # Offset below and left to avoid overlap
                ha='right',  # Right-align text
                fontsize=11,  # Appropriate font size
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8)  # Add background
            )

        except RuntimeError as e:
            print(f"Error loading data for {game}: {e}")
            continue

    # Set axis properties
    plt.xscale('log', base=2)
    plt.yscale('log', base=10)
    plt.xlabel('Number of Parallel Environments', fontsize=14, fontweight='bold')
    plt.ylabel('Steps per Second', fontsize=14, fontweight='bold')
    plt.title('JAXtari vs OCAtari Performance Comparison Across Games', fontsize=18, fontweight='bold', pad=20)
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Configure x-axis to use powers of 2
    all_workers = []
    for game in games:
        try:
            data = load_game_data(game)
            all_workers.extend(data['jax_workers'])
            all_workers.extend(data['atari_workers'])
        except:
            continue

    all_x_values = np.sort(np.unique(all_workers))
    plt.xticks(all_x_values)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: f'$2^{{{int(np.log2(x))}}}$' if x > 0 else '0'
    ))

    # Create a more organized legend with a better layout
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.15), ncol=3, frameon=True,
               facecolor='white', edgecolor='gray', framealpha=0.9)

    # Add annotation about hardware and steps per env
    plt.figtext(0.5, 0.02,
                f"Benchmark testing was conducted using {STEPS_PER_ENV:,} steps per environment on an {CPU_INFO} processor with an {GPU_INFO} graphics card.",
                ha='center', fontsize=11, color='#333333', style='italic')

    # Tight layout with space for the legend only
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Change from 0.18 to 0.12

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save the figure
    for fmt in ['png', 'svg']:
        out_path = f"{OUTPUT_DIR}/{output_file.replace('.svg', f'.{fmt}')}"
        plt.savefig(out_path, format=fmt, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {out_path}")

    return plt.gcf()


if __name__ == "__main__":
    # Games to visualize
    games = ['seaquest', 'kangaroo', 'pong', 'tennis', 'skiing', 'freeway', 'breakout']
    display_names = ['Seaquest', 'Kangaroo', 'Pong', 'Tennis', 'Skiing', 'Freeway', 'Breakout']

    try:
        fig = create_jax_vs_ocatari_comparison(games, display_names, "jax_vs_ocatari_comparison.svg")
        plt.show()
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nTo resolve this issue:")
        print("1. Run the benchmark script to generate data for the required games")
        print("2. Ensure data is saved to ./results/{game_name}/raw/ directory")
        print("3. Check that all required .npy files exist")