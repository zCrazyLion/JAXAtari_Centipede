import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from pathlib import Path

# Set style for scientific paper quality with enhanced readability
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 13
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12


def load_game_data(game_name):
    """
    Load data for a specific game. Raises RuntimeError if files not found.
    """
    base_path = Path(f"/home/paul/Documents/JAXAtari/results/{game_name}/raw")

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


def create_visualization(games, display_names=None):
    """
    Create the JAX vs OC-Atari visualization using real data only.
    Raises RuntimeError if required data is missing.

    Args:
        games: List of game names matching the data directory structure
        display_names: Optional display names for the games (defaults to games list)
    """
    if display_names is None:
        display_names = [g.capitalize() for g in games]

    # Define colors for each game - use distinct colors for better visibility
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    # Create the figure with two subplots - LARGER figure size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.3})

    # ---- LEFT PANEL: PERFORMANCE ACROSS DIFFERENT GAMES ----
    # Load actual data for all games
    game_data = {}
    for game in games:
        try:
            game_data[game] = load_game_data(game)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load data for game '{game}' for left panel. Original error: {str(e)}")

    # Plot each game's JAX throughput data
    for i, (game, display_name) in enumerate(zip(games, display_names)):
        data = game_data[game]

        # Enhance line visibility with thicker lines and larger markers
        ax1.plot(
            data['jax_workers'],
            data['jax_throughput'],
            'o-',
            color=colors[i % len(colors)],
            label=display_name,
            linewidth=2.5,  # Thicker line
            markersize=8  # Larger marker
        )

    # Set axis properties
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=10)
    ax1.set_xlabel('Number of Parallel Environments')
    ax1.set_ylabel('Steps per Second')
    ax1.set_title('(a) JAX Performance Across Different Games', pad=20)
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # Customize tick labels
    all_x_values = np.sort(np.unique(np.concatenate([data['jax_workers'] for data in game_data.values()])))
    ax1.set_xticks(all_x_values)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'$2^{{{int(np.log2(x))}}}$' if x > 0 else '0'))

    # Move legend to top left corner OUTSIDE the plot area
    ax1.legend(loc='upper left', bbox_to_anchor=(-0.12, 1.02), ncol=2, frameon=True,
               facecolor='white', edgecolor='gray', framealpha=0.9)

    # ---- RIGHT PANEL: JAX vs OC-ATARI IMPLEMENTATION COMPARISON ----
    comparison_game = games[0]
    try:
        comp_data = game_data[comparison_game]
    except KeyError:
        raise RuntimeError(f"Data for game '{comparison_game}' is required for the right panel comparison")

    # Plot scaling on right panel
    ax2.set_title(f'(b) JAX vs OC-Atari Implementation Comparison ({display_names[0]})', pad=20)
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=10)
    ax2.set_xlabel('Number of Parallel Environments')
    ax2.set_ylabel('Steps per Second')

    # Get the data for plotting
    jax_workers = comp_data['jax_workers']
    jax_throughput = comp_data['jax_throughput']
    atari_workers = comp_data['atari_workers']
    atari_throughput = comp_data['atari_throughput']

    # Plot the comparison with enhanced visibility
    ax2.plot(atari_workers, atari_throughput, 'o-', color='#1E88E5',
             label='OC-Atari (CPU)', linewidth=2.5, markersize=8)
    ax2.plot(jax_workers, jax_throughput, 'o-', color='#E53935',
             label='JAX (GPU)', linewidth=2.5, markersize=8)

    # Annotate with throughput values for key points
    important_indices_jax = [0, len(jax_throughput) // 3, 2 * len(jax_throughput) // 3, -1]
    for i in important_indices_jax:
        if i < len(jax_throughput) and i >= 0:
            ax2.annotate(f"{jax_throughput[i]:.1f}",
                         (jax_workers[i], jax_throughput[i]),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         fontsize=10)

    important_indices_atari = [0, len(atari_throughput) // 3, 2 * len(atari_throughput) // 3, -1]
    for i in important_indices_atari:
        if i < len(atari_throughput) and i >= 0:
            ax2.annotate(f"{atari_throughput[i]:.1f}",
                         (atari_workers[i], atari_throughput[i]),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         fontsize=10)

    # Set x-axis ticks
    all_x_values_right = np.sort(np.unique(np.concatenate([jax_workers, atari_workers])))
    ax2.set_xticks(all_x_values_right)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'$2^{{{int(np.log2(x))}}}$' if x > 0 else '0'))

    # Add legend in the same position as the left panel for uniformity
    ax2.legend(loc='upper left', bbox_to_anchor=(-0.12, 1.02), frameon=True,
               facecolor='white', edgecolor='gray', framealpha=0.9)
    ax2.grid(True, which="both", ls="-", alpha=0.2)

    # Find and indicate the divergence point - where JAX starts outperforming OC-Atari
    common_x = np.intersect1d(jax_workers, atari_workers)
    if len(common_x) >= 2:
        jax_interp = np.interp(common_x, jax_workers, jax_throughput)
        atari_interp = np.interp(common_x, atari_workers, atari_throughput)
        ratios = jax_interp / atari_interp
        ratio_diffs = np.diff(ratios)
        if len(ratio_diffs) > 0:
            divergence_idx = np.argmax(ratio_diffs) + 1
            if divergence_idx < len(common_x):
                divergence_x = common_x[divergence_idx]
                arrow_y = (jax_interp[divergence_idx] + atari_interp[divergence_idx]) / 2
                ax2.annotate("Performance\nDivergence",
                             xy=(divergence_x, arrow_y),
                             xytext=(divergence_x * 2, arrow_y * 0.7),
                             arrowprops=dict(facecolor='black', shrink=0.05, width=1.0, headwidth=6, alpha=0.7),
                             fontsize=11, ha='center')

    # Add subtle note text at bottom of panels - more academic style
    ax1.text(0.5, 0.03, "All game data shown using JAX GPU implementation",
             transform=ax1.transAxes, ha='center', fontsize=11, fontstyle='italic', alpha=0.8)

    ax2.text(0.5, 0.03, "Same metric (steps/second) comparing implementation approaches",
             transform=ax2.transAxes, ha='center', fontsize=11, fontstyle='italic', alpha=0.8)

    # Add a figure title
    fig.suptitle('Comparing JAX Performance: Across Games and Against OC-Atari', fontsize=20, y=0.98)

    # Adjust the spacing between subplots - more space for legends
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, left=0.08, right=0.95)

    return fig, (ax1, ax2)


if __name__ == "__main__":
    # Games from the script - use only those that have been benchmarked
    games = ['seaquest', 'kangaroo', 'pong', 'tennis', 'skiing', 'freeway']
    display_names = ['Seaquest', 'Kangaroo', 'Pong', 'Tennis', 'Skiing', 'Freeway']

    try:
        fig, axes = create_visualization(games, display_names)
        plt.show()

        # Save the figure if needed
        # plt.savefig("/home/paul/Documents/JAXAtari/jax_performance_visualization.png", dpi=300, bbox_inches='tight')
    except RuntimeError as e:
        print(f"ERROR: {e}")
        print("\nTo resolve this issue:")
        print("1. Run the benchmark script to generate data for the required games")
        print("2. Ensure data is saved to ./results/{game_name}/raw/ directory")
        print("3. Check that all required .npy files exist")