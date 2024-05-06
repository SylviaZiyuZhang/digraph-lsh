import numpy as np
import matplotlib as mpl

rc_fonts = {
    "font.family": "serif",
    "text.usetex": True,
    "text.latex.preamble": r"""
        \usepackage{libertine}
        \usepackage[libertine]{newtxmath}
        """,
}
mpl.rcParams.update(rc_fonts)
import matplotlib.pyplot as plt
import yaml
import os
import argparse
from matplotlib.axes import Axes


LINE_FORMATTING_DATA = {
    "random_single_edge_change": {
        "label": r"\textsc{RandomEdgeEdit}",
        "color": "#d3d3d3",
        "marker": "o",
        "path": "random_single_edge_change",
    },
    "best_single_edge_change": {
        "label": r"\textsc{BestEdgeEdit}",
        "color": "#7F9FBA",
        "marker": "o",
        "path": "best_single_edge_change",
    },
    "best_single_adjustment_set_change_naive": {
        "label": r"\textsc{AdjSetEdit}",
        "color": "#7FBA82",
        "marker": "o",
        "path": "best_single_adjustment_set_change_naive",
    },
    "astar_single_edge_change": {
        "label": r"\textsc{A*EdgeEdit}",
        "color": "#ba8a7f",
        "marker": "o",
        "path": "astar_single_edge_change",
    },
}

FONTSIZE = 20

def plot_edit_distance(ax: Axes, method: str, points: int, base_path: str) -> float:
    """
    Plots the edit distance for the given method.

    Parameters:
        ax: The axis to plot on.
        method: The method to plot.
        points: The number of points to plot.
        base_path: The base path to the results.

    Returns:
        The maximum plotted value.
    """

    accumulator = None
    file_count = 0

    path = os.path.join(base_path, LINE_FORMATTING_DATA[method]["path"], "data")

    if not os.path.exists(path):
        return 0

    for filename in os.listdir(path):
        if filename.endswith("edit_distance_trajectory.npy"):
            # Load the list from the file
            filepath = os.path.join(path, filename)
            data = np.load(filepath)

            if len(data) < points:
                data = np.pad(data, (0, points - len(data)), "edge")

            if accumulator is None:
                accumulator = [float(i) for i in data]
            else:
                accumulator += data

            file_count += 1

    if file_count == 0:
        return 0

    elementwise_average = accumulator / file_count

    ax.plot(
        range(len(elementwise_average)),
        elementwise_average,
        label=LINE_FORMATTING_DATA[method]["label"],
        marker=LINE_FORMATTING_DATA[method]["marker"],
        color=LINE_FORMATTING_DATA[method]["color"],
    )

    return max(elementwise_average)


def plot_ate_diff(ax: Axes, method: str, points: int, base_path: str) -> float:
    """
    Plots the Absolute Relative ATE Error for the given method.

    Parameters:
        ax: The axis to plot on.
        method: The method to plot.
        points: The number of points to plot.
        base_path: The base path to the results.

    Returns:
        The maximum plotted value.
    """

    accumulator = None
    file_count = 0
    count_zeros = 0

    path = os.path.join(base_path, LINE_FORMATTING_DATA[method]["path"], "data")

    if not os.path.exists(path):
        return 0

    for filename in os.listdir(path):
        if filename.endswith("ate_diff_trajectory.npy"):
            # Load the list from the file
            filepath = os.path.join(path, filename)
            diff_data = np.load(filepath, allow_pickle=True)

            if len(diff_data) < points:
                diff_data = np.pad(diff_data, (0, points - len(diff_data)), "edge")

            if diff_data[-1] == 0:
                count_zeros += 1

            # Load the ate data to compute ground truth ATE
            ate_filepath = filepath.replace("ate_diff", "ate")
            ate_data = np.load(ate_filepath, allow_pickle=True)
            if len(ate_data) < points:
                ate_data = np.pad(ate_data, (0, points - len(ate_data)), "edge")

            ground_truth_ate = ate_data[0] - diff_data[0]

            # Compute the absolute relative ATE error
            data = np.abs(diff_data / ground_truth_ate)

            for i, v in enumerate(data):
                if v < 10e-4:
                    data[i] = 0

            if accumulator is None:
                accumulator = [float(i) for i in data]
            else:
                accumulator += data

            file_count += 1

    print(f"File count was {file_count}")
    print(f"The final ATE difference was 0 for {count_zeros} files")

    if file_count == 0:
        return 0

    elementwise_average = accumulator / file_count

    ax.plot(
        range(len(elementwise_average)),
        elementwise_average,
        label=LINE_FORMATTING_DATA[method]["label"],
        marker=LINE_FORMATTING_DATA[method]["marker"],
        color=LINE_FORMATTING_DATA[method]["color"],
    )

    return max(elementwise_average)


def plot_invocation_duration(
    ax: Axes, method: str, points: int, base_path: str
) -> float:
    """
    Plots the duration of each invocation for the given method.

    Parameters:
        ax: The axis to plot on.
        method: The method to plot.
        points: The number of points to plot.
        base_path: The base path to the results.

    Returns:
        The maximum plotted value.
    """

    accumulator = [0.0] * points
    file_counts = [0] * points

    path = os.path.join(base_path, LINE_FORMATTING_DATA[method]["path"], "data")

    if not os.path.exists(path):
        return 0

    for filename in os.listdir(path):
        if filename.endswith("invocation_duration_trajectory.npy"):
            # Load the list from the file
            filepath = os.path.join(path, filename)
            data = np.load(filepath)

            for i in range(len(data)):
                file_counts[i] += 1

            if len(data) < points:
                data = np.pad(
                    data, (0, points - len(data)), "constant", constant_values=0
                )

            accumulator += data

    if all(x == 0 for x in file_counts):
        return 0

    elementwise_average = [
        i / j if j != 0 else 0 for i, j in zip(accumulator, file_counts)
    ]

    ax.plot(
        range(len(elementwise_average)),
        elementwise_average,
        label=LINE_FORMATTING_DATA[method]["label"],
        marker=LINE_FORMATTING_DATA[method]["marker"],
        color=LINE_FORMATTING_DATA[method]["color"],
    )

    return max(elementwise_average)


def plot_edits_per_invocation(
    ax: Axes, method: str, points: int, base_path: str
) -> float:
    """
    Plots the numbert of edits for each invocation for the given method.

    Parameters:
        ax: The axis to plot on.
        method: The method to plot.
        points: The number of points to plot.
        base_path: The base path to the results.

    Returns:
        The maximum plotted value.
    """

    accumulator = [0] * points
    file_counts = [0] * points

    path = os.path.join(base_path, LINE_FORMATTING_DATA[method]["path"], "data")

    if not os.path.exists(path):
        return 0

    for filename in os.listdir(path):
        if filename.endswith("edits_per_invocation_trajectory.npy"):
            # Load the list from the file
            filepath = os.path.join(path, filename)
            data = np.load(filepath)

            for i in range(len(data)):
                file_counts[i+1] += (1 if data[i] > 0 else 0)

            if len(data) < points:
                orig_len = len(data)
                data = np.pad(
                    data, (1, points - orig_len - 1), "constant", constant_values=0
                )

            if accumulator is None:
                accumulator = data
            else:
                accumulator += data


    if all(x == 0 for x in file_counts):
        return 0
    

    elementwise_average = [
        i / j if j != 0 else 0 for i, j in zip(accumulator, file_counts)
    ]

    retval = max(elementwise_average)

    elementwise_average[0] = None

    ax.plot(
        range(len(elementwise_average)),
        elementwise_average,
        label=LINE_FORMATTING_DATA[method]["label"],
        marker=LINE_FORMATTING_DATA[method]["marker"],
        color=LINE_FORMATTING_DATA[method]["color"],
    )

    return retval

def plot_zero_ate_diff(ax: Axes, method: str, points: int, base_path: str) -> float:
    """
    Plots the fraction of experiments with zero ATE difference at each round.
    
    Parameters:
        ax: The axis to plot on.
        method: The method to plot.
        points: The number of points to plot.
        base_path: The base path to the results.
    """

    accumulator = [0] * points
    file_count = 0

    path = os.path.join(base_path, LINE_FORMATTING_DATA[method]["path"], "data")

    if not os.path.exists(path):
        return 0

    for filename in os.listdir(path):
        if filename.endswith("ate_diff_trajectory.npy"):
            # Load the list from the file
            filepath = os.path.join(path, filename)
            diff_data = np.load(filepath, allow_pickle=True)

            if len(diff_data) < points:
                diff_data = np.pad(diff_data, (0, points - len(diff_data)), "edge")

            accumulator += (diff_data < 10e-4).astype(int) 

            file_count += 1

    if file_count == 0:
        return 0

    elementwise_average = accumulator / file_count

    ax.plot(
        range(len(elementwise_average)),
        elementwise_average,
        label=LINE_FORMATTING_DATA[method]["label"],
        marker=LINE_FORMATTING_DATA[method]["marker"],
        color=LINE_FORMATTING_DATA[method]["color"],
    )

    return max(elementwise_average)


def wrapup_plot(filename: str, ax: Axes, max_val: float, num_points:int) -> None:
    """
    Set final formatting for the plot and save it to a file.

    Parameters:
        filename: The name of the file to save the plot to.
        ax: The axis to save.
        maxes: The maximum value of the plotted data.
        num_points: The number of points to plot.
    """

    ax.set_ylim(0, 1.1 * max_val)
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax.set_xlabel("User Interaction Index", fontsize=FONTSIZE)
    ax.set_xticks(np.arange(0, num_points, 2))
    ax.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(filename)
    plt.cla()


def plotter(path: str, skip: bool = False):
    """
    Plot the experiment data at `path`.

    Parameters:
        path: The path to the experiment data.
        skip: Whether to skip re-generating plots that exist.
    """

    # Load the experiment configuration file
    with open(os.path.join(path, "config.yml"), "r") as f:
        config = yaml.safe_load(f)
    num_points = 1 + config["run_eccs"]["num_steps"]

    # Create a directory for the plots
    plots_path = os.path.join(path, "plots")
    os.makedirs(plots_path, exist_ok=True)

    ### Graph edit distance
    print("Plotting graph edit distance...")
    if skip and os.path.exists(os.path.join(plots_path, "edit_distance.png")):
        print("Skipping edit distance plot")
    else:
        _, ax = plt.subplots()
        max_y = 0
        for method in LINE_FORMATTING_DATA:
            max_y = max(
                max_y,
                plot_edit_distance(ax, method, num_points, path),
            )
        ax.set_ylabel("Graph Edit Distance", fontsize=FONTSIZE)
        wrapup_plot(
            os.path.join(plots_path, "edit_distance.png"),
            ax,
            max_y,
            num_points
        )

    ### ATE difference
    print("Plotting ATE difference...")
    if skip and os.path.exists(os.path.join(plots_path, "ate_error.png")):
        print("Skipping ATE Error plot")
    else:
        _, ax = plt.subplots()
        max_y = 0
        for method in LINE_FORMATTING_DATA:
            max_y = max(max_y, plot_ate_diff(ax, method, num_points, path))
        ax.set_ylabel("ARE_ATE", fontsize=FONTSIZE)
        wrapup_plot(
            os.path.join(plots_path, "ate_error.png"),
            ax,
            max_y,
            num_points
        )

    ### Invocation Duration
    print("Plotting Invocation Duration...")
    if skip and os.path.exists(os.path.join(plots_path, "invocation_duration.png")):
        print("Skipping Invocation Duration plot")
    else:
        _, ax = plt.subplots()
        max_y = 0
        for method in LINE_FORMATTING_DATA:
            max_y = max(max_y, plot_invocation_duration(ax, method, num_points, path))

        ax.set_ylabel("Latency (s)", fontsize=FONTSIZE)
        wrapup_plot(
            os.path.join(plots_path, "invocation_duration.png"),
            ax,
            max_y,
            num_points
        )

    ### Edits per Invocation
    print("Plotting Edits per Invocation...")
    if skip and os.path.exists(os.path.join(plots_path, "edits_per_invocation.png")):
        print("Skipping Edits per Invocation plot")
    else:
        _, ax = plt.subplots()
        max_y = 0
        for method in LINE_FORMATTING_DATA:
            max_y = max(max_y, plot_edits_per_invocation(ax, method, num_points, path))

        ax.set_ylabel(r"\# Suggested Edits", fontsize=FONTSIZE)
        wrapup_plot(
            os.path.join(plots_path, "edits_per_invocation.png"),
            ax,
            max_y,
            num_points
        )

    ### Fraction of experiments with zero ATE difference at that round
    print("Plotting Fraction of experiments with zero ATE difference...")
    if skip and os.path.exists(os.path.join(plots_path, "zero_ate_diff.png")):
        print("Skipping Zero ATE Difference plot")
    else:
        _, ax = plt.subplots()
        max_y = 0
        for method in LINE_FORMATTING_DATA:
            max_y = max(max_y, plot_zero_ate_diff(ax, method, num_points, path))

        ax.set_ylabel("Fraction of Experiments with\nZero ARE_ATE", fontsize=FONTSIZE)
        wrapup_plot(
            os.path.join(plots_path, "zero_ate_diff.png"),
            ax,
            max_y,
            num_points
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the results")
    parser.add_argument(
        "--skip", type=bool, default=False, help="Don't re-generate plots that exist"
    )
    args = parser.parse_args()

    plotter(args.path, args.skip)
