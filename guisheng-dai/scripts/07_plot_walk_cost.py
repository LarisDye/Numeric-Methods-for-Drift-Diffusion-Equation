"""Plot the recorded random-walk cost experiment; never run a solver here.

Usage (from any working directory):
    python scripts/07_plot_walk_cost.py [walk-cost-20261118]

Input:  results/<run_id>/raw/timings.csv
Output: results/<run_id>/summary/summary.csv and three PDF/PNG figure pairs
        in presentation/report/figures/.

Each timing observation is the average call time within a calibrated batch.
Figures show medians across repeated batches; error bars show their IQR.
Parallel speedup uses the separately compiled serial C++ solver as baseline,
not the OpenMP solver requested to use one thread.
"""

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, ScalarFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BLUE = "#045073"
RED = "#EC5858"
GREEN = "#5FB2A8"
GRAY = "#777777"
GROUP_COLUMNS = ["sweep", "config_id", "backend", "N", "n", "M", "p", "query_x", "q"]


def summarise(raw):
    """Keep all configurations distinct and preserve the measured evidence."""
    required = set(GROUP_COLUMNS + ["rep", "t_solve", "actual_threads", "mean_steps"])
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Timing CSV is missing columns: {sorted(missing)}")
    if not np.isfinite(raw.t_solve).all() or (raw.t_solve <= 0).any():
        raise ValueError("Every recorded solve time must be positive and finite.")
    if raw.duplicated(GROUP_COLUMNS + ["rep"]).any():
        raise ValueError("Found duplicate repetition IDs within a configuration.")

    rows = []
    for key, group in raw.groupby(GROUP_COLUMNS, dropna=False, sort=False):
        row = dict(zip(GROUP_COLUMNS, key))
        row.update(
            median=group.t_solve.median(),
            Q1=group.t_solve.quantile(0.25),
            Q3=group.t_solve.quantile(0.75),
            repetitions=len(group),
            mean_steps=group.mean_steps.mean(),
            actual_threads_min=int(group.actual_threads.min()),
            actual_threads_max=int(group.actual_threads.max()),
        )
        for column in ["batch_A", "estimate", "se", "alive", "sum_steps"]:
            if column in group:
                row[f"{column}_mean"] = group[column].mean()
        rows.append(row)
    summary = pd.DataFrame(rows)

    parallel = summary[summary.sweep == "parallel"]
    serial = parallel[parallel.backend == "serial"]
    if len(serial) != 1:
        raise ValueError("The parallel comparison needs exactly one serial baseline.")
    # A speedup is meaningful only for the same mathematical task and path budget.
    for column in ["N", "n", "M", "query_x", "q"]:
        if parallel[column].nunique(dropna=False) != 1:
            raise ValueError(f"Parallel comparison has inconsistent {column}.")
    baseline = float(serial["median"].iloc[0])
    mask = summary.sweep == "parallel"
    summary.loc[mask, "speedup"] = baseline / summary.loc[mask, "median"]
    summary.loc[mask, "efficiency"] = summary.loc[mask, "speedup"] / summary.loc[mask, "p"]
    return summary


def time_errors(group):
    return np.vstack((group["median"] - group.Q1, group.Q3 - group["median"])) * 1000


def time_curve(ax, group, x_column, color=BLUE, label=None):
    ax.errorbar(
        group[x_column], group["median"] * 1000,
        yerr=time_errors(group), fmt="o-", color=color,
        lw=1.8, markersize=5, capsize=3, label=label,
    )


def powers_of_two_axis(ax, values):
    ax.set_xscale("log", base=2)
    ax.set_xticks(values, [str(int(value)) for value in values])
    ax.xaxis.set_minor_locator(NullLocator())


def require_sweep(summary, sweep, sort_by):
    group = summary[summary.sweep == sweep].sort_values(sort_by)
    if group.empty:
        raise ValueError(f"No measurements for sweep {sweep!r}.")
    if group[sort_by].duplicated().any():
        raise ValueError(f"Sweep {sweep!r} has multiple configurations at the same x value.")
    return group


def save_figure(figure, directory, name):
    figure.tight_layout(pad=0.8)
    figure.savefig(directory / f"{name}.pdf")
    figure.savefig(directory / f"{name}.png", dpi=220)
    plt.close(figure)


def plot_grid(summary, directory):
    time_grid = require_sweep(summary, "time_grid", "n")
    space_grid = require_sweep(summary, "space_grid", "N")
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 2.95))
    for ax, group, x_column, color in [
        (axes[0], time_grid, "n", BLUE),
        (axes[1], space_grid, "N", RED),
    ]:
        time_curve(ax, group, x_column, color)
        powers_of_two_axis(ax, group[x_column].to_numpy())
        ax.set_ylabel("求解耗时（毫秒）")
        ax.set_ylim(bottom=0)
    axes[0].set(
        title=rf"固定空间网格 $N={int(time_grid.N.iloc[0])}$",
        xlabel=r"时间步数 $n$",
    )
    axes[1].set(
        title=rf"固定时间网格 $n={int(space_grid.n.iloc[0])}$",
        xlabel=r"空间分段数 $N$",
    )
    save_figure(figure, directory, "walk_cost_grid")


def plot_coupled(summary, directory):
    group = require_sweep(summary, "coupled_grid", "N")
    figure, ax = plt.subplots(figsize=(6.9, 2.95))
    time_curve(ax, group, "N", BLUE, "实测耗时")
    reference = group["median"].iloc[0] * 1000 * (group.N / group.N.iloc[0]) ** 2
    ax.plot(group.N, reference, "--", color=GREEN, lw=1.7, label=r"二次增长参考 $\propto N^2$")
    powers_of_two_axis(ax, group.N.to_numpy())
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    for row in group.itertuples():
        ax.annotate(
            rf"$n={int(row.n)}$", (row.N, row.median * 1000),
            xytext=(4, -18), textcoords="offset points", fontsize=9,
        )
    ax.set(
        xlabel=r"空间分段数 $N$",
        ylabel="求解耗时（毫秒）",
        title=r"同时细化网格：$n=N^2/8$",
    )
    ax.margins(x=0.13, y=0.22)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("实测耗时"), labels.index(r"二次增长参考 $\propto N^2$")]
    ax.legend([handles[index] for index in order], [labels[index] for index in order],
              loc="upper left", fontsize=8.5)
    save_figure(figure, directory, "walk_cost_coupled")


def plot_parallel(summary, directory):
    group = summary[summary.sweep == "parallel"]
    serial = group[group.backend == "serial"]
    parallel = group[group.backend == "openmp"].sort_values("p")
    if parallel.empty or parallel.p.duplicated().any():
        raise ValueError("The parallel figure needs one OpenMP configuration per thread count.")
    group = pd.concat([serial, parallel], ignore_index=True)
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.05))
    labels = ["串行"] + [f"并行\n{int(p)} 线程" for p in parallel.p]
    colors = [GRAY] + [BLUE] * len(parallel)
    bars = axes[0].bar(
        np.arange(len(group)), group["median"] * 1000,
        yerr=time_errors(group), capsize=3,
        width=0.6, color=colors, alpha=0.95,
    )
    axes[0].bar_label(bars, fmt="%.1f", padding=4, fontsize=8.5)
    axes[0].set_xticks(np.arange(len(group)), labels)
    axes[0].set(title="同一求解任务的耗时", ylabel="求解耗时（毫秒）")
    axes[0].set_ylim(0, float(group.Q3.max()) * 1000 * 1.2)
    axes[0].grid(axis="x", visible=False)
    axes[0].set_axisbelow(True)

    axes[1].plot(parallel.p, parallel.speedup, "o-", color=RED, lw=1.8, label="实测加速比")
    axes[1].plot(parallel.p, parallel.p, "--", color=GREEN, lw=1.5, label="理想加速比")
    for row in parallel.itertuples():
        axes[1].annotate(
            f"{row.speedup:.2f}", (row.p, row.speedup),
            xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5,
        )
    axes[1].set(
        title=r"加速比 $S_p=T_{\mathrm{serial}}/T_p$",
        xlabel=r"并行线程数 $p$", ylabel="加速比",
        xticks=parallel.p.to_numpy(),
    )
    axes[1].set_ylim(0, max(float(parallel.p.max()), float(parallel.speedup.max())) * 1.16)
    axes[1].legend(loc="upper left", fontsize=8.5)
    save_figure(figure, directory, "walk_cost_parallel")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", nargs="?", default="walk-cost-20261118")
    args = parser.parse_args()
    run_directory = ROOT / "results" / args.run_id
    raw = pd.read_csv(run_directory / "raw" / "timings.csv")
    summary = summarise(raw)
    output_directory = run_directory / "summary"
    output_directory.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_directory / "summary.csv", index=False)
    figure_directory = ROOT / "presentation" / "report" / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": ["Microsoft YaHei", "DejaVu Sans"],
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.2,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
    })
    plot_grid(summary, figure_directory)
    plot_coupled(summary, figure_directory)
    plot_parallel(summary, figure_directory)
    print(summary.to_string(index=False))
    print(f"Figures written to {figure_directory}")


if __name__ == "__main__":
    main()
