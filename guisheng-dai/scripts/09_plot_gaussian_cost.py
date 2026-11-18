"""Plot recorded Gaussian direct-convolution experiments; never run a solver.

Usage (from any working directory):
    python path/to/guisheng-dai/scripts/09_plot_gaussian_cost.py

Inputs:
    results/openmp-final-20261117/summary/kernel.csv (existing E3 accuracy)
    results/gaussian-cost-20261118/summary/summary.csv (new cost experiment)
Outputs: PDF/PNG pairs in presentation/report/figures/.

All time curves show measured medians and interquartile ranges. The total time
is a separately measured preparation-plus-solve call, not a sum of medians.
Parallel speedup uses the same OpenMP implementation with one thread, T1/Tp.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BLUE = "#045073"
RED = "#EC5858"
GREEN = "#5FB2A8"
GRAY = "#777777"
C_VALUES = [1, 2, 3, 4, 6, 8]
P_VALUES = [1, 2, 4, 8]
STAGES = ["t_input", "t_solve", "t_reused_new_input"]


def require_columns(frame, columns, source):
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")


def validate_data(accuracy, cost):
    require_columns(accuracy, ["N", "c", "K", "W", "RMSE"], "E3 accuracy")
    require_columns(
        cost,
        ["sweep", "config_id", "N", "c", "p", "q", "K", "W", "actual_threads",
         "speedup_solve", "speedup_total"]
        + [f"{stage}_{stat}" for stage in STAGES for stat in ["median", "Q1", "Q3"]],
        "Cost summary",
    )
    if cost.config_id.duplicated().any():
        raise ValueError("Cost summary contains duplicate configuration IDs.")
    if not np.equal(cost.actual_threads, cost.p).all():
        raise ValueError("Requested and measured OpenMP thread counts differ.")
    if not np.equal(cost.q, cost.N + 1).all():
        raise ValueError("A timing configuration does not solve all N+1 grid points.")
    if not np.equal(cost.W, 2 * cost.K + 1).all():
        raise ValueError("Kernel widths do not satisfy W = 2K + 1.")
    for stage in STAGES:
        values = cost[[f"{stage}_Q1", f"{stage}_median", f"{stage}_Q3"]].to_numpy()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{stage}: invalid timing statistics.")
        if (values[:, 0] > values[:, 1]).any() or (values[:, 1] > values[:, 2]).any():
            raise ValueError(f"{stage}: quartiles do not bracket the median.")
    if (cost.t_solve_median <= 0).any() or (cost.t_reused_new_input_median <= 0).any():
        raise ValueError("Solve and total medians must be positive.")

    kernel = select_sweep(cost, "kernel_width", "c", C_VALUES)
    if not ((kernel.N == 512) & (kernel.p == 2)).all():
        raise ValueError("Kernel-width sweep must hold N=512 and p=2 fixed.")
    accuracy = accuracy[(accuracy.N == 512) & accuracy.c.isin(C_VALUES)].sort_values("c")
    if accuracy.c.tolist() != C_VALUES:
        raise ValueError("E3 accuracy must have one N=512 result for each plotted c.")
    if not np.isfinite(accuracy.RMSE).all() or (accuracy.RMSE <= 0).any():
        raise ValueError("E3 RMSE values must be positive and finite for the log axis.")
    for column in ["K", "W"]:
        if not np.equal(accuracy[column].to_numpy(), kernel[column].to_numpy()).all():
            raise ValueError(f"Old accuracy and new cost experiments use different {column}.")

    for sweep, expected_n in [("parallel_large", 8192), ("parallel_small", 512)]:
        group = select_sweep(cost, sweep, "p", P_VALUES)
        for column, expected in [("N", expected_n), ("c", 6)]:
            if not (group[column] == expected).all():
                raise ValueError(f"{sweep}: inconsistent {column}.")
        for column in ["q", "K", "W"]:
            if group[column].nunique() != 1:
                raise ValueError(f"{sweep}: the mathematical task changes ({column}).")
        for stage, speedup_column in [
            ("t_solve", "speedup_solve"), ("t_reused_new_input", "speedup_total")
        ]:
            expected = group[f"{stage}_median"].iloc[0] / group[f"{stage}_median"]
            if not np.allclose(group[speedup_column], expected, rtol=1e-8, atol=1e-10):
                raise ValueError(f"{sweep}: {speedup_column} does not use the one-thread baseline.")
    return accuracy


def select_sweep(cost, sweep, x_column, expected_values):
    group = cost[cost.sweep == sweep].sort_values(x_column)
    if group[x_column].tolist() != expected_values:
        raise ValueError(f"{sweep}: expected exactly {expected_values} for {x_column}.")
    return group


def time_errors(group, stage):
    median = group[f"{stage}_median"]
    return np.vstack((median - group[f"{stage}_Q1"], group[f"{stage}_Q3"] - median)) * 1000


def save_figure(figure, directory, name):
    figure.tight_layout(pad=0.8, w_pad=1.4)
    figure.savefig(directory / f"{name}.pdf")
    figure.savefig(directory / f"{name}.png", dpi=220)
    plt.close(figure)


def plot_kernel_tradeoff(accuracy, cost, directory):
    kernel = select_sweep(cost, "kernel_width", "c", C_VALUES)
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 2.8))
    axes[0].semilogy(accuracy.c, accuracy.RMSE, "o-", color=BLUE, lw=1.8, ms=5)
    axes[0].set(
        title="核宽与精度（E3）", xlabel=r"覆盖倍数 $c$", ylabel="内部网格点 RMSE",
        xticks=C_VALUES,
    )
    axes[0].set_yticks([1e-2, 1e-6, 1e-10, 1e-14])
    reference = accuracy[accuracy.c == 8].iloc[0]
    axes[0].annotate(
        r"$c=8$：宽核参考", (reference.c, reference.RMSE),
        xytext=(0.48, 0.11), textcoords="axes fraction", ha="center", va="center",
        fontsize=9.5, arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.8},
    )
    axes[0].set_ylim(1e-17, 0.3)
    axes[0].grid(which="minor", visible=False)

    for stage, color, label, marker in [
        ("t_solve", BLUE, "求解（输入已准备）", "o"),
        ("t_reused_new_input", RED, "准备 + 求解", "s"),
    ]:
        axes[1].errorbar(
            kernel.c, kernel[f"{stage}_median"] * 1000,
            yerr=time_errors(kernel, stage), fmt=f"{marker}-", color=color,
            lw=1.8, ms=5, capsize=3, label=label,
        )
    axes[1].set(
        title=r"核宽与耗时（$N=512$，2 线程）",
        xlabel=r"覆盖倍数 $c$", ylabel="耗时（毫秒）", xticks=C_VALUES,
    )
    axes[1].set_ylim(0, kernel.t_reused_new_input_Q3.max() * 1000 * 1.5)
    axes[1].legend(loc="upper right", fontsize=9.5)
    save_figure(figure, directory, "gaussian_kernel_tradeoff")


def plot_parallel(cost, directory):
    large = select_sweep(cost, "parallel_large", "p", P_VALUES)
    small = select_sweep(cost, "parallel_small", "p", P_VALUES)
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 2.8))
    bars = axes[0].bar(
        np.arange(len(large)), large.t_solve_median * 1000,
        yerr=time_errors(large, "t_solve"), capsize=3, width=0.58, color=BLUE,
    )
    axes[0].bar_label(bars, fmt="%.2f", padding=4, fontsize=9.5)
    axes[0].set_xticks(np.arange(len(large)), [str(value) for value in P_VALUES])
    axes[0].set(
        title=r"整条曲线的求解耗时（$N=8192$）",
        xlabel=r"OpenMP 线程数 $p$", ylabel="耗时（毫秒）",
        ylim=(0, large.t_solve_Q3.max() * 1000 * 1.2),
    )
    axes[0].grid(axis="x", visible=False)
    axes[0].set_axisbelow(True)

    axes[1].plot(P_VALUES, P_VALUES, "--", color=GREEN, lw=1.5, label="理想加速比")
    axes[1].plot(large.p, large.speedup_solve, "o-", color=BLUE, lw=1.8, ms=5,
                 label=r"$N=8192$")
    axes[1].plot(small.p, small.speedup_solve, "s:", color=GRAY, lw=1.1, ms=4,
                 label=r"$N=512$")
    for row in large.itertuples():
        axes[1].annotate(
            f"{row.speedup_solve:.2f}", (row.p, row.speedup_solve),
            xytext=(0, -16 if row.p == 8 else 8), textcoords="offset points",
            ha="center", fontsize=9.5,
        )
    axes[1].set(
        title=r"加速比 $S_p=T_1/T_p$（$c=6$）",
        xlabel=r"OpenMP 线程数 $p$", ylabel="加速比", xticks=P_VALUES,
        ylim=(0, max(8, large.speedup_solve.max(), small.speedup_solve.max()) * 1.15),
    )
    axes[1].legend(loc="upper left", fontsize=9.5)
    save_figure(figure, directory, "gaussian_curve_parallel")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", nargs="?", default="gaussian-cost-20261118")
    parser.add_argument("--accuracy-run", default="openmp-final-20261117")
    args = parser.parse_args()
    cost_path = ROOT / "results" / args.run_id / "summary" / "summary.csv"
    accuracy_path = ROOT / "results" / args.accuracy_run / "summary" / "kernel.csv"
    cost = pd.read_csv(cost_path)
    accuracy = validate_data(pd.read_csv(accuracy_path), cost)
    directory = ROOT / "presentation" / "report" / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": ["Microsoft YaHei", "DejaVu Sans"],
        "font.size": 10.5,
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
    plot_kernel_tradeoff(accuracy, cost, directory)
    plot_parallel(cost, directory)
    print(f"Accuracy source: {accuracy_path}")
    print(accuracy[["N", "c", "K", "W", "RMSE"]].to_string(index=False))
    print(f"Cost source: {cost_path}")
    print(cost[["sweep", "N", "c", "p", "t_solve_median", "t_reused_new_input_median",
                "speedup_solve", "speedup_total"]].to_string(index=False))
    print(f"Figures written to {directory}")


if __name__ == "__main__":
    main()
