"""从同一轮训练保存的终态预测和独立评价绘制两张报告用图。"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np


METHOD_DIR = Path(__file__).resolve().parents[1]
CONSTRAINTS = {"soft": "软约束", "hard": "硬约束"}
BLUE = "#285784"
RUST = "#B45827"
TEAL = "#18766B"
STAGES = (
    ("adam_baseline", "原始 Adam", BLUE, "o", 0),
    ("adam_continued", "继续 Adam", RUST, "s", -0.16),
    ("adam_then_lbfgs", "改用 L-BFGS", TEAL, "^", 0.16),
)


def configure_style():
    word_font = Path("/Applications/Microsoft Word.app/Contents/Resources/DFonts/Simsun.ttc")
    if word_font.is_file():
        font_manager.fontManager.addfont(str(word_font))
    plt.rcParams.update({
        "font.family": ["Times New Roman", "SimSun"],
        "font.size": 10,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "axes.linewidth": 0.7,
        "axes.labelcolor": "#202832",
        "xtick.color": "#263340",
        "ytick.color": "#263340",
    })


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def panel_style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", which="major", color="#DADFE2", linewidth=0.6)
    axis.tick_params(which="major", length=3.3, width=0.7)
    axis.tick_params(which="minor", length=1.8, width=0.4)
    axis.set_axisbelow(True)


def shared_legend(fig, axis):
    fig.legend(*axis.get_legend_handles_labels(), loc="upper center", ncol=3,
               bbox_to_anchor=(0.52, 0.994), frameon=False, handlelength=2.4,
               columnspacing=1.65, handletextpad=0.45)


def save(fig, target):
    fig.savefig(target, dpi=320, facecolor="white")
    plt.close(fig)
    print(target)


def plot_terminal(rows, target):
    fig, (curves, errors) = plt.subplots(1, 2, figsize=(16.5 / 2.54, 7.7 / 2.54))
    fig.subplots_adjust(left=0.105, right=0.979, bottom=0.225, top=0.824, wspace=0.45)
    for constraint, label in CONSTRAINTS.items():
        series = sorted((r for r in rows if r["constraint"] == constraint
                         and int(r["seed"]) == 7 and r["stage"] == "adam_baseline"),
                        key=lambda row: float(row["x"]))
        if not series:
            raise ValueError(f"缺少种子 7 的 {constraint} 初始训练结果")
        x = np.array([float(r["x"]) for r in series])
        pred = np.array([float(r["prediction"]) for r in series])
        exact = np.array([float(r["exact"]) for r in series])
        if constraint == "soft":
            curves.plot(x, exact, color="#2E353D", linewidth=2.2, label="解析解")
            curves.plot(x[-1], pred[-1], marker="o", color=RUST, markersize=3.3)
            curves.annotate(f"右端 {pred[-1]:.4f}".replace("-", "−"),
                            xy=(x[-1], pred[-1]), xytext=(0.61, 0.17),
                            textcoords="axes fraction", fontsize=8.8, ha="center",
                            arrowprops=dict(arrowstyle="-", lw=0.7, color=RUST))
        color, linestyle = (RUST, "-.") if constraint == "soft" else (TEAL, "--")
        curves.plot(x, pred, color=color, linestyle=linestyle, linewidth=1.4, label=label)
        # 已知端点不计入内部误差；取对数时，仅绘制大于零的误差。
        error = np.abs(pred[1:-1] - exact[1:-1])
        errors.plot(x[1:-1], np.ma.masked_less_equal(error, 0),
                    color=color, linestyle=linestyle, linewidth=1.5)
    curves.axhline(0, color="#90989D", linewidth=0.6, zorder=0)
    curves.set_ylabel("终态信号强度")
    errors.set_ylabel("内部节点绝对误差")
    errors.set_yscale("log")
    for axis in (curves, errors):
        panel_style(axis)
        axis.set_xlim(-0.018, 1.022)
        axis.xaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1]))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    fig.supxlabel("空间位置 x", y=0.042, fontsize=10)
    shared_legend(fig, curves)
    save(fig, target)


def plot_optimizers(rows, target):
    values = {(int(r["seed"]), r["constraint"], r["stage"]): float(r["terminal_rmse"])
              for r in rows}
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(16.5 / 2.54, 7.7 / 2.54))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.78, wspace=0.19)
    seeds = sorted({int(row["seed"]) for row in rows})
    groups = np.arange(len(seeds), dtype=float)
    for axis, (constraint, label) in zip(axes, CONSTRAINTS.items()):
        panel_style(axis)
        base = [values[seed, constraint, "adam_baseline"] for seed in seeds]
        for stage, stage_label, color, marker, offset in STAGES:
            series = [values[seed, constraint, stage] for seed in seeds]
            if offset:
                for i, (start, end) in enumerate(zip(base, series)):
                    axis.plot([groups[i], groups[i] + offset], [start, end],
                              color="#9BA5AB", linewidth=0.85, zorder=1)
            axis.plot(groups + offset, series, linestyle="none", marker=marker,
                      markersize=5.6, markeredgewidth=1.35, markerfacecolor="white",
                      markeredgecolor=color, label=stage_label, zorder=2)
        axis.set_title(label, fontsize=10.8, pad=5)
        axis.set_xlim(-0.43, len(seeds) - 0.57)
        axis.set_yscale("log")
        axis.set_xticks(groups, labels=[str(s) for s in seeds])
    axes[0].set_ylabel("终态内部 RMSE")
    fig.supxlabel("初始化种子", y=0.044, fontsize=10)
    shared_legend(fig, axes[0])
    save(fig, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=METHOD_DIR / "results",
                        help="包含 metrics.csv 和 terminal_seed7.csv 的实验目录")
    results = parser.parse_args().results
    configure_style()
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_terminal(read_rows(results / "terminal_seed7.csv"), figures / "baseline_terminal.png")
    plot_optimizers(read_rows(results / "metrics.csv"), figures / "optimizer_comparison.png")


if __name__ == "__main__":
    main()
