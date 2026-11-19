"""Draw the report figures from the reference profiles and measured errors."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import pandas as pd

from compare import CASES, FD_METHODS, RESULTS, TIMES


NAMES = dict(baseline="原光滑基准", center="中央窄峰", near="近左边界窄峰", compact="中央有限光滑峰")
METHODS = dict(zip((*FD_METHODS, "pinn"), ("显式 Euler", "隐式 Euler", "Crank–Nicolson", "PINN")))
COLORS = ("#1D577E", "#AD592C", "#27766C", "#743F73")
MARKERS = ("o", "s", "^", "D")
STYLES = ("-", "--", "-.", ":")


def configure():
    from matplotlib import font_manager
    word_font = Path("/Applications/Microsoft Word.app/Contents/Resources/DFonts/Simsun.ttc")
    if word_font.is_file():
        font_manager.fontManager.addfont(str(word_font))
    plt.rcParams.update({"font.family": ["Times New Roman", "SimSun"], "font.size": 10,
                         "axes.labelcolor": "#222C32", "axes.linewidth": 0.7,
                         "mathtext.fontset": "stix", "axes.unicode_minus": False})


def panels():
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6.50, 3.72))
    fig.subplots_adjust(left=0.108, bottom=0.145, right=0.977, top=0.838,
                        wspace=0.13, hspace=0.38)
    for ax, case in zip(axes.flat, CASES):
        ax.set_title(NAMES[case], fontsize=10.4, pad=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linewidth=0.5, color="#D3DADE")
        ax.tick_params(length=3, labelsize=8.7)
    return fig, axes


def finish(fig, ax, name):
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=4,
               frameon=False, fontsize=9.3, columnspacing=1.1, handlelength=2.1,
               bbox_to_anchor=(0.52, 1.007), handletextpad=0.4)
    target = RESULTS / "figures" / name
    fig.savefig(target, dpi=300, facecolor="white")
    plt.close(fig)


def draw_shapes():
    initial = pd.read_csv(RESULTS / "reference/initial_grids.csv")
    ref = pd.read_csv(RESULTS / "reference/profiles.csv")
    fig, axes = panels()
    for ax, case in zip(axes.flat, CASES):
        for i, t in enumerate(TIMES):
            if t == 0:
                curve = initial[(initial["case"] == case) & (initial["N"] == 128)]
                field = "initial"
            else:
                curve = ref[(ref["case"] == case) & (ref["t"] == t)]
                field = "reference"
            curve = curve.sort_values("x")
            ax.plot(curve.x, curve[field], color=COLORS[i], linestyle=STYLES[i],
                    lw=1.5, label=f"t = {t:g}")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "0.25", "0.5", "0.75", "1"])
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.012, 0.49)
    fig.supylabel("信号强度", x=0.013, fontsize=10)
    fig.supxlabel("空间位置 x", y=0.012, fontsize=10)
    finish(fig, axes.flat[0], "reference_profiles.png")


def draw_errors():
    data = pd.read_csv(RESULTS / "summary.csv")
    data = data[(data["t"] > 0) & (data.N.isna() | data.N.eq(64))]
    fig, axes = panels()
    for ax, case in zip(axes.flat, CASES):
        for index, (method, name) in enumerate(METHODS.items()):
            series = data[(data["case"] == case) & (data["method"] == method)].sort_values("t")
            if len(series) != 3:
                raise ValueError(f"Incomplete comparison for {case}, {method}")
            opts = dict(color=COLORS[index], marker=MARKERS[index], markersize=4.1,
                        markerfacecolor="white", linewidth=1.15, linestyle=STYLES[index])
            ax.plot(range(3), series.rmse, label=name, **opts)
            if method == "pinn":
                limits = [series.rmse - series.rmse_min, series.rmse_max - series.rmse]
                ax.errorbar(range(3), series.rmse, yerr=limits, fmt="none",
                            ecolor=COLORS[index], elinewidth=0.95, capsize=2.0, zorder=1)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([0, 1, 2], labels=["1/32", "1/8", "1/2"])
        ax.set_xlim(-0.15, 2.15)
    fig.supylabel("内部节点 RMSE", x=0.005, fontsize=10)
    fig.supxlabel("评价时刻 t", y=0.012, fontsize=10)
    finish(fig, axes.flat[0], "error_by_time.png")


if __name__ == "__main__":
    configure()
    (RESULTS / "figures").mkdir(exist_ok=True)
    draw_shapes()
    draw_errors()
