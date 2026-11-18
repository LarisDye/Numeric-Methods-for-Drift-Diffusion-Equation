"""根据本轮 Rcpp 网格实验的汇总表绘制精度与耗时图。"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "explicit_euler": ("显式 Euler", "#285784", "o", "-"),
    "implicit_euler": ("隐式 Euler", "#B45827", "s", "--"),
    "crank_nicolson": ("Crank–Nicolson", "#18766B", "^", "-."),
}


def configure_style():
    # macOS 上，宋体可能只安装在 Word 的字体目录中。
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


def main():
    configure_style()
    source = ROOT / "results" / "grid_summary.csv"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    fig, (accuracy, runtime) = plt.subplots(1, 2, figsize=(16.5 / 2.54, 8.1 / 2.54))
    fig.subplots_adjust(left=0.105, right=0.98, top=0.81, bottom=0.22, wspace=0.39)
    for name, (label, color, marker, line) in METHODS.items():
        series = sorted((row for row in rows if row["method"] == name), key=lambda r: int(r["N"]))
        grids = [int(row["N"]) for row in series]
        if grids != [16, 32, 64, 128]:
            raise ValueError(f"{source}: {name} 应包含 16、32、64、128 四组网格")
        rmse = [float(row["rmse"]) for row in series]
        times = [float(row["median_ms"]) for row in series]
        lower = [t - float(row["q25_ms"]) for row, t in zip(series, times)]
        upper = [float(row["q75_ms"]) - t for row, t in zip(series, times)]
        style = dict(color=color, marker=marker, linestyle=line,
                     markersize=4.6, markerfacecolor="white", markeredgewidth=1.2,
                     linewidth=1.6)
        accuracy.plot(grids, rmse, label=label, **style)
        runtime.errorbar(grids, times, yerr=[lower, upper], capsize=3.0,
                         elinewidth=1.15, **style)

    for axis in (accuracy, runtime):
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlim(14, 147)
        axis.xaxis.set_major_locator(FixedLocator([16, 32, 64, 128]))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
        axis.xaxis.set_minor_locator(NullLocator())
        axis.grid(axis="y", which="major", color="#DADFE2", linewidth=0.65)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(which="major", length=3.5, width=0.7)
        axis.tick_params(axis="y", which="minor", length=1.7, width=0.4)

    accuracy.set_ylabel("终态 RMSE")
    runtime.set_ylabel("一次求解耗时 (ms)")
    fig.supxlabel("空间小区间数 N", fontsize=10, y=0.045)
    fig.legend(*accuracy.get_legend_handles_labels(), loc="upper center", ncol=3,
               bbox_to_anchor=(0.52, 0.98), frameon=False, handlelength=2.7,
               columnspacing=1.8, handletextpad=0.65)

    target = ROOT / "results" / "figures" / "grid_comparison.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=320, facecolor="white")
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
