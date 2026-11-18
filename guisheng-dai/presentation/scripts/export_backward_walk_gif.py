"""Export the existing illustrative paths as a looping PowerPoint GIF.

The existing PDF and its JSON remain unchanged.  Reuses the original simulator,
seed and parameters; this animation illustrates the difference scheme and is
not a numerical accuracy experiment.  Run with Python, NumPy, Matplotlib and
Pillow.  The PNG poster records the final frame for static previews.
"""

from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageSequence

import build_backward_walk_animation as source


OUT = source.OUT.with_suffix(".gif")
POSTER = source.OUT.with_name("backward_walk_animation_poster.png")
MANIFEST = source.OUT.with_name("backward_walk_gif_manifest.json")
WIDTH, HEIGHT, DPI = 900, 620, 150


def render(k, states, deaths):
    times = source.T - source.DT * np.arange(source.N_STEPS + 1)
    survivors = deaths > source.N_STEPS
    highlights = list(np.flatnonzero(survivors)[:3])
    colors = dict(zip(highlights, ["#1261a0", "#008473", "#7551a6"]))
    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI,
                     facecolor="white")
    ax = fig.add_axes([.14, .23, .82, .62])
    ax.set(xlim=(-.025, 1.025), ylim=(-.015, source.T + .035),
           xticks=[0, .5, 1], yticks=[0, .25, .5])
    ax.set_xlabel("空间 $x$", labelpad=4)
    ax.set_ylabel("时间 $t$", labelpad=5)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color("#8b959e")
    ax.tick_params(length=3, pad=3)
    for x in (0, 1):
        ax.axvline(x, color="#c8cdd2", lw=1.4)
    ax.axhline(0, color="#326fa8", lw=2)
    ax.axhline(times[k], color="#aeb7c0", ls=":", lw=1, zorder=0)
    ax.text(.015, .018, "初值 $f_0(x)$", transform=ax.transAxes,
            color="#326fa8", fontsize=11, va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    for j in range(source.N_PATHS):
        trajectory = states[j, :k + 1]
        valid = trajectory >= 0
        absorbed = deaths[j] <= k
        highlighted = j in colors
        color = "#b9bfc5" if absorbed else colors.get(j, "#719fc2")
        ax.plot(source.H * trajectory[valid], times[:k + 1][valid],
                color=color, lw=1.9 if highlighted else 1.15,
                alpha=1 if highlighted else .6,
                zorder=3 if highlighted else 2)
        if absorbed:
            death = deaths[j]
            ax.scatter(source.H * states[j, death], times[death],
                       marker="x", color="#b34b42", s=65, lw=1.8, zorder=6)
        else:
            ax.scatter(source.H * states[j, k], times[k],
                       s=30 if highlighted else 18, color=color,
                       edgecolors="none", zorder=4)
    ax.scatter(source.X0, source.T, s=47, color="#202b35", zorder=7)
    ax.annotate("$(x_i,T)$", xy=(source.X0, source.T), xytext=(7, 8),
                textcoords="offset points", fontsize=11)
    alive = int(np.count_nonzero(deaths > k))
    fig.text(.14, .96, "从查询点回溯至初始时刻", fontsize=14,
             weight="bold", ha="left", va="top", color="#263847")
    fig.text(.96, .96, f"{k:02d} / {source.N_STEPS} 步", fontsize=11,
             ha="right", va="top", color="#435364")
    fig.text(.14, .075, f"触边吸收：{source.N_PATHS - alive} 条（贡献为 0）",
             fontsize=11, color="#a34e43")
    tail = (f"到达初值线：{alive} 条，取 $f_0(X_n)$"
            if k == source.N_STEPS else f"继续回溯：{alive} 条")
    fig.text(.14, .025, tail, fontsize=11, color="#326fa8")
    fig.canvas.draw()
    result = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)
    return result


def main():
    original = json.loads(source.OUT.with_suffix(".json").read_text(encoding="utf-8"))
    states, deaths = source.simulate(original["seed"])
    assert original["frame_count"] == source.N_STEPS + 1
    assert np.isclose(source.P.sum(), 1) and np.all(source.P >= 0)
    assert np.allclose(source.P, [original["probabilities"][name]
                                for name in ["p_minus", "p_zero", "p_plus"]])
    assert np.all(states[:, 0] == round(source.X0 / source.H))
    assert np.isclose(source.N_STEPS * source.DT, source.T)
    assert np.isclose(source.N_CELLS * source.H, 1)
    terminal = np.zeros(source.N_PATHS)
    for j in range(source.N_PATHS):
        valid = states[j] >= 0
        assert np.all(np.isin(np.diff(states[j, valid]), [-1, 0, 1]))
        expected_death = original["absorption_steps"][j]
        if deaths[j] <= source.N_STEPS:
            assert deaths[j] == expected_death
            assert states[j, deaths[j]] in (0, source.N_CELLS)
            assert np.all(states[j, deaths[j] + 1:] == -1)
        else:
            assert expected_death is None
            assert np.all((states[j] > 0) & (states[j] < source.N_CELLS))
            x = source.H * states[j, -1]
            terminal[j] = np.exp(-2 * x) * np.sin(np.pi * x)
    assert np.allclose(terminal, original["terminal_contributions"])
    for candidate in [Path("C:/Windows/Fonts/msyh.ttc"),
                      Path("C:/Windows/Fonts/simhei.ttf")]:
        if candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
            break
    else:
        raise RuntimeError("A Chinese font is required; configure a local CJK font.")
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "axes.unicode_minus": False})
    frames = [render(k, states, deaths) for k in range(source.N_STEPS + 1)]
    frames[-1].save(POSTER)
    # A shared palette avoids flicker in the paths and text during GIF playback.
    palette = frames[-1].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE)
               for frame in frames]
    durations = [1000] + [280] * (source.N_STEPS - 1) + [2400]
    indexed[0].save(OUT, save_all=True, append_images=indexed[1:],
                    duration=durations, loop=0, disposal=2, optimize=False)
    with Image.open(OUT) as exported:
        assert exported.n_frames == source.N_STEPS + 1
        assert exported.info["loop"] == 0
        assert exported.size == (WIDTH, HEIGHT)
        hashes = []
        actual_durations = []
        for frame in ImageSequence.Iterator(exported):
            hashes.append(sha256(frame.convert("RGB").tobytes()).hexdigest())
            actual_durations.append(frame.info["duration"])
        assert len(set(hashes)) == len(frames)
        assert actual_durations == durations
    manifest = {
        "generator": Path(__file__).name, "source_generator": source.__file__,
        "source_metadata": source.OUT.with_suffix(".json").name,
        "gif": OUT.name, "poster": POSTER.name,
        "size_px": [WIDTH, HEIGHT], "frame_count": len(frames),
        "loop": "infinite", "duration_ms": sum(durations),
        "frame_durations_ms": durations, "seed": original["seed"],
        "absorbed_paths": original["absorbed_paths"],
        "surviving_paths": original["surviving_paths"],
        "checks": ["all frames differ", "infinite loop", "exact dimensions and timing",
                   "same seed, paths and payoffs as original animation",
                   "nearest-neighbour jumps", "boundary-only absorption",
                   "absorbed paths stop and contribute zero",
                   "unabsorbed paths reach t=0"]}
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
