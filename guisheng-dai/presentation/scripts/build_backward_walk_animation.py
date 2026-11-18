"""Render illustrative backward-walk frames for the Beamer presentation.

Uses the same three transition probabilities and absorbing endpoints as
make_problem() in R/problem.R and src/random_walk.cpp. This records paths only;
it does not replace the existing solver or produce performance measurements.
Run from any directory with Python, NumPy and Matplotlib installed.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

A, D, T = 0.2, 0.05, 0.5
H, DT, X0 = 0.05, 0.0125, 0.5
N_PATHS, BASE_SEED = 16, 20261117
N_STEPS, N_CELLS = round(T / DT), round(1 / H)
P = np.array([D * DT / H**2 - A * DT / (2 * H),
              1 - 2 * D * DT / H**2,
              D * DT / H**2 + A * DT / (2 * H)])
OUT = (Path(__file__).resolve().parents[1] / "report" / "figures"
       / "backward_walk_animation.pdf")


def simulate(seed):
    rng = np.random.default_rng(seed)
    states = np.full((N_PATHS, N_STEPS + 1), -1, dtype=int)
    states[:, 0] = round(X0 / H)
    deaths = np.full(N_PATHS, N_STEPS + 1, dtype=int)
    for k in range(1, N_STEPS + 1):
        live = deaths > k - 1
        jumps = rng.choice([-1, 0, 1], N_PATHS, p=P)
        states[live, k] = states[live, k - 1] + jumps[live]
        killed = live & ((states[:, k] == 0) | (states[:, k] == N_CELLS))
        deaths[killed] = k
    return states, deaths


def main():
    assert np.isclose(P.sum(), 1) and np.all(P >= 0)
    assert np.isclose(N_STEPS * DT, T) and np.isclose(N_CELLS * H, 1)
    assert np.allclose(P, [.225, .5, .275])
    # Select a deterministic illustrative realization with both outcomes.
    for seed in range(BASE_SEED, BASE_SEED + 10000):
        states, deaths = simulate(seed)
        n_absorbed = int(np.count_nonzero(deaths <= N_STEPS))
        if 0 < n_absorbed < N_PATHS:
            break
    else:
        raise RuntimeError("No mixed absorbing/surviving realization found")

    survivors = deaths > N_STEPS
    terminal = np.zeros(N_PATHS)
    terminal_x = H * states[survivors, -1]
    terminal[survivors] = np.exp(-2 * terminal_x) * np.sin(np.pi * terminal_x)
    for j in range(N_PATHS):
        valid = states[j] >= 0
        assert np.all(np.isin(np.diff(states[j, valid]), [-1, 0, 1]))
        if deaths[j] <= N_STEPS:
            k = deaths[j]
            assert states[j, k] in (0, N_CELLS)
            assert np.all(states[j, k + 1:] == -1)
            assert terminal[j] == 0
        else:
            assert np.all((states[j] > 0) & (states[j] < N_CELLS))

    highlights = list(np.flatnonzero(survivors)[:3])
    highlights += list(np.flatnonzero(~survivors)[:2])
    colors = dict(zip(highlights, ["#1261a0", "#007f73", "#7551a6",
                                   "#ad5a17", "#427d45"]))
    times = T - DT * np.arange(N_STEPS + 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "pdf.fonttype": 42})
    with PdfPages(OUT, metadata={
        "Title": "Illustrative backward random walks",
        "Subject": "Absorbing paths for the drift-diffusion difference scheme",
        "Keywords": f"seed={seed}; a={A}; D={D}; h={H}; dt={DT}",
    }) as pdf:
        for k in range(N_STEPS + 1):
            fig = plt.figure(figsize=(6.4 / 2.54, 4.4 / 2.54))
            ax = fig.add_axes([.16, .26, .80, .62])
            ax.set(xlim=(-.015, 1.015), ylim=(-.015, T + .015),
                   xticks=[0, .5, 1], yticks=[0, .25, .5])
            ax.set_xlabel("space x", labelpad=2)
            ax.set_ylabel("time t", labelpad=2)
            for name in ("top", "right"):
                ax.spines[name].set_visible(False)
            for name in ("left", "bottom"):
                ax.spines[name].set_color("#8b959e")
            ax.tick_params(length=2, pad=2)
            for x in (0, 1):
                ax.axvline(x, color="#c8cdd2", lw=1)
            ax.axhline(0, color="#326fa8", lw=1.5)
            ax.axhline(times[k], color="#aeb7c0", ls=":", lw=.65, zorder=0)
            ax.text(.015, .012, "initial data f0(x)", transform=ax.transAxes,
                    color="#326fa8", fontsize=6, va="bottom")
            for j in range(N_PATHS):
                trajectory = states[j, :k + 1]
                valid = trajectory >= 0
                absorbed = deaths[j] <= k
                highlighted = j in colors
                color = "#9aa1a8" if absorbed else colors.get(j, "#6e9fc2")
                ax.plot(H * trajectory[valid], times[:k + 1][valid],
                        color=color, lw=1.0 if highlighted else .65,
                        alpha=.9 if highlighted else .5,
                        zorder=3 if highlighted else 2)
                if absorbed:
                    death = deaths[j]
                    ax.scatter(H * states[j, death], times[death], marker="x",
                               color="#b34b42", s=14, lw=.8, zorder=5)
                else:
                    ax.scatter(H * states[j, k], times[k],
                               s=9 if highlighted else 5, color=color,
                               edgecolors="none", zorder=4)
            ax.scatter(X0, T, s=16, color="#202b35", zorder=6)
            ax.annotate("(x, T)", xy=(X0, T), xytext=(5, -10),
                        textcoords="offset points", fontsize=6)
            alive_count = int(np.count_nonzero(deaths > k))
            fig.text(.16, .97, "Backward paths", fontsize=7.5,
                     weight="bold", ha="left", va="top")
            fig.text(.96, .97, f"k={k:02d}/{N_STEPS}", fontsize=6.2,
                     ha="right", va="top", color="#435364")
            fig.text(.16, .035, f"live: {alive_count}/{N_PATHS}    "
                     f"absorbed: {N_PATHS - alive_count}",
                     fontsize=6.2, color="#52616e")
            pdf.savefig(fig)
            plt.close(fig)

    metadata = {
        "generator": Path(__file__).name,
        "purpose": "Illustrative discrete paths, not a convergence experiment",
        "frame_count": N_STEPS + 1, "seed": seed,
        "a": A, "D": D, "T": T, "h": H, "tau": DT, "x0": X0,
        "probabilities": dict(zip(["p_minus", "p_zero", "p_plus"], P.tolist())),
        "n_paths": N_PATHS, "absorbed_paths": n_absorbed,
        "surviving_paths": N_PATHS - n_absorbed,
        "absorption_steps": [int(k) if k <= N_STEPS else None for k in deaths],
        "terminal_contributions": terminal.tolist(),
        "terminal_function": "exp(-2*x)*sin(pi*x); zero after boundary absorption",
        "checks": ["nonnegative normalized probabilities", "grid/time dimensions",
                   "nearest-neighbour increments", "boundary-only absorption",
                   "absorbed paths never resume", "absorbed contributions zero"],
    }
    OUT.with_suffix(".json").write_text(json.dumps(metadata, indent=2),
                                       encoding="utf-8")
    print(json.dumps({"pdf": str(OUT), "frames": N_STEPS + 1,
                      "seed": seed, "absorbed": n_absorbed,
                      "surviving": N_PATHS - n_absorbed}, indent=2))


if __name__ == "__main__":
    main()
