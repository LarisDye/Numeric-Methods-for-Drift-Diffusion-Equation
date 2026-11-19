"""输出四种初值和有限区间零边界的独立谱参照，并核验数值精度。"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import roots_legendre

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from profiles import (
    AMPLITUDE, BASELINE_PEAK_X, BETA, CASES, CENTER_HALF_WIDTH, COMPACT_RADIUS,
    DIFFUSION, DRIFT, EVAL_TIMES, PEAK_SCALES, PEAK_X, WIDTH, initial_numpy,
)

KAPPA = DRIFT**2 / (4 * DIFFUSION)
N_MODES, N_QUAD = 64, 128


def smooth_intervals(case):
    if case == "compact":
        return (0.5 - COMPACT_RADIUS, 0.5 + COMPACT_RADIUS)
    return (0.0, 1.0)


def sine_solution(case, x, times, n_modes=N_MODES, n_quad=N_QUAD):
    """Expand exp(beta*x)*f in the Dirichlet sine basis, then propagate each mode."""
    low, high = smooth_intervals(case)
    nodes, weights = roots_legendre(n_quad)
    y = low + (high - low) * (nodes + 1) / 2
    weights = weights * (high - low) / 2
    freq = np.pi * np.arange(1, n_modes + 1)
    projected = np.exp(BETA * y) * initial_numpy(case, y)
    coefs = 2 * np.sin(freq[:, None] * y) @ (weights * projected)
    x = np.asarray(x)
    sine = np.sin(freq[:, None] * x) * np.exp(-BETA * x)
    values = []
    for t in times:
        if t == 0:
            values.append(initial_numpy(case, x))
        else:
            decay = np.exp(-(DIFFUSION * freq**2 + KAPPA) * t)
            u = (coefs * decay) @ sine
            values.append(np.where((x == 0) | (x == 1), 0.0, u))
    return np.asarray(values)


def image_solution(case, x, t):
    """Independent check using the reflected heat kernel and adaptive quadrature."""
    if t <= 0:
        raise ValueError("镜像核只在正时间使用")
    # For D=0.05 and t<=0.5, omitted images start at distances >= 8.
    shifts = 2 * np.arange(-4, 5)[:, None]
    inv_variance = 1 / (4 * DIFFUSION * t)
    normalizer = 1 / np.sqrt(4 * np.pi * DIFFUSION * t)

    def integrand(y):
        direct = np.exp(-(x - y + shifts)**2 * inv_variance)
        mirrored = np.exp(-(x + y + shifts)**2 * inv_variance)
        return ((direct - mirrored).sum(axis=0) * normalizer
                * np.exp(BETA * (y - x) - KAPPA * t) * initial_numpy(case, y))

    low, high = smooth_intervals(case)
    result, _ = quad_vec(integrand, low, high, epsabs=1e-13, epsrel=1e-13)
    return result


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def generate(output):
    output.mkdir(parents=True, exist_ok=True)
    x = np.arange(65) / 64
    fine_x = np.arange(129) / 128
    grid_rows, solution_rows, check_rows = [], [], []
    for case in CASES:
        initial = initial_numpy(case, x)
        fine_initial = initial_numpy(case, fine_x)
        if not np.array_equal(initial, fine_initial[::2]):
            raise ArithmeticError(f"{case} 的两套初值网格不一致")
        for n, xx, ff in ((64, x, initial), (128, fine_x, fine_initial)):
            grid_rows.extend((case, n, xj, fj) for xj, fj in zip(xx, ff))
        values = sine_solution(case, x, EVAL_TIMES)
        fine = sine_solution(case, x, EVAL_TIMES, 2 * N_MODES, 2 * N_QUAD)
        for i, t in enumerate(EVAL_TIMES):
            u = values[i]
            solution_rows.extend((case, xj, t, uj) for xj, uj in zip(x, u))
            if t == 0:
                continue
            image_error = np.max(np.abs(u - image_solution(case, x, t)))
            refine_error = np.max(np.abs(u - fine[i]))
            closed_error = ""
            if case == "baseline":
                closed = initial * np.exp(-(DIFFUSION * np.pi**2 + KAPPA) * t)
                closed_error = np.max(np.abs(u - closed))
            if max(refine_error, image_error, closed_error or 0) > 1e-12:
                raise ArithmeticError(f"{case}, t={t}：参照的独立核验误差超过 1e-12")
            check_rows.append((case, t, refine_error, image_error, closed_error))
    write_csv(output / "initial_grids.csv", ["case", "N", "x", "initial"], grid_rows)
    write_csv(output / "profiles.csv", ["case", "x", "t", "reference"], solution_rows)
    write_csv(output / "checks.csv",
              ["case", "t", "refinement_max_abs", "image_max_abs", "closed_form_max_abs"],
              check_rows)
    metadata = {
        "drift": DRIFT, "diffusion": DIFFUSION, "eval_times": EVAL_TIMES,
        "peak_height": AMPLITUDE, "peak_x": {"baseline": BASELINE_PEAK_X,
            **PEAK_X, "compact": 0.5},
        "gaussian_width": WIDTH, "peak_scale_multipliers": PEAK_SCALES,
        "center_and_compact_fwhm": 2 * CENTER_HALF_WIDTH, "compact_radius": COMPACT_RADIUS,
        "modes": N_MODES, "quadrature_nodes": N_QUAD,
        "refinement": "double both mode count and Gauss-Legendre nodes",
        "kernel_check": "adaptive Gauss-Kronrod quadrature; Dirichlet images -4..4",
        "max_refinement_difference": max(row[2] for row in check_rows),
        "max_image_difference": max(row[3] for row in check_rows),
        "max_baseline_closed_form_difference": max(row[4] for row in check_rows if row[0] == "baseline"),
    }
    with (output / "settings.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Four references verified and written to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "reference")
    generate(parser.parse_args().output)
