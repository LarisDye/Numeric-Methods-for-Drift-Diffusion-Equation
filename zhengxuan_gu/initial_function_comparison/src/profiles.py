"""一维漂移扩散实验使用的连续初值。NumPy 和 PyTorch 调用同一套表达式。"""

from math import atan, exp, pi, sin, sqrt

import numpy as np
from scipy.optimize import brentq


CASES = ("baseline", "center", "near", "compact")
DRIFT, DIFFUSION, END_TIME = 0.2, 0.05, 0.5
EVAL_TIMES = (0.0, 0.03125, 0.125, END_TIME)
BETA = DRIFT / (2 * DIFFUSION)
WIDTH = 0.08
PEAK_CENTERS = {"center": 0.5, "near": 0.2}
BASELINE_PEAK_X = atan(pi / BETA) / pi
AMPLITUDE = exp(-BETA * BASELINE_PEAK_X) * sin(pi * BASELINE_PEAK_X)


def _peak_x(center):
    # The derivative of log f decreases strictly from +infinity to -infinity.
    def slope(x):
        return 3 / x - 3 / (1 - x) - 2 * (x - center) / WIDTH**2

    return brentq(slope, 1e-12, 1 - 1e-12, xtol=1e-15)


def _unscaled_peak(x, center, exponential):
    return x**3 * (1 - x)**3 * exponential(-((x - center) / WIDTH)**2)


PEAK_X = {case: _peak_x(center) for case, center in PEAK_CENTERS.items()}
PEAK_SCALES = {
    case: AMPLITUDE / _unscaled_peak(PEAK_X[case], center, exp)
    for case, center in PEAK_CENTERS.items()
}


def _center_half_height(x):
    return PEAK_SCALES["center"] * _unscaled_peak(x, 0.5, exp) - AMPLITUDE / 2


CENTER_HALF_WIDTH = brentq(_center_half_height, 0.5, 1.0, xtol=1e-15) - 0.5
COMPACT_RADIUS = CENTER_HALF_WIDTH / sqrt(1 - 2**(-1 / 3))


def _initial(case, x, lib):
    if case == "baseline":
        # Same operation order as the method-four benchmark.
        return lib.exp(-DRIFT * x / (2 * DIFFUSION)) * lib.sin(pi * x)
    if case in PEAK_CENTERS:
        return PEAK_SCALES[case] * _unscaled_peak(x, PEAK_CENTERS[case], lib.exp)
    if case == "compact":
        core = 1 - ((x - 0.5) / COMPACT_RADIUS)**2
        positive = np.maximum(core, 0) if lib is np else core.clamp_min(0)
        return AMPLITUDE * positive**3
    raise ValueError(f"未知初值 {case!r}，可选 {CASES}")


def initial_numpy(case, x):
    x = np.asarray(x, dtype=np.float64)
    value = _initial(case, x, np)
    # Enforce the analytic zero at the two endpoints of each exported grid.
    return np.where((x == 0) | (x == 1), 0.0, value)


def initial_torch(case, x):
    """保留输入张量的梯度图、设备和精度。"""
    import torch

    return _initial(case, x, torch)
