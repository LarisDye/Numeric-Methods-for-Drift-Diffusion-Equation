"""核对初值归一化、PyTorch 二阶导数和零边界谱参照。"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_reference import KAPPA, image_solution, sine_solution
from profiles import (
    AMPLITUDE, CASES, CENTER_HALF_WIDTH, COMPACT_RADIUS, DIFFUSION,
    initial_numpy, initial_torch,
)


@pytest.mark.parametrize("case", CASES)
def test_continuous_profiles(case):
    peak = minimize_scalar(lambda x: -initial_numpy(case, x), bounds=(0, 1),
                           method="bounded", options={"xatol": 1e-14})
    assert peak.success
    assert -peak.fun == pytest.approx(AMPLITUDE, abs=1e-14)
    np.testing.assert_equal(initial_numpy(case, [0, 1]), [0.0, 0.0])
    xx = np.linspace(0.001, 0.999, 57)
    x = torch.tensor(xx, dtype=torch.float64, requires_grad=True)
    y = initial_torch(case, x)
    np.testing.assert_allclose(y.detach(), initial_numpy(case, xx), atol=2e-15)
    slope = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    curv = torch.autograd.grad(slope.sum(), x)[0].detach().numpy()
    h = 1e-5
    finite_difference = (initial_numpy(case, xx + h) - 2 * initial_numpy(case, xx)
                         + initial_numpy(case, xx - h)) / h**2
    np.testing.assert_allclose(curv, finite_difference, atol=1e-5, rtol=1e-5)


def test_compact_has_matched_width_and_vanishing_curvature_at_joins():
    for case in ("center", "compact"):
        half_x = np.array([0.5 - CENTER_HALF_WIDTH, 0.5 + CENTER_HALF_WIDTH])
        np.testing.assert_allclose(initial_numpy(case, half_x), AMPLITUDE / 2, atol=1e-14)
    x = torch.tensor([0.5 - COMPACT_RADIUS, 0.5 + COMPACT_RADIUS], dtype=torch.float64,
                     requires_grad=True)
    value = initial_torch("compact", x)
    slope = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
    curv = torch.autograd.grad(slope.sum(), x)[0]
    for tensor in (value, slope, curv):
        assert torch.max(torch.abs(tensor)).item() < 1e-11


@pytest.mark.parametrize("case", CASES)
def test_reference_at_independent_points(case):
    x = np.array([0, 0.029, 0.3421, 0.573, 0.76123, 1])
    t = 0.024
    u = sine_solution(case, x, [t])[0]
    np.testing.assert_allclose(u, image_solution(case, x, t), atol=1e-12, rtol=0)
    if case == "baseline":
        closed = np.exp(-(DIFFUSION * np.pi**2 + KAPPA) * t) * initial_numpy(case, x)
        np.testing.assert_allclose(u, closed, atol=1e-12, rtol=0)
