"""验证数学与训练机制；不依赖完整 2000 步实验。"""

import sys
from pathlib import Path
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pinn import (DTYPE, PINN, Problem, TrainConfig, evaluate_model, load_checkpoint,
                  initialize_training, train_adam, refine_lbfgs, loss_terms, make_training_points,
                  pde_residual, save_checkpoint, train_model)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(123)


def test_analytic_solution_has_zero_pde_residual():
    problem = Problem()
    z = torch.rand(31, 2, dtype=DTYPE)
    z[:, 1] *= problem.T

    # 独立写下本算例的解析函数，检查符号与一、二阶自动微分。
    def exact(z):
        x, t = z[:, :1], z[:, 1:2]
        return torch.exp(-2 * x) * torch.sin(torch.pi * x) * torch.exp(-(0.05 * torch.pi**2 + 0.2) * t)

    residual = pde_residual(exact, z, problem, create_graph=False)
    assert residual.abs().max().item() < 1e-14


def test_physical_time_chain_rule():
    problem = Problem(T=0.5)
    z = torch.tensor([[0.2, 0.1], [0.7, 0.4]], dtype=DTYPE)

    def normalized_time_function(z):
        return z[:, :1].square() + z[:, 1:2] / problem.T

    actual = pde_residual(normalized_time_function, z, problem, create_graph=False)
    expected = 1 / problem.T - 2 * problem.a * z[:, :1] - 2 * problem.D
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-14)


def test_hard_constraints_hold_before_training():
    problem = Problem()
    model = PINN(problem, TrainConfig(constraint="hard", width=8))
    x = torch.linspace(0, 1, 25, dtype=DTYPE).reshape(-1, 1)
    initial = torch.cat((x, torch.zeros_like(x)), dim=1)
    t = problem.T * torch.rand(25, 1, dtype=DTYPE)
    left = torch.cat((torch.zeros_like(t), t), dim=1)
    right = torch.cat((torch.ones_like(t), t), dim=1)
    torch.testing.assert_close(model(initial), problem.initial(x), rtol=0, atol=1e-15)
    assert model(left).abs().max().item() < 1e-15
    assert model(right).abs().max().item() < 1e-15


@pytest.mark.parametrize("constraint", ["soft", "hard"])
def test_residual_parameter_gradient_matches_finite_difference(constraint):
    config = TrainConfig(constraint=constraint, width=8, interior_points=11, initial_points=7, boundary_points=5)
    model = PINN(Problem(), config)
    points = make_training_points(model.problem, config)
    loss = loss_terms(model, points)["total"]
    loss.backward()
    parameter = model.net[0].weight
    analytic = parameter.grad[0, 0].item()
    assert torch.isfinite(parameter.grad).all()
    epsilon = 1e-6
    original = parameter[0, 0].item()
    with torch.no_grad():
        parameter[0, 0] = original + epsilon
    plus = loss_terms(model, points)["total"].item()
    with torch.no_grad():
        parameter[0, 0] = original - epsilon
    minus = loss_terms(model, points)["total"].item()
    with torch.no_grad():
        parameter[0, 0] = original
    numeric = (plus - minus) / (2 * epsilon)
    assert analytic == pytest.approx(numeric, rel=2e-5, abs=1e-7)


def test_checkpoint_restores_prediction_and_problem(tmp_path):
    model = PINN(Problem(a=-0.1, D=0.1), TrainConfig(constraint="hard", width=8))
    z = torch.tensor([[0.2, 0.1], [0.5, 0.4]], dtype=DTYPE)
    prediction = model(z).detach()
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, model)
    restored = load_checkpoint(checkpoint)
    assert restored.problem == model.problem
    torch.testing.assert_close(restored(z), prediction, rtol=0, atol=0)


def test_lbfgs_uses_adam_endpoint_and_round_trips(tmp_path):
    config = TrainConfig(steps=30, lbfgs_iterations=12, width=8, interior_points=32,
                         initial_points=12, boundary_points=8)
    model, history, timing = train_model(Problem(), config)
    baseline = history[config.steps]
    trials = [row for row in history if row["stage"] == "lbfgs_trial"]
    assert baseline["stage"] == "adam" and trials[0]["total"] == baseline["total"]
    assert len(trials) == timing["lbfgs_gradient_evaluations"]
    assert 0 < timing["lbfgs_iterations_completed"] <= config.lbfgs_iterations
    assert history[-1]["stage"] == "lbfgs_final"
    assert history[-1]["total"] < baseline["total"]
    assert timing["train_seconds"] == timing["adam_train_seconds"] + timing["lbfgs_train_seconds"]

    checkpoint = tmp_path / "refined.pt"
    save_checkpoint(checkpoint, model)
    restored = load_checkpoint(checkpoint)
    assert restored.config == config
    metrics, arrays = evaluate_model(restored, 8, 2, residual_points=8)
    assert metrics["method"] == "pinn_hard_lbfgs"
    assert (arrays["prediction"] == evaluate_model(model, 8, 2, residual_points=8)[1]["prediction"]).all()


def test_old_checkpoint_without_lbfgs_option(tmp_path):
    model = PINN(Problem(), TrainConfig(width=8))
    path = tmp_path / "old.pt"
    save_checkpoint(path, model)
    old = torch.load(path, weights_only=True)
    del old["train_config"]["lbfgs_iterations"]
    torch.save(old, path)
    restored = load_checkpoint(path)
    assert restored.config.lbfgs_iterations == 0
    assert evaluate_model(restored, 8, 2, residual_points=8)[0]["method"] == "pinn_hard"


def test_training_points_are_fixed_across_initialization_seeds():
    a = make_training_points(Problem(), TrainConfig(seed=1, points_seed=42))
    b = make_training_points(Problem(), TrainConfig(seed=99, points_seed=42))
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_short_training_improves_independent_error():
    problem = Problem()
    config = TrainConfig(constraint="hard", steps=100, width=16, interior_points=64,
                         initial_points=24, boundary_points=16, seed=7)
    torch.manual_seed(config.seed)
    untrained = PINN(problem, config)
    before, _ = evaluate_model(untrained, 32, 8, residual_points=32)
    model, history, timing = train_model(problem, config)
    after, _ = evaluate_model(model, 32, 8, residual_points=32)
    assert after["terminal_rmse"] < before["terminal_rmse"] * 0.3
    assert history[-1]["pde"] < history[0]["pde"] * 0.1
    assert after["boundary_max_error"] < 1e-14
    assert timing["train_seconds"] > 0


@pytest.mark.parametrize("arguments", [{"D": 0}, {"T": -1}, {"a": float("nan")}, {"x_max": 2}])
def test_invalid_problem_is_rejected(arguments):
    with pytest.raises(ValueError):
        Problem(**arguments)

def test_optimizer_branches_share_weights_and_adam_keeps_momentum():
    config = TrainConfig(steps=12, width=8, interior_points=20, initial_points=8, boundary_points=6)
    model, points, optimizer, _ = initialize_training(Problem(), config)
    adam_history, _ = train_adam(model, points, optimizer, config.steps)
    refined = deepcopy(model)
    trial_history = []
    result = refine_lbfgs(refined, points, iterations=4, history=trial_history)
    assert adam_history[-1]["total"] == trial_history[0]["total"]
    extra = result["lbfgs_gradient_evaluations"]
    extra_history, _ = train_adam(model, points, optimizer, extra, start_step=config.steps)
    assert adam_history[-1]["total"] == extra_history[0]["total"]
    total = config.steps + extra
    reference, _, _ = train_model(Problem(), replace(config, steps=total))
    assert model.config.steps == total
    for parameter, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter, expected, atol=0, rtol=0)


def test_soft_loss_uses_initial_and_boundary_weights_and_hard_uses_pde():
    problem = Problem()
    for constraint in ("soft", "hard"):
        config = TrainConfig(constraint=constraint, lambda_initial=2.0, lambda_boundary=3.0,
                             width=8, interior_points=20, initial_points=8, boundary_points=6)
        model = PINN(problem, config)
        terms = loss_terms(model, make_training_points(problem, config))
        expected = terms["pde"]
        if constraint == "soft":
            expected = expected + 2.0*terms["initial"] + 3.0*terms["boundary"]
        torch.testing.assert_close(terms["total"], expected, rtol=0, atol=0)


def test_training_does_not_read_exact_solution(monkeypatch):
    def forbidden(*args):
        raise AssertionError("训练使用了解析解")
    monkeypatch.setattr(Problem, "exact", forbidden)
    config = TrainConfig(constraint="soft", steps=2, lbfgs_iterations=2, width=8,
                         interior_points=20, initial_points=8, boundary_points=6)
    train_model(Problem(), config)
