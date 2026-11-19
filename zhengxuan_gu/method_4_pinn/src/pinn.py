"""一维漂移扩散 PINN：模型、残差、训练与独立评价。

求解 u_t = a*u_x + D*u_xx，0<x<1，两个端点为零。
训练只读取初值；exact() 仅用于独立验证，不进入损失函数。
"""

from __future__ import annotations

import math
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

DTYPE = torch.float64


@dataclass(frozen=True)
class Problem:
    a: float = 0.2
    D: float = 0.05
    T: float = 0.5
    x_min: float = 0.0
    x_max: float = 1.0
    boundary: str = "homogeneous_dirichlet"
    initial_condition: str = "eigenmode"
    name: str = "dirichlet_eigenmode"

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.a, self.D, self.T)):
            raise ValueError("a, D, T 必须为有限数值")
        if self.D <= 0 or self.T <= 0:
            raise ValueError("D 和 T 必须大于零")
        if (self.x_min, self.x_max) != (0.0, 1.0):
            raise ValueError("当前实现只支持空间区间 [0, 1]")
        if self.boundary != "homogeneous_dirichlet":
            raise ValueError("当前实现只支持两端固定为零")
        if self.initial_condition != "eigenmode":
            raise ValueError("当前初值接口只实现 eigenmode；更换初值须同时更新验证基准")

    def initial(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.a * x / (2 * self.D)) * torch.sin(torch.pi * x)

    def exact(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        decay = self.D * math.pi**2 + self.a**2 / (4 * self.D)
        return self.initial(x) * torch.exp(-decay * t)


@dataclass(frozen=True)
class TrainConfig:
    constraint: str = "hard"
    steps: int = 2000
    width: int = 32
    layers: int = 2
    interior_points: int = 1024
    initial_points: int = 128
    boundary_points: int = 64
    learning_rate: float = 1e-3
    lambda_initial: float = 1.0
    lambda_boundary: float = 1.0
    seed: int = 7
    points_seed: int = 2026
    threads: int = 1
    lbfgs_iterations: int = 0

    def __post_init__(self):
        if self.constraint not in ("soft", "hard"):
            raise ValueError("constraint 必须是 soft 或 hard")
        if type(self.lbfgs_iterations) is not int or self.lbfgs_iterations < 0:
            raise ValueError("lbfgs_iterations 必须是非负整数")
        for name in ("steps", "width", "layers", "interior_points", "initial_points", "boundary_points", "threads"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        for name in ("learning_rate", "lambda_initial", "lambda_boundary"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        if (type(self.seed) is not int or type(self.points_seed) is not int
                or not (0 <= self.seed < 2**32 and 0 <= self.points_seed < 2**32 - 100001)):
            raise ValueError("随机种子必须为非负小整数")


class PINN(nn.Module):
    """网络只接收物理坐标 (x,t)，时间归一化发生在可求导的前向中。"""

    def __init__(self, problem: Problem, config: TrainConfig):
        super().__init__()
        self.problem = problem
        self.config = config
        modules = []
        input_width = 2
        for _ in range(config.layers):
            modules.extend([nn.Linear(input_width, config.width), nn.Tanh()])
            input_width = config.width
        modules.append(nn.Linear(input_width, 1))
        self.net = nn.Sequential(*modules).to(dtype=DTYPE, device="cpu")
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        x, t = coordinates[:, :1], coordinates[:, 1:2]
        s = t / self.problem.T
        raw = self.net(torch.cat((x, s), dim=1))
        if self.config.constraint == "soft":
            return raw
        # 初值与零边界由结构保证；不能只对 raw 求 PDE 导数。
        return self.problem.initial(x) + s * 4 * x * (1 - x) * raw


def pde_residual(
    model: Callable[[torch.Tensor], torch.Tensor],
    coordinates: torch.Tensor,
    problem: Problem,
    *,
    create_graph: bool = True,
) -> torch.Tensor:
    """按完整输出计算 U_t-a*U_x-D*U_xx，保留训练所需的高阶计算图。"""
    z = coordinates.detach().clone().requires_grad_(True)
    u = model(z)
    derivatives = torch.autograd.grad(u, z, torch.ones_like(u), create_graph=True)[0]
    ux, ut = derivatives[:, :1], derivatives[:, 1:2]
    second = torch.autograd.grad(
        ux, z, torch.ones_like(ux), create_graph=create_graph, allow_unused=True
    )[0] if ux.requires_grad else None
    uxx = torch.zeros_like(ux) if second is None else second[:, :1]
    return ut - problem.a * ux - problem.D * uxx


def make_training_points(problem: Problem, config: TrainConfig):
    """配点种子独立于网络初始化种子，软/硬版本可使用相同坐标。"""
    generator = torch.Generator(device="cpu").manual_seed(config.points_seed)

    def uniform(count, columns=1):
        return 1e-12 + (1 - 2e-12) * torch.rand(count, columns, generator=generator, dtype=DTYPE)

    interior = uniform(config.interior_points, 2)
    interior[:, 1] *= problem.T
    initial_x = uniform(config.initial_points)
    initial = torch.cat((initial_x, torch.zeros_like(initial_x)), dim=1)
    times = problem.T * uniform(config.boundary_points)
    left = torch.cat((torch.zeros_like(times), times), dim=1)
    right = torch.cat((torch.ones_like(times), times), dim=1)
    return {"interior": interior, "initial": initial, "boundary": torch.cat((left, right), dim=0)}


def loss_terms(model: PINN, points: dict):
    residual_loss = pde_residual(model, points["interior"], model.problem).square().mean()
    initial_prediction = model(points["initial"])
    initial_loss = (initial_prediction - model.problem.initial(points["initial"][:, :1])).square().mean()
    boundary_loss = model(points["boundary"]).square().mean()
    total = residual_loss
    if model.config.constraint == "soft":
        total = total + model.config.lambda_initial * initial_loss + model.config.lambda_boundary * boundary_loss
    return {"total": total, "pde": residual_loss, "initial": initial_loss, "boundary": boundary_loss}


def lbfgs_options(iterations):
    return {"lr": 1, "max_iter": iterations, "max_eval": 2 * iterations,
            "history_size": 50, "line_search_fn": "strong_wolfe",
            "tolerance_grad": 1e-12, "tolerance_change": 1e-15}


def refine_lbfgs(model, points, iterations, *, history=None):
    """在给定 Adam 权重和固定配点上精修，单独统计线搜索的梯度求值次数。"""
    if type(iterations) is not int or iterations < 1:
        raise ValueError("L-BFGS 迭代数必须为正整数")
    setup_start = time.perf_counter()
    optimizer = torch.optim.LBFGS(model.parameters(), **lbfgs_options(iterations))
    setup_seconds = time.perf_counter() - setup_start
    evaluations = 0

    def closure():
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        terms = loss_terms(model, points)
        if not torch.isfinite(terms["total"]):
            raise RuntimeError("L-BFGS 遇到非有限训练损失")
        terms["total"].backward()
        evaluations += 1
        if history is not None:
            history.append({"stage": "lbfgs_trial", "step": None,
                            "gradient_evaluations": model.config.steps + evaluations,
                            **{key: value.detach().item() for key, value in terms.items()}})
        return terms["total"]

    start = time.perf_counter()
    optimizer.step(closure)
    seconds = time.perf_counter() - start
    completed = optimizer.state[next(model.parameters())]["n_iter"]
    model.config = replace(model.config, lbfgs_iterations=iterations)
    if history is not None:
        final = loss_terms(model, points)
        history.append({"stage": "lbfgs_final", "step": None,
                        "gradient_evaluations": model.config.steps + evaluations,
                        **{key: value.detach().item() for key, value in final.items()}})
    if completed >= iterations:
        reason = "lbfgs_iteration_limit"
    elif evaluations >= 2 * iterations:
        reason = "lbfgs_evaluation_limit"
    else:
        reason = "lbfgs_tolerance_or_no_progress"
    return {"lbfgs_iterations_completed": completed, "lbfgs_gradient_evaluations": evaluations,
            "lbfgs_setup_seconds": setup_seconds, "lbfgs_train_seconds": seconds,
            "stop_reason": reason}


def initialize_training(problem: Problem, config: TrainConfig):
    """为相同问题初始化网络、固定配点和 Adam；预热不改变权重。"""
    setup_start = time.perf_counter()
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    model = PINN(problem, replace(config, lbfgs_iterations=0))
    points = make_training_points(problem, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # 只预热导数与反向传播，不更新参数，不偷跑优化步数。
    warm = loss_terms(model, points)
    warm["total"].backward()
    optimizer.zero_grad(set_to_none=True)
    del warm
    setup_seconds = time.perf_counter() - setup_start
    return model, points, optimizer, setup_seconds


def train_adam(model, points, optimizer, steps, *, start_step=0, verbose=False):
    """直接更新传入的模型与 Adam 动量，返回训练损失记录与本段耗时。"""
    if type(steps) is not int or steps < 1 or type(start_step) is not int or start_step < 0:
        raise ValueError("steps 必须为正整数，start_step 必须为非负整数")
    stop_step = start_step + steps
    model.train()
    history = []
    train_start = time.perf_counter()
    for step in range(start_step, stop_step):
        optimizer.zero_grad(set_to_none=True)
        terms = loss_terms(model, points)
        if not torch.isfinite(terms["total"]):
            raise RuntimeError(f"第 {step} 步损失非有限值，请检查输入或学习率")
        # step 表示已完成的参数更新数；每一行记录此时网络的损失。
        history.append({"step": step, **{key: value.detach().item() for key, value in terms.items()}})
        terms["total"].backward()
        optimizer.step()
        if verbose and (step == 0 or (step + 1) % 200 == 0):
            print(f"step {step + 1:5d}/{stop_step}: loss(before update)={history[-1]['total']:.6g}", flush=True)
    train_seconds = time.perf_counter() - train_start
    final_terms = loss_terms(model, points)
    history.append({"step": stop_step, **{key: value.detach().item() for key, value in final_terms.items()}})
    model.config = replace(model.config, steps=stop_step, lbfgs_iterations=0)
    model.eval()
    return history, train_seconds


def train_model(problem: Problem, config: TrainConfig, *, verbose=False):
    """训练单个 PINN。比较两个后续优化器时可复用上面的分段接口。"""
    model, points, optimizer, setup_seconds = initialize_training(problem, config)
    history, train_seconds = train_adam(model, points, optimizer, config.steps, verbose=verbose)
    timing = {"setup_seconds": setup_seconds, "adam_train_seconds": train_seconds,
              "adam_steps_completed": config.steps, "train_seconds": train_seconds,
              "stop_reason": "requested_steps_completed"}
    if config.lbfgs_iterations:
        for row in history:
            row.update(stage="adam", gradient_evaluations=row["step"])
        refinement = refine_lbfgs(model, points, config.lbfgs_iterations, history=history)
        timing.update(refinement)
        timing["setup_seconds"] += refinement["lbfgs_setup_seconds"]
        timing["train_seconds"] += refinement["lbfgs_train_seconds"]
        if verbose:
            print(f"L-BFGS: {refinement['lbfgs_iterations_completed']} iterations, "
                  f"{refinement['lbfgs_gradient_evaluations']} gradient evaluations", flush=True)
    model.eval()
    return model, history, timing


def save_checkpoint(path: Path | str, model: PINN):
    torch.save({
        "format_version": 1,
        "problem": asdict(model.problem),
        "train_config": asdict(model.config),
        "state_dict": model.state_dict(),
        "dtype": "float64",
    }, path)


def load_checkpoint(path: Path | str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("dtype") != "float64":
        raise ValueError("不支持此模型文件格式")
    problem = Problem(**checkpoint["problem"])
    config = TrainConfig(**checkpoint["train_config"])
    model = PINN(problem, config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def environment_info():
    return {
        "python": sys.version.split()[0], "torch": str(torch.__version__),
        "numpy": np.__version__, "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "device": "cpu", "dtype": "float64", "torch_threads": torch.get_num_threads(),
    }


def evaluate_model(model: PINN, space_intervals=64, time_intervals=40, *, residual_points=512):
    if space_intervals < 2 or time_intervals < 1 or residual_points < 1:
        raise ValueError("评价网格至少包含两个空间小区间、一个时间小区间")
    evaluation_start = time.perf_counter()
    p = model.problem
    x = torch.linspace(0, 1, space_intervals + 1, dtype=DTYPE)
    t = torch.linspace(0, p.T, time_intervals + 1, dtype=DTYPE)
    tt, xx = torch.meshgrid(t, x, indexing="ij")
    coordinates = torch.stack((xx.flatten(), tt.flatten()), dim=1)
    terminal = torch.stack((x, torch.full_like(x, p.T)), dim=1)
    with torch.no_grad():
        for _ in range(3):
            model(terminal)
        query_times = []
        for _ in range(7):
            start = time.perf_counter()
            terminal_prediction = model(terminal).flatten()
            query_times.append(time.perf_counter() - start)
        prediction = model(coordinates).reshape(time_intervals + 1, space_intervals + 1)
        truth = p.exact(xx, tt)
        terminal_truth = p.exact(x, torch.full_like(x, p.T))

    generator = torch.Generator(device="cpu").manual_seed(model.config.points_seed + 100000)
    test_initial_x = torch.rand(257, 1, generator=generator, dtype=DTYPE)
    initial_coordinates = torch.cat((test_initial_x, torch.zeros_like(test_initial_x)), dim=1)
    boundary_times = torch.cat((torch.linspace(0, p.T, 201, dtype=DTYPE).reshape(-1, 1),
                                p.T * torch.rand(257, 1, generator=generator, dtype=DTYPE)))
    boundary_coordinates = torch.cat((torch.cat((torch.zeros_like(boundary_times), boundary_times), dim=1),
                                      torch.cat((torch.ones_like(boundary_times), boundary_times), dim=1)))
    test_interior = torch.rand(residual_points, 2, generator=generator, dtype=DTYPE)
    test_interior[:, 1] *= p.T
    with torch.no_grad():
        initial_error = model(initial_coordinates) - p.initial(test_initial_x)
        boundary_values = model(boundary_coordinates)
    residual = pde_residual(model, test_interior, p, create_graph=False).detach()
    terminal_error = terminal_prediction[1:-1] - terminal_truth[1:-1]
    # 排除已知的空间边界和初始时间，避免硬约束让整体误差显得过小。
    space_time_error = prediction[1:, 1:-1] - truth[1:, 1:-1]
    metrics = {
        "method": f"pinn_{model.config.constraint}" + ("_lbfgs" if model.config.lbfgs_iterations else ""),
        "terminal_rmse": terminal_error.square().mean().sqrt().item(),
        "terminal_max_error": terminal_error.abs().max().item(),
        "spacetime_rmse": space_time_error.square().mean().sqrt().item(),
        "initial_rmse": initial_error.square().mean().sqrt().item(),
        "boundary_max_error": boundary_values.abs().max().item(),
        "residual_rms": residual.square().mean().sqrt().item(),
        "query_seconds": float(np.median(query_times)),
        "query_iqr_seconds": float(np.percentile(query_times, 75) - np.percentile(query_times, 25)),
        "query_repeats_seconds": query_times,
        "query_point_count": len(x),
        "space_intervals": space_intervals, "time_intervals": time_intervals,
        "test_seed": model.config.points_seed + 100000,
        "boundary_test_points": len(boundary_coordinates), "residual_test_points": residual_points,
        "evaluation_seconds": time.perf_counter() - evaluation_start,
    }
    return metrics, {
        "x": x.numpy(), "t": t.numpy(), "prediction": prediction.numpy(), "exact": truth.numpy(),
        "terminal_prediction": terminal_prediction.numpy(), "terminal_exact": terminal_truth.numpy(),
    }
