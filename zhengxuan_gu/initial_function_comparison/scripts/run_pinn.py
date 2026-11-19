"""更换初值后重训方法四的硬约束 PINN，输出公共评价点上的预测与计时。"""

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "method_4_pinn" / "src"))
sys.path.insert(0, str(ROOT / "src"))
from pinn import DTYPE, PINN, Problem, TrainConfig, environment_info, pde_residual, train_model
from profiles import CASES, DIFFUSION, DRIFT, END_TIME, EVAL_TIMES, initial_torch

@dataclass(frozen=True)
class ProfileProblem(Problem):
    """复用同一方程和边界的硬约束网络，只替换它读取的初值。"""

    initial_condition: str = "baseline"
    name: str = "dirichlet_profile"

    def __post_init__(self):
        if self.initial_condition not in CASES:
            raise ValueError(f"未知初值 {self.initial_condition}")
        if (self.a, self.D, self.T) != (DRIFT, DIFFUSION, END_TIME):
            raise ValueError("本实验只改变初值，三个物理参数固定")
        if (self.x_min, self.x_max, self.boundary) != (0., 1., "homogeneous_dirichlet"):
            raise ValueError("本实验固定区间 [0,1] 和零边界")

    def initial(self, x):
        return initial_torch(self.initial_condition, x)

    def exact(self, x, t):
        raise NotImplementedError("请用本实验的独立参考解核验预测")


def load_model(path):
    """读取包含初值标识的模型，可直接送入双精度的 (x, t) 坐标。"""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    equation = saved["equation"]
    problem = ProfileProblem(initial_condition=saved["case"], **equation)
    model = PINN(problem, TrainConfig(**saved["train_config"]))
    model.load_state_dict(saved["weights"])
    return model.eval()


def evaluate(model):
    """记录三个物理时刻和初态，在训练点之外检查残差与边界。"""
    problem, seed = model.problem, model.config.seed
    x = torch.linspace(0., 1., 65, dtype=DTYPE)
    rows, query_times = [], []
    with torch.no_grad():
        terminal = torch.stack((x, torch.full_like(x, problem.T)), dim=1)
        for _ in range(3):
            model(terminal)
        for _ in range(7):
            begin = time.perf_counter()
            model(terminal)
            query_times.append(time.perf_counter() - begin)
        for t in EVAL_TIMES:
            coordinates = torch.stack((x, torch.full_like(x, t)), dim=1)
            values = model(coordinates).flatten()
            if not torch.isfinite(values).all():
                raise RuntimeError("预测结果中存在非有限数值")
            rows.extend(dict(case=problem.initial_condition, method="pinn", seed=seed, x=xi,
                             t=t, prediction=value) for xi, value in zip(x.tolist(), values.tolist()))

        generator = torch.Generator().manual_seed(model.config.points_seed + 100000)
        initial_x = torch.rand(257, 1, generator=generator, dtype=DTYPE)
        initial_coords = torch.cat((initial_x, torch.zeros_like(initial_x)), dim=1)
        initial_error = (model(initial_coords) - problem.initial(initial_x)).square().mean().sqrt().item()
        boundary_t = torch.cat((torch.linspace(0, problem.T, 201, dtype=DTYPE).view(-1, 1),
                                problem.T * torch.rand(257, 1, generator=generator, dtype=DTYPE)))
        boundary_coords = torch.cat((torch.cat((torch.zeros_like(boundary_t), boundary_t), dim=1),
                                     torch.cat((torch.ones_like(boundary_t), boundary_t), dim=1)))
        boundary_error = model(boundary_coords).abs().max().item()
    interior = torch.rand(512, 2, generator=generator, dtype=DTYPE)
    interior[:, 1] *= problem.T
    residual = pde_residual(model, interior, problem, create_graph=False)
    diagnostics = dict(initial_rmse=initial_error, boundary_max_error=boundary_error,
                       residual_rms=residual.detach().square().mean().sqrt().item())
    return rows, query_times, diagnostics


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "pinn")
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--adam-steps", type=int, default=2000)
    parser.add_argument("--lbfgs-iterations", type=int, default=400)
    args = parser.parse_args()
    if len(set(args.cases)) != len(args.cases) or len(set(args.seeds)) != len(args.seeds):
        parser.error("案例及种子不可重复")
    if min(args.adam_steps, args.lbfgs_iterations) < 1:
        parser.error("训练步数与迭代数须为正")
    args.output.mkdir(exist_ok=True, parents=True)
    profiles, timings, queries = [], [], []
    start = time.perf_counter()
    with TemporaryDirectory(dir=args.output) as directory:
        temp = Path(directory)
        for case in args.cases:
            for seed in args.seeds:
                config = TrainConfig(seed=seed, steps=args.adam_steps, lbfgs_iterations=args.lbfgs_iterations)
                model, _, timing = train_model(ProfileProblem(initial_condition=case), config)
                rows, repeats, diagnostics = evaluate(model)
                profiles.extend(rows)
                timings.append(dict(
                    case=case, method="pinn", seed=seed,
                    setup_seconds=timing["setup_seconds"], train_seconds=timing["train_seconds"],
                    query_seconds=median(repeats),
                    total_seconds=timing["setup_seconds"] + timing["train_seconds"] + median(repeats),
                    adam_steps=timing["adam_steps_completed"], adam_seconds=timing["adam_train_seconds"],
                    lbfgs_seconds=timing["lbfgs_train_seconds"],
                    lbfgs_iterations=timing["lbfgs_iterations_completed"],
                    extra_gradient_evaluations=timing["lbfgs_gradient_evaluations"],
                    stop_reason=timing["stop_reason"], **diagnostics))
                queries.extend(dict(case=case, seed=seed, repeat=i, seconds=seconds)
                               for i, seconds in enumerate(repeats, 1))
                path = temp / f"{case}_seed{seed}.pt"
                torch.save(dict(case=case, train_config=asdict(model.config),
                                equation=dict(a=model.problem.a, D=model.problem.D, T=model.problem.T),
                                weights=model.state_dict()), path)
                reloaded = load_model(path)
                check = torch.tensor([[.4, .03], [.8, .5]], dtype=DTYPE)
                with torch.no_grad():
                    torch.testing.assert_close(reloaded(check), model(check), atol=0., rtol=0.)
                print(f"{case} seed={seed}  train={timing['train_seconds']:.2f}s  "
                      f"held-out residual={diagnostics['residual_rms']:.5g}", flush=True)

        write_csv(temp / "profiles.csv", profiles)
        write_csv(temp / "timings.csv", timings)
        write_csv(temp / "query_timings.csv", queries)
        settings = dict(cases=args.cases, seeds=args.seeds, adam_steps=args.adam_steps,
                        lbfgs_iterations=args.lbfgs_iterations, space_intervals=64, times=EVAL_TIMES,
                        default_config=asdict(TrainConfig(steps=args.adam_steps,
                                                          lbfgs_iterations=args.lbfgs_iterations)),
                        environment=environment_info(), run_seconds=time.perf_counter() - start)
        (temp / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        for file in temp.iterdir():
            file.replace(args.output / file.name)


if __name__ == "__main__":
    main()
