"""训练基线，比较同一起点上的两种优化方式，再单独改变软约束的边界权重。"""

import argparse
import csv
import gzip
import json
import platform
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pinn import (Problem, TrainConfig, environment_info, evaluate_model, initialize_training,
                  lbfgs_options, load_checkpoint, refine_lbfgs, save_checkpoint, train_adam)

ERROR_COLUMNS = ["terminal_rmse", "terminal_max_error", "spacetime_rmse",
                 "initial_rmse", "boundary_max_error", "residual_rms"]


def write_csv(path, rows):
    open_file = gzip.open if path.suffix == ".gz" else open
    with open_file(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def evaluate(model):
    # 65 个节点的查询是以后和差分对比的共同任务。更密的时空网格只作独立验证。
    metrics, common = evaluate_model(model, 64, 40)
    dense_metrics, dense = evaluate_model(model, 256, 40)
    errors = {key: metrics[key] for key in ERROR_COLUMNS}
    errors["spacetime_rmse"] = dense_metrics["spacetime_rmse"]
    return errors, metrics, common, dense


def loss_records(history, seed, constraint, stage):
    for record in history:
        yield {
            "seed": seed, "constraint": constraint, "stage": stage,
            "record": record.get("stage", "adam"), "adam_updates": record.get("step"),
            "gradient_evaluations": record.get("gradient_evaluations", record["step"]),
            **{key: record[key] for key in ("total", "pde", "initial", "boundary")},
        }


def record_model(model, stage, base_seconds, extra_seconds, setup, evaluations, iterations,
                 stop_reason, history, records, output, example_seed):
    errors, bench, common, dense = evaluate(model)
    seed, constraint = model.config.seed, model.config.constraint
    key = {"constraint": constraint, "seed": seed, "stage": stage}
    train_seconds = base_seconds + extra_seconds
    row = dict(key, extra_gradient_evaluations=evaluations, lbfgs_iterations=iterations,
               stop_reason=stop_reason, baseline_train_seconds=base_seconds,
               extra_train_seconds=extra_seconds, train_seconds=train_seconds,
               setup_seconds=setup, query_seconds=bench["query_seconds"],
               total_seconds=setup+train_seconds+bench["query_seconds"], **errors)
    records["metrics"].append(row)
    records["loss"].extend(loss_records(history, seed, constraint, stage))
    records["query"].extend(dict(key, repeat=i, query_seconds=seconds)
                            for i, seconds in enumerate(bench["query_repeats_seconds"], start=1))
    # 各方案完全相同的坐标和真值只存一份。
    if not records["grids"]:
        records["grids"].update(x=dense["x"], t=dense["t"], exact=dense["exact"])
    records["grids"][f"{constraint}_{seed}_{stage}"] = dense["prediction"]
    if seed == example_seed:
        for x, pred, exact in zip(common["x"], common["terminal_prediction"], common["terminal_exact"]):
            records["terminal"].append(dict(key, x=x, t=model.problem.T, prediction=pred,
                                            exact=exact, abs_error=abs(pred-exact)))
        if constraint == "hard" and stage == "adam_then_lbfgs":
            checkpoint = output / f"hard_seed{example_seed}.pt"
            save_checkpoint(checkpoint, model)
            restored = load_checkpoint(checkpoint)
            _, loaded = evaluate_model(restored, 64, 40)
            np.testing.assert_array_equal(loaded["terminal_prediction"], common["terminal_prediction"])
    print(f'{constraint} seed={seed} {stage}: terminal RMSE={errors["terminal_rmse"]:.7g}; '
          f'train={train_seconds:.3f}s, extra gradients={evaluations}', flush=True)


def paired_run(problem, config, iterations, records, output, example_seed):
    model, points, adam, setup = initialize_training(problem, config)
    history, base = train_adam(model, points, adam, config.steps)
    record_model(model, "adam_baseline", base, 0., setup, 0, 0,
                 "requested_adam_steps", history, records, output, example_seed)

    # 用副本精修，原模型及其 Adam 动量将用于继续 Adam 的配对对照。
    refined = deepcopy(model)
    lbfgs_history = []
    timing = refine_lbfgs(refined, points, iterations, history=lbfgs_history)
    extra = timing["lbfgs_gradient_evaluations"]
    record_model(refined, "adam_then_lbfgs", base, timing["lbfgs_train_seconds"],
                 setup+timing["lbfgs_setup_seconds"], extra, timing["lbfgs_iterations_completed"],
                 timing["stop_reason"], lbfgs_history, records, output, example_seed)

    continuation, seconds = train_adam(model, points, adam, extra, start_step=config.steps)
    record_model(model, "adam_continued", base, seconds, setup, extra, 0,
                 "matched_full_batch_gradients", continuation, records, output, example_seed)


def boundary_sweep(problem, seeds, steps, metrics):
    rows = []
    for seed in seeds:
        baseline = next(row for row in metrics if row["constraint"] == "soft" and
                        row["seed"] == seed and row["stage"] == "adam_baseline")
        for weight in (0.1, 1., 10.):
            if weight == 1.:
                errors = {key: baseline[key] for key in ERROR_COLUMNS}
            else:
                config = TrainConfig(constraint="soft", seed=seed, steps=steps, lambda_boundary=weight)
                model, points, optimizer, _ = initialize_training(problem, config)
                train_adam(model, points, optimizer, steps)
                errors, _, _, _ = evaluate(model)
            rows.append(dict(seed=seed, lambda_boundary=weight, **errors))
            print(f'boundary weight={weight:g} seed={seed}: terminal RMSE={errors["terminal_rmse"]:.7g}', flush=True)
    return rows


def hardware_info():
    if platform.system() == "Darwin":
        program = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True,
                                 text=True, check=False)
        kept = ("Chip:", "Total Number of Cores:", "Memory:")
        return [line.strip() for line in program.stdout.splitlines() if line.strip().startswith(kept)]
    return [platform.machine(), platform.processor()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--adam-steps", type=int, default=2000)
    parser.add_argument("--lbfgs-iterations", type=int, default=400)
    args = parser.parse_args()
    if min(args.adam_steps, args.lbfgs_iterations) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Adam 步数和 L-BFGS 次数须为正，初始化种子不可重复。")
    example_seed = 7 if 7 in args.seeds else args.seeds[0]
    problem = Problem()
    args.output.mkdir(parents=True, exist_ok=True)
    records = {"metrics": [], "query": [], "terminal": [], "loss": [], "grids": {}}
    start = time.perf_counter()
    # 中断时保护上次完整结果。成功后只更新本程序负责的文件，保留图表。
    with TemporaryDirectory(dir=args.output) as temp_dir:
        temp = Path(temp_dir)
        for seed in args.seeds:
            for constraint in ("soft", "hard"):
                config = TrainConfig(seed=seed, constraint=constraint, steps=args.adam_steps)
                paired_run(problem, config, args.lbfgs_iterations, records, temp, example_seed)
        weights = boundary_sweep(problem, args.seeds, args.adam_steps, records["metrics"])
        write_csv(temp/"metrics.csv", records["metrics"])
        write_csv(temp/f"terminal_seed{example_seed}.csv", records["terminal"])
        write_csv(temp/"query_timings.csv", records["query"])
        write_csv(temp/"loss_history.csv.gz", records["loss"])
        write_csv(temp/"boundary_weights.csv", weights)
        np.savez_compressed(temp/"spacetime_predictions.npz", **records["grids"])
        settings = {
            "problem": asdict(problem), "base_config": asdict(TrainConfig(steps=args.adam_steps)),
            "lbfgs_options": lbfgs_options(args.lbfgs_iterations), "seeds": args.seeds,
            "terminal_space_intervals": 64, "spacetime_space_intervals": 256,
            "spacetime_time_intervals": 40, "residual_points": 512, "test_seed": 102026,
            "boundary_weights": [0.1, 1, 10], "environment": environment_info(),
            "hardware": hardware_info(), "experiment_seconds": time.perf_counter()-start,
        }
        (temp/"experiment.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2)+"\n")
        for file in sorted(temp.iterdir()):
            file.replace(args.output / file.name)
    print("Results:", args.output.resolve())


if __name__ == "__main__":
    main()
