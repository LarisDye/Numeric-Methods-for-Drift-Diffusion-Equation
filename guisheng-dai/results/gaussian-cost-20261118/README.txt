Gaussian direct-convolution cost experiment, 2026-11-18

Numerical problem
  u_t = a u_x + D u_xx on [0,1], homogeneous Dirichlet boundaries.
  a=0.2, D=0.05, T=0.5, f(x)=exp(-2x)*sin(pi*x).
  Every solve queries all N+1 nodes, including both endpoints.
  The numerical algorithm in src/gaussian_convolution.cpp and
  R/gaussian_convolution.R was reused without changes.

Configurations
  kernel_width: N=512, p=2, c=1,2,3,4,6,8.
  parallel_small: N=512, c=6, p=1,2,4,8.
  parallel_large: N=8192, c=6, p=1,2,4,8.
  K=ceil(c*sqrt(2DT)*N), W=2K+1.

Timing
  Windows 11, R 4.6.1, Intel Core i7-11800H (8 physical / 16 logical cores).
  All timings use the existing steady_clock C++ wall clock.
  Three calls are measured separately: prepare_gaussian; solve_gaussian on
  prepared input; and prepare_gaussian followed by solve_gaussian for new input.
  New-input total times are independent measurements, not sums of medians.
  Small fluctuations can therefore make total and prepared-solve medians overlap.
  R startup, compilation, calibration, warmup, pre-batch gc, validation and I/O
  are excluded from individual timed batches. Each sample is a batch mean;
  median, Q1 and Q3 are calculated from nine batch means per configuration.
  Configurations and timing phase order rotate across rounds. Batches are
  calibrated toward 0.2 s; actual batches ranged from 0.095 to 0.370 s.
  OMP_DYNAMIC=FALSE, OMP_MAX_ACTIVE_LEVELS=1, OMP_WAIT_POLICY=PASSIVE;
  OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1. Armadillo internal OpenMP is
  disabled in the pre-existing source; explicit outer OpenMP remains enabled.
  Speedup = median one-thread solve time / median p-thread solve time.
  The baseline is the identical solve_gaussian wrapper with one thread, whose
  OpenMP region is serialized by if(threads>1); it is not a separately compiled
  non-OpenMP executable. Thread counts were measured inside the parallel region.

Files and validation
  raw/timings.csv: 378 samples (14 configurations x 9 rounds x 3 phases).
  raw/checks.csv: 581 successful checks, including actual team size, endpoints,
  equality across thread counts and unchanged numerical source.
  Cross-thread outputs were bitwise identical for all tested configurations.
  R scalar summation independently verifies a small convolution example.
  c=8 and c=10 agree to less than 1e-12, validating the wide-kernel reference.
  config.R/rds, environment.txt, compiler_and_teams.txt/rds, batch_sizes.csv,
  source snapshots, and MD5 source_hashes.csv retain the experiment context.
  Official timing-phase elapsed duration was 90.818 s; this duration includes
  untimed validation/checkpoint activity and is not an individual solve time.

Accuracy convention
  The initial run wrote all-node RMSE into raw/accuracy.csv. After timing,
  scripts/08a_gaussian_internal_rmse.R re-evaluated the same configurations
  without measuring performance, verified the numerical source hashes and
  reproduced the original all-node RMSE. It then computed internal-node RMSE
  to match the report convention. Original raw timings and accuracy are intact.
  raw/accuracy_internal.csv contains the correction checks; derived summary
  files use rmse=rmse_internal and retain rmse_all explicitly.
  The current 08_benchmark_gaussian_cost.R directly records both conventions
  for future runs. The original executed script remains in source/ with its
  original hash. The postprocessing script and hash are also retained.

Reproduce from guisheng-dai/ using a new run identifier
  Rscript --vanilla scripts/08_benchmark_gaussian_cost.R --run-id gaussian-cost-NEW
  Existing result directories are refused rather than overwritten.

Selected results
  N=8192,c=6,W=21983,8193 outputs: prepared solve medians are
  173.954, 91.422, 46.463, 24.823 ms for p=1,2,4,8; speedups
  are 1.000,1.903,3.744,7.008. Internal-node RMSE is 1.970931e-10.
  N=512,c=6: prepared solve medians are 0.704,0.444,0.287,0.239 ms;
  speedups are 1.000,1.585,2.454,2.947. Small tasks gain less from threads.
  For N=512,p=2,c=1,2,3,4,6,8, prepared solve medians are
  0.119,0.181,0.255,0.311,0.439,0.567 ms.

Plot reproduction and accuracy-only postprocessing
  From guisheng-dai/, redraw the saved result without rerunning a solver:
    python scripts/09_plot_gaussian_cost.py gaussian-cost-20261118
  This reads this run's summary/summary.csv and the existing E3 accuracy in
  results/openmp-final-20261117/summary/kernel.csv. It writes PDF/PNG figures
  to presentation/report/figures/. The plotting script snapshot is retained
  in source/09_plot_gaussian_cost.py with plot_source_hashes.csv; the original
  execution source_hashes.csv is unchanged.

  The accuracy-only postprocessing command used for this archived run was:
    Rscript --vanilla scripts/08a_gaussian_internal_rmse.R gaussian-cost-20261118
  It recomputes errors only; it does not repeat or modify timing measurements.
  This postprocessing has already completed, and its results are saved in
  raw/accuracy_internal.csv and summary/summary.csv/rds. There is no need to
  repeat it for this archive. Original raw/accuracy.csv remains intact.
