# Issue 326 benchmark utilities

This directory contains scripts for parsing and plotting timing data from
`gpuRefactor` and `kluRefactor`.

The scripts support ReSolve issue 326. They do not change solver behavior.

## Scripts

* `parse_refactor_logs.py`: parses raw benchmark logs into CSV.
* `plot_refactor_results.py`: generates timing and residual plots from the parsed CSV.

## Dependencies

`parse_refactor_logs.py` uses only the Python standard library.

`plot_refactor_results.py` requires `matplotlib`:

```bash
python3 -m pip install matplotlib
```

A temporary virtual environment can be used if needed:

```bash
python3 -m venv /tmp/resolve-plot-venv
source /tmp/resolve-plot-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install matplotlib
```

Do not commit virtual environments, logs, CSV files, or generated plots.

## Timing row format

Run the examples with `-t` or `--time` to emit timing rows:

```text
TIMING,example,backend,ir_enabled,system,time_ms
```

Example:

```text
TIMING,gpuRefactor,CUDA,0,2,2.4852799099999999e+02
```

Column meanings:

```text
TIMING       Marker used to identify timing rows
example      Example executable name
backend      Hardware/backend label
ir_enabled   0 for no iterative refinement, 1 for iterative refinement
system       Linear system index
time_ms      Solve time in milliseconds
```

## Expected benchmark cases

Issue 326 uses the GridKit cases on Frontier for:

```text
N=125
N=250
N=500
N=1000
```

For each `N`, collect logs for:

```text
kluRefactor
kluRefactor -i
gpuRefactor
gpuRefactor -i
```

Run the benchmark cases for both CUDA and HIP builds.

## Output directory

Use an output directory outside tracked source files:

```bash
mkdir -p issue326_outputs/logs
mkdir -p issue326_outputs/plots
```

To keep generated files out of local `git status`, add the output directory to
the local exclude file:

```bash
echo "issue326_outputs/" >> .git/info/exclude
```

## Frontier log collection

The Python scripts do not run the benchmark executables. They only process logs
after the benchmark runs are complete.

Create logs by running the examples and saving output with `tee`.

Example CUDA run without iterative refinement:

```bash
./build-cuda/examples/gpuRefactor.exe \
  -m <matrix_prefix> \
  -r <rhs_prefix> \
  -n <num_systems> \
  -t | tee issue326_outputs/logs/cuda_N125_gpu.log
```

Example CUDA run with iterative refinement:

```bash
./build-cuda/examples/gpuRefactor.exe \
  -m <matrix_prefix> \
  -r <rhs_prefix> \
  -n <num_systems> \
  -i \
  -t | tee issue326_outputs/logs/cuda_N125_gpu_ir.log
```

Use log names that include the backend, problem size, and method:

```text
cuda_N125_klu.log
cuda_N125_klu_ir.log
cuda_N125_gpu.log
cuda_N125_gpu_ir.log
hip_N125_klu.log
hip_N125_klu_ir.log
hip_N125_gpu.log
hip_N125_gpu_ir.log
```

Repeat the same naming pattern for:

```text
N250
N500
N1000
```

The `N` value in the log filename allows `parse_refactor_logs.py` to infer the
GridKit problem size automatically.

## Parse logs

After logs are collected, parse them into one CSV file:

```bash
python3 scripts/issue326/parse_refactor_logs.py \
  issue326_outputs/logs/*.log \
  -o issue326_outputs/issue326_timings.csv
```

If the GridKit problem size cannot be inferred from the log filename, pass it
explicitly:

```bash
python3 scripts/issue326/parse_refactor_logs.py \
  issue326_outputs/logs/*.log \
  --N 125 \
  -o issue326_outputs/issue326_timings.csv
```

The parser writes:

```text
source_log,N,example,backend,method,ir_enabled,system,time_ms,residual
```

## Generate plots

Generate plots from the parsed CSV:

```bash
python3 scripts/issue326/plot_refactor_results.py \
  issue326_outputs/issue326_timings.csv \
  -o issue326_outputs/plots
```

With all four GridKit sizes present, the script generates nine plots:

```text
N125_solve_time.png
N125_residual.png
N250_solve_time.png
N250_residual.png
N500_solve_time.png
N500_residual.png
N1000_solve_time.png
N1000_residual.png
average_solve_time_scaling.png
```

## View plots

Open the plot directory in VS Code:

```bash
code issue326_outputs/plots
```

Click each `.png` file in the VS Code Explorer to preview it.

If working through VS Code Remote SSH on Frontier, open the generated plot
directory there. Otherwise, copy the output directory back to a local machine
and open the plots locally.

## Local smoke test

A local smoke test can be run using existing ReSolve test logs and a placeholder
`N` value:

```bash
python3 scripts/issue326/parse_refactor_logs.py \
  /tmp/klu_timing_test.log \
  /tmp/klu_timing_ir_test.log \
  /tmp/gpu_timing_test.log \
  /tmp/gpu_timing_ir_test.log \
  --N 2000 \
  -o /tmp/issue326_local_timings.csv

python3 scripts/issue326/plot_refactor_results.py \
  /tmp/issue326_local_timings.csv \
  -o /tmp/issue326_local_plots
```

To view temporary local smoke-test plots in VS Code:

```bash
rm -rf issue326_local_plots
mkdir -p issue326_local_plots
cp /tmp/issue326_local_plots/*.png issue326_local_plots/
code issue326_local_plots
```

After checking the images, remove the temporary folder:

```bash
rm -rf issue326_local_plots
```

The local smoke test only verifies that parsing and plotting work. The final
benchmark should use the GridKit cases on Frontier.



