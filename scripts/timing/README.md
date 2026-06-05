# GridKit timing utilities

This directory contains scripts for parsing and plotting timing data from
`gpuRefactor`, `kluRefactor`, `gluRefactor`, and `sysRefactor`.

The scripts support the GridKit timing workflow. They do not change solver behavior.

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

## Benchmark workflow

The timing workflow is split between Frontier and a local CUDA machine.

Frontier is used for:

- CPU/KLU timing logs from `kluRefactor` and `kluRefactor -i`
- HIP timing logs from `gpuRefactor`, `gpuRefactor -i`, `sysRefactor`, and `sysRefactor -i`

The local CUDA machine is used for:

- CUDA timing logs from `gpuRefactor`, `gpuRefactor -i`, `gluRefactor`, `sysRefactor`, and `sysRefactor -i`
- combining CPU, HIP, and CUDA logs into one CSV
- generating the final plots

Recommended workflow:

1. Configure and build ReSolve on Frontier with HIP and KLU support.
2. Run the CPU/KLU timing cases with `kluRefactor` and `kluRefactor -i` on Frontier.
3. Run the HIP timing cases with `gpuRefactor`, `gpuRefactor -i`, `sysRefactor`, and `sysRefactor -i` on Frontier.
4. Copy the CPU/KLU and HIP logs from Frontier to the local machine.
5. Copy the GridKit matrices from Frontier to the local machine if they are not already present.
6. Configure and build ReSolve locally with CUDA support.
7. Run the CUDA timing cases with `gpuRefactor`, `gpuRefactor -i`, `sysRefactor`, `sysRefactor -i`, and `gluRefactor` locally.
8. Parse all collected `kluRefactor`, `gpuRefactor`, `sysRefactor`, and `gluRefactor` logs into one CSV.
9. Generate the final plots from the combined CSV.

Only the benchmark executables need the appropriate compute environment. Parsing,
CSV generation, and plotting can be done locally after the logs are collected.

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

The timing workflow uses the GridKit cases on Frontier for:

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
gluRefactor
sysRefactor
sysRefactor -i
```

Run the benchmark cases for the supported backends. `gpuRefactor` and `sysRefactor` support CUDA and HIP builds while `gluRefactor` is only CUDA. `kluRefactor` provides the CPU/KLU timing baseline.

## Output directory

Use an output directory outside tracked source files:

```bash
mkdir -p gridkit_outputs/logs
mkdir -p gridkit_outputs/plots
```

To keep generated files out of local `git status`, add the output directory to
the local exclude file:

```bash
echo "gridkit_outputs/" >> .git/info/exclude
```

## Frontier environment setup

Before collecting Frontier logs, configure a Frontier environment with HIP,
ROCm, KLU, and the AMD GPU target available. The exact module stack is
site-specific and may change, so verify the environment before configuring
ReSolve.

Check that the HIP compiler is visible:

```bash
which hipcc
hipcc --version
echo "$ROCM_PATH"
```

Then configure a HIP/KLU build from the ReSolve source directory:

```bash
cd /ccs/home/$USER/resolve/source

cmake -S . -B ../build-hip-klu \
  -DCMAKE_BUILD_TYPE=Release \
  -DRESOLVE_USE_HIP=ON \
  -DRESOLVE_USE_CUDA=OFF \
  -DRESOLVE_USE_KLU=ON

cmake --build ../build-hip-klu -j
```

Confirm that the expected benchmark executables were built:

```bash
find ../build-hip-klu \
  -path '*gpuRefactor.exe' -o \
  -path '*kluRefactor.exe' -o \
  -path '*sysRefactor.exe' | sort
```

Run HIP benchmark commands from an allocated Frontier compute node.

## Frontier log collection

The Python scripts do not run the benchmark executables. They only process logs
after the benchmark runs are complete.

Create logs by running the examples and saving output with `tee`.

On Frontier, collect the CPU/KLU logs with `kluRefactor` and the HIP logs with
`gpuRefactor` and `sysRefactor`. Run HIP benchmark commands from an allocated
compute node.

Example CPU/KLU run without iterative refinement:

```bash
../build-hip-klu/examples/kluRefactor.exe \
  -m <matrix_prefix> \
  -r <rhs_prefix> \
  -n <num_systems> \
  -t | tee gridkit_outputs/logs/cpu_N125_klu.log
```

Example CPU/KLU run with iterative refinement:

```bash
../build-hip-klu/examples/kluRefactor.exe \
  -m <matrix_prefix> \
  -r <rhs_prefix> \
  -n <num_systems> \
  -i \
  -t | tee gridkit_outputs/logs/cpu_N125_klu_ir.log
```

Example HIP `gpuRefactor` run without iterative refinement:

```bash
../build-hip-klu/examples/gpuRefactor.exe \
  -m <matrix_prefix> \
  -r <rhs_prefix> \
  -n <num_systems> \
  -t | tee gridkit_outputs/logs/hip_N125_gpu.log
```

Example HIP `gpuRefactor` run with iterative refinement:

```bash
../build-hip-klu/examples/gpuRefactor.exe \
  -m <matrix_prefix> \
  -r <rhs_prefix> \
  -n <num_systems> \
  -i \
  -t | tee gridkit_outputs/logs/hip_N125_gpu_ir.log
```

Example HIP `sysRefactor` run without iterative refinement:

```bash
../build-hip-klu/examples/sysRefactor.exe \
  -m <matrix_prefix> \
  -r <rhs_prefix> \
  -n <num_systems> \
  -t | tee gridkit_outputs/logs/hip_N125_sys.log
```

Example HIP `sysRefactor` run with iterative refinement:

```bash
../build-hip-klu/examples/sysRefactor.exe \
  -m <matrix_prefix> \
  -r <rhs_prefix> \
  -n <num_systems> \
  -i \
  -t | tee gridkit_outputs/logs/hip_N125_sys_ir.log
```

Use log names that include the backend, problem size, and method:

```text
cpu_N125_klu.log
cpu_N125_klu_ir.log
hip_N125_gpu.log
hip_N125_gpu_ir.log
hip_N125_sys.log
hip_N125_sys_ir.log
```

Repeat the same naming pattern for:

```text
N250
N500
N1000
```

The `N` value in the log filename allows `parse_refactor_logs.py` to infer the
GridKit problem size automatically.

## Local CUDA log collection

Collect CUDA logs on a local CUDA machine using the same GridKit matrices.

Example CUDA `gpuRefactor` run without iterative refinement:

```bash
./build-cuda/examples/gpuRefactor.exe \
  -m <matrix_prefix> \
  -r <rhs_prefix> \
  -n <num_systems> \
  -t | tee gridkit_outputs/logs/cuda_N125_gpu.log
```

Example CUDA `gpuRefactor` run with iterative refinement:

```bash
./build-cuda/examples/gpuRefactor.exe \
  -m <matrix_prefix> \
  -r <rhs_prefix> \
  -n <num_systems> \
  -i \
  -t | tee gridkit_outputs/logs/cuda_N125_gpu_ir.log
```

Example CUDA `gluRefactor` run without iterative refinement:

```bash
./build-cuda/examples/gluRefactor.exe \
  -m <matrix_prefix> \
  -r <rhs_prefix> \
  -n <num_systems> \
  -t | tee gridkit_outputs/logs/cuda_N125_glu.log
```

Example CUDA `sysRefactor` run without iterative refinement:

```bash
./build-cuda/examples/sysRefactor.exe \
  -m <matrix_prefix> \
  -r <rhs_prefix> \
  -n <num_systems> \
  -t | tee gridkit_outputs/logs/cuda_N125_sys.log
```

Example CUDA `sysRefactor` run with iterative refinement:

```bash
./build-cuda/examples/sysRefactor.exe \
  -m <matrix_prefix> \
  -r <rhs_prefix> \
  -n <num_systems> \
  -i \
  -t | tee gridkit_outputs/logs/cuda_N125_sys_ir.log
```

Use log names that include the backend, problem size, and method:

```text
cuda_N125_gpu.log
cuda_N125_gpu_ir.log
cuda_N125_glu.log
cuda_N125_sys.log
cuda_N125_sys_ir.log
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

After CPU/KLU, HIP, and CUDA logs are collected, parse them into one CSV file:

```bash
python3 scripts/timing/parse_refactor_logs.py \
  gridkit_outputs/logs/*.log \
  -o gridkit_outputs/gridkit_timings.csv
```

If the GridKit problem size cannot be inferred from the log filename, pass it
explicitly:

```bash
python3 scripts/timing/parse_refactor_logs.py \
  gridkit_outputs/logs/*.log \
  --N 125 \
  -o gridkit_outputs/gridkit_timings.csv
```

The parser writes:

```text
source_log,N,example,backend,method,ir_enabled,system,time_ms,residual
```

## Generate plots

Generate plots from the parsed CSV:

```bash
python3 scripts/timing/plot_refactor_results.py \
  gridkit_outputs/gridkit_timings.csv \
  -o gridkit_outputs/plots
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
code gridkit_outputs/plots
```

Click each `.png` file in the VS Code Explorer to preview it.

If working through VS Code Remote SSH on Frontier, open the generated plot
directory there. Otherwise, copy the output directory back to a local machine
and open the plots locally.

## Local smoke test

A local smoke test can be run using existing ReSolve test logs and a placeholder
`N` value:

```bash
python3 scripts/timing/parse_refactor_logs.py \
  /tmp/klu_timing_test.log \
  /tmp/klu_timing_ir_test.log \
  /tmp/gpu_timing_test.log \
  /tmp/gpu_timing_ir_test.log \
  /tmp/glu_timing_test.log \
  /tmp/sys_timing_test.log \
  /tmp/sys_timing_ir_test.log \
  --N 2000 \
  -o /tmp/gridkit_local_timings.csv

python3 scripts/timing/plot_refactor_results.py \
  /tmp/gridkit_local_timings.csv \
  -o /tmp/gridkit_local_plots
```

To view temporary local smoke-test plots in VS Code:

```bash
rm -rf gridkit_local_plots
mkdir -p gridkit_local_plots
cp /tmp/gridkit_local_plots/*.png gridkit_local_plots/
code gridkit_local_plots
```

After checking the images, remove the temporary folder:

```bash
rm -rf gridkit_local_plots
```

The local smoke test only verifies that parsing and plotting work. The final
benchmark should use the GridKit cases on Frontier.
