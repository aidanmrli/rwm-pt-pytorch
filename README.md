# rwm-pt-pytorch

GPU-accelerated reference implementation of Random Walk Metropolis (RWM) and Parallel Tempering RWM (PT-RWM) algorithms for high-dimensional, multi-modal Bayesian inference research. The codebase contains reproducible experiments, optimized CUDA kernels, and utilities for analysing sampler efficiency via acceptance rates and expected squared jump distance (ESJD).

## Key Features
- **CPU & GPU samplers**: Baseline NumPy implementations alongside optimized PyTorch GPU algorithms with fused kernels and batched updates.
- **Flexible proposals**: Drop-in `Normal`, `Laplace`, and `UniformRadius` proposal distributions with automatic temperature-aware scaling.
- **Rich target library**: Analytic and PyTorch-native targets (rough carpet, multi-mixture, Rosenbrock, funnels, hypercubes, IID products) for benchmarking.
- **Experiment harness**: High-level `MCMCSimulation` interfaces for generating chains, computing ESJD, plotting trace/histograms, and saving artefacts.
- **Cluster ready**: Example Slurm batch scripts for large-scale GPU sweeps and automated plotting helpers for post-processing.

## Installation
Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

For editable development with tooling:

```bash
python -m pip install -e .[dev]
```

### GPU Dependencies
The GPU implementations rely on a CUDA-enabled PyTorch build. Install the CUDA wheels that match your driver/toolkit as documented on [pytorch.org](https://pytorch.org/). The CPU implementations work with the CPU-only build.

## Quick Start
### 1. Run a CPU experiment
```bash
python experiment.py
```
Configurable options inside the script let you change target distributions, dimensions, and sampler parameters while automatically reporting acceptance rates and ESJD.

### 2. Run GPU parallel tempering
```bash
python example_pt_gpu.py
```
This walks through three progressively harder scenarios and prints timing, swap acceptance, and ESJD diagnostics. The script falls back to CPU when CUDA is unavailable.

### 3. Launch a single GPU simulation interactively
```python
from algorithms import ParallelTemperingRWM_GPU_Optimized
from target_distributions import RoughCarpetTorch

target = RoughCarpetTorch(dim=20)
pt = ParallelTemperingRWM_GPU_Optimized(
    dim=20,
    var=0.9,
    target_dist=target,
    burn_in=2000,
    pre_allocate_steps=10000,
    swap_every=10,
    geom_temp_spacing=True,
)
samples = pt.generate_samples(10000)
print("Swap acceptance:", pt.swap_acceptance_rate)
print("PT ESJD:", pt.expected_squared_jump_distance_gpu())
```
Switch proposal families by passing a `NormalProposal`, `LaplaceProposal`, or `UniformRadiusProposal` from `proposal_distributions`.

## Project Layout
- `algorithms/` – Core CPU and GPU RWM / PT-RWM implementations plus proposal abstractions.
- `interfaces/` – High-level simulation APIs (`MCMCSimulation`, `MCMCSimulation_GPU`, `TargetDistribution`, etc.).
- `proposal_distributions/` – Modular proposal family implementations and utilities.
- `target_distributions/` – Library of analytic and PyTorch-native target densities for benchmarking.
- `data/` – Cached JSON experiment outputs used by plotting utilities.
- `images/` – Generated figures (trace plots, histograms, ESJD curves).
- `example_pt_gpu.py`, `experiment*.py` – Ready-to-run experiment entry points.
- `plot.py` – Helper that turns averaged JSON logs into publication-quality figures.
- `run_*.sbatch` – Sample Slurm scripts for HPC deployments.
- `tests/` – Regression and performance smoke tests for GPU kernels and proposal logic.

## Development Workflow
1. Install dev dependencies: `python -m pip install -e .[dev]`
2. Run targeted checks (GPU tests require CUDA hardware):
   ```bash
   pytest tests/test_rwm_correctness.py
   pytest tests/test_pt_gpu.py
   ```
3. Format/lint before commits:
   ```bash
   ruff check .
   black .
   ```

## Reproducing Figures
1. Execute experiments (CPU or GPU) to populate `data/` with JSON summaries.
2. Generate ESJD plots with `python plot.py`, which writes PNGs into `images/averaged/`.
3. Use `images/publishing/` artefacts for trace plots and histograms created via `MCMCSimulation` helpers.

## Citation
If this repository informs your research, please cite the associated article:

```
@article{li2025exploring,
   author = {Aidan Li and Liyan Wang and Tianye Dou and Jeffrey S. Rosenthal},
   title = {Exploring the generalizability of the optimal 0.234 acceptance rate in random-walk metropolis and parallel tempering algorithms},
   journal = {Communications in Statistics - Simulation and Computation},
   volume = {0},
   number = {0},
   pages = {1--31},
   year = {2025},
   publisher = {Taylor \& Francis},
   doi = {10.1080/03610918.2025.2544242},
   URL = { 
         https://doi.org/10.1080/03610918.2025.2544242
   },
   eprint = { 
         https://doi.org/10.1080/03610918.2025.2544242
   }
}
```

## License
Released under the terms of the MIT License. See `LICENSE` for details.
