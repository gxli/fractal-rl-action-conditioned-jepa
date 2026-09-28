# Fractal RL: Action-Conditioned JEPA for Continuous Control

Research code for comparing **Soft Actor-Critic (SAC)** with SAC augmented by an action-conditioned Joint-Embedding Predictive Architecture (JEPA) on a continuous-control navigation task. An agent moves through periodic, fractal potential landscapes using local terrain observations and bounded longitudinal/lateral acceleration.

The implementation includes the environment and terrain generator, SAC/PPO baselines, JEPA variants and regularizers, experiment configurations, evaluation utilities, and analysis/visualization scripts.

## Repository layout

```text
fractal_rl/
├── config/       # Base configurations and experimental overlays
├── driver/       # Training, evaluation, plotting, and analysis entry points
├── src/          # Environment, terrain, physics, RL, and utility modules
├── tests/        # Automated tests
├── docs/         # Project reports and their LaTeX sources
└── requirements.txt
```

Generated data—checkpoints, replay/training logs, plots, videos, and local PDF-rendering files—are intentionally excluded from version control. Runs write to `fractal_rl/dumps/` by default.

## Setup

Requires Python 3.10 or newer. From the repository root:

```bash
cd fractal_rl
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

Install a PyTorch build appropriate for your CPU or CUDA platform if the generic `pip` installation is not suitable; see the [official PyTorch instructions](https://pytorch.org/get-started/locally/).

## Quick start

All commands below run from `fractal_rl/`.

```bash
# Optional: create and preview a terrain
python -m driver.generate_terrain

# Small SAC smoke run
python -m driver.train_sac \
  --set training.total_timesteps=10000 \
  --set environment.num_parallel_envs=4

# SAC + action-conditioned JEPA
python -m driver.train_sac \
  --overlay config/sac_jepa.yaml \
  --set training.total_timesteps=10000 \
  --set environment.num_parallel_envs=4
```

Training checkpoints, resolved configurations, metrics, and plots are saved under `dumps/`. The default experiment is substantially larger (`5,000,000` timesteps); use the overrides above for a quick validation run.

## Evaluation and analysis

```bash
# Replace with a checkpoint emitted by training
python -m driver.evaluate dumps/<run>/checkpoints/sac_<step>.pt

# Produce diagnostic plots for a completed run
python -m driver.flush_analysis dumps/<run>
```

See `config/default.yaml` for the shared environment/optimization settings, `config/sac.yaml` for SAC, and `config/sac_jepa.yaml` plus the named overlays for JEPA ablations and transfer experiments.

## Environment

Each episode uses a mean-centered Gaussian potential field sampled periodically over a 512×512 terrain. The agent observes a local 32×32 terrain patch in a world-aligned frame plus velocity, mass, and prior-action state. The reward is traveled distance with optional control costs, so policies learn to sustain speed while avoiding costly climbs and exploiting connected valleys.

## Reproducibility and safety

Experiment outputs are not committed because checkpoints can be large and `torch.load` may deserialize Python objects. Only load checkpoints from trusted sources. Every run writes its resolved configuration alongside its outputs; preserve that file with exported results to reproduce a run.

## Reports

Research reports and their sources are available in [`fractal_rl/docs/`](fractal_rl/docs/). They document the evolving SAC/JEPA comparisons; generated working PDFs and rendering intermediates are excluded.

## License

No license has been selected yet. Add a `LICENSE` file before publishing if you want others to be able to reuse the code.
