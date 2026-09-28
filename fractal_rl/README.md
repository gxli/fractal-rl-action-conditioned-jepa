# Fractal potential-field RL

Python 3.10+ implementation of a continuous 2D agent moving in a **periodic 512×512 mean-centered Gaussian potential field**, with conservative field acceleration, linear velocity damping, signed longitudinal and perpendicular control acceleration, local velocity-aligned 32×32 potential observations, and a CNN PPO baseline.

## Install

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Run commands **from this repository's root**:

```bash
python -m driver.generate_terrain
python -m driver.train --set training.total_timesteps=10000 --set environment.num_parallel_envs=4
python -m driver.evaluate dumps/checkpoints/ppo_000010000.pt
python -m driver.make_movie dumps/checkpoints/ppo_000010000.pt --steps 500 --format gif
```

Training counts whole vectorized rollout batches, so the final checkpoint can slightly exceed the requested timesteps if that number is not divisible by the number of environments. Use the actual checkpoint name printed in `dumps/checkpoints/`.

Additional YAML overlays: `python -m driver.train --overlay config/ppo.yaml --set physics.potential_scale=0.0`. To enable the longitudinal mechanical-power cap, add `--overlay config/power_limited.yaml`. Defaults are in `config/default.yaml`; the other config files are **optional overlays**, not implicitly merged.

## Physical definitions

Positions are unbounded physical coordinates, but periodic bilinear terrain sampling uses coordinates modulo `resolution * pixel_size`. FFT filtering of real white noise creates a periodic low-frequency-dominated Gaussian field `I` with radial power ~ frequency^(-beta). The finite-sample mean is subtracted and the result is standardized before defining the physical potential as `phi = I^2`; this makes the potential-observation scale stable across seeds. The default `beta=3` deliberately produces broad basins and connected valleys rather than pixel-scale corrugation. The environmental acceleration is `a_phi = -potential_scale * grad(phi) / g95`, where `g95` is the configurable 95th percentile of the potential-gradient magnitude. Linear damping is `a_resist = -linear_damping * velocity`; the default `linear_damping=0.6` bounds flat-field steady speed near 3.3 for maximum forward control. Propulsion is commanded **acceleration**, not force. Optionally, `physics.power_limit` caps longitudinal mechanical power: `|a_parallel| <= min(max_forward_acceleration, power_limit / (mass * max(speed, power_speed_epsilon)))`; `null` leaves the acceleration-only baseline unchanged. At near-zero speed the last stored heading defines the control axes. Both controls are clamped to [-1, 1]. The semi-implicit solver uses fixed substeps.

The observation image is rotated into the direction of travel, with the top of the patch forward and horizontal positive to the agent's right. The state vector contains local forward/right velocity, speed, mass, and previous throttle/steering. No absolute position or terrain seed is exposed. The displayed movie FOV outline is axis-aligned for readability; the *actual* observation rotates with heading.

Reward is the estimated distance integral during the step (`speed * dt`), optionally minus control penalties. With fixed episode length it maximizes mean speed, **not** net progress; in the potential landscape the policy must spend its bounded acceleration to avoid uphill traps and follow connected low-potential corridors. An episode lasts 2,000 steps x 0.02 = 40 time units. Its start position is sampled uniformly from the periodic terrain tile by default (`environment.random_start_position=true`), so consecutive episodes do not always begin at the same location. Episodes end by time-limit truncation. GAE bootstraps the terminal observation for truncations but resets advantage propagation at episode boundaries.

## Outputs

- `dumps/terrains/`: numeric terrain and PNG preview.
- `dumps/checkpoints/`: model and optimizer checkpoint, resolved configuration and RNG states. Checkpoints contain Python objects and should only be loaded from trusted sources.
- `dumps/metrics/`: PPO training CSV and evaluation JSON.
- `dumps/plots/`: learning curves and potential-overlaid latent-trajectory UMAPs.
- `dumps/latents/`: 128-D shared-encoder trajectory dumps in CSV and compressed NPZ formats.
- `dumps/trajectories/`: per-step trajectory CSV.
- `movies/`: MP4 (requires ffmpeg) or GIF.

Evaluation uses held-out terrain seeds `[100001..100004]`; ensure your chosen training seed range does not overlap these. Training automatically uses a fixed bank of distinct seeds; the initial set of environments is distributed over that bank.

## Notes

This is a functional PPO baseline, not a guaranteed trained high-speed policy. Default 5-million-step training is computationally substantial. The prior checkpoint was trained with the old resistance-dependent drag and is not comparable to this revised potential-field task; retrain before drawing conclusions. For a no-field ablation override `physics.potential_scale=0.0`. The FFT-derived gradient is consistent with the Fourier interpolant, while the physics queries it using bilinear interpolation; this is an approximate numerical derivative of the bilinearly sampled potential. Use smaller `dt`/more substeps if integration is unstable. The FFT and terrain bank require memory proportional to `terrain_count * resolution²`.
