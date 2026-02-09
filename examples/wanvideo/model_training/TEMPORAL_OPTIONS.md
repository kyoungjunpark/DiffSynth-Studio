# Temporal Loss Options — Quick Guide ✅

This note summarizes the new temporal loss options added to `train_with_ground_truth.py` and gives recommended flag combinations and defaults for quick experiments.

## New CLI flags
- `--use_per_frame_mse` (default: False)
  - Compute per-frame MSE across all T frames and apply monotonic weights w_t = (t/(T-1))^p (normalized). Helps make intermediate frames progressively closer to GT.
- `--temporal_frame_power` (default: 1.0)
  - Power `p` in w_t = (t/(T-1))^p. Use `1.0` for linear increase, `2.0` to emphasize later frames more.
- `--use_pseudo_target_interp` (default: False)
  - Create pseudo-targets for intermediate frames by linear interpolation between first and last training targets. Currently uses latent-space interpolation (set `--pseudo_interp_space latent`).
- `--pseudo_interp_space` (default: `latent`)
  - Space for interpolation: `latent` (recommended) or `image` (not implemented; currently falls back to `latent`).
- `--use_monotonicity_loss` (default: False)
  - Adds penalty for any increase in distance-to-GT across frames: sum_t max(0, dist(t+1) - dist(t)).
- `--monotonicity_loss_weight` (default: 0.0)
  - Scale for the monotonicity penalty (small values recommended, e.g., `0.001`–`0.01`).

## How these interact with existing options
- `--temporal_loss_weight` (existing) still controls the frame-to-frame *reward* (encourages change). The new data terms (per-frame MSE / pseudo-targets) replace or complement the old last-frame MSE depending on flags.
- `--temporal_normalize` and `--temporal_warmup_steps` work the same as before and apply to the temporal reward term.
- The training logger now saves both raw and MA loss curves by default: `loss_curve.png` (raw) and `loss_curve_ma.png` + `loss_history_ma.json`.

## Recommended quick setups 💡
- Baseline (same behavior as before):
  - No extra flags (only `--temporal_loss_weight` if you want the change reward).

- Simple, recommended (fast to try): monotonic per-frame MSE
  - `--use_per_frame_mse --temporal_frame_power 1.0`
  - Optionally combine with `--temporal_loss_weight 0.01 --temporal_normalize --temporal_warmup_steps 5000`
  - Effect: encourages intermediate frames to gradually move toward GT, minimal extra compute.

- More precise (smoother intermediate targets): pseudo-target interpolation
  - `--use_pseudo_target_interp --pseudo_interp_space latent`
  - Combine with per-frame MSE if desired: `--use_per_frame_mse --use_pseudo_target_interp`
  - Effect: generates smoother transitions; slightly more compute but still lightweight in latent space.

- Enforce monotonic improvement (guard against oscillation):
  - `--use_monotonicity_loss --monotonicity_loss_weight 0.001`
  - Use with per-frame MSE or pseudo interpolation for best effect.

## Suggested defaults to start with
- `--use_per_frame_mse --temporal_frame_power 1.0 --use_pseudo_target_interp --pseudo_interp_space latent --use_monotonicity_loss --monotonicity_loss_weight 0.001 --temporal_loss_weight 0.01 --temporal_normalize --temporal_warmup_steps 5000`

## Quick debug tips
- Run with a small `--dataset_repeat` or a subset of data to sanity-check losses and generated frames.
- Check `loss_curve_ma.png` (smoothed) to inspect long-term trends and `loss_history_ma.json` for numeric MA values.

---
If you'd like, I can add a commented example command into `train_coco_ground_truth.sh` and/or a short unit test that exercises these options. Which would you prefer next?