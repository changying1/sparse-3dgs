# I2-D1 Receiver-v2 audit

## Decision

**GO for one Receiver-v2 K20 5k run; not a quality claim.**

The geometry/initialization audit, CPU regressions, and the 1201→1252 short GPU
run passed.  The 5k launcher is prepared but was only run in preflight mode.
K remains 20 and the frozen 20 Donor UIDs are unchanged.  No test view was used
to construct, select, rank, or parameterize Receiver-v2.

## Frozen inputs

- Source checkpoint SHA256: `1d5ee65d9bb1f7b50b2d8ea6bd277e5b52e513b212d0216e3a205e014b428c1f`
- Receiver-v1 plan SHA256: `2ca1e203848515a1c596368c03b46b04ae6dde85176c81e847ea732cc64ac9c2`
- Receiver-v2 plan SHA256: `0bcd58837112fcf06cbb9041cda09afb9a97a3b0034845deb24f463f94827f46`
- Receiver-v2 evidence SHA256: `e1c3472cf9fb74d54dfd5c74b45ef27c4c4e1ed8fcf6f41dac6f7d24b22d9c08`
- Initialization table SHA256: `1fd42d0f9b51961f26a21cd9cde0c4aa4f9e65a970c8eabd611200c7d65c810a`

The original checkpoint, v1 plan, and all three completed 5k metric files retain
their pre-work hashes.

## Receiver-v2 formula

For anchor position `X`, activated scale `s`, copied rotation matrix `R`, source
camera projection `pi`, and residual pixel `p`:

1. Select the two columns of `R` associated with the two largest anchor scales,
   producing tangent basis `B`.
2. Evaluate the analytic perspective projection Jacobian `J = d pi(X) / d X`.
3. Solve `delta = B (J B)^-1 (p - pi(X))`.
4. Clamp `||delta|| <= 2 ||s_tangent||`; reject if the 2×2 condition number is
   above 25, the displacement is below `0.1 min(s_tangent)`, or the projected
   residual does not decrease sufficiently.
5. In all three training cameras require positive depth and a locally bounded
   depth shift.  Rasterizer depth is used only to veto conspicuous front
   occlusion; it is never back-projected to construct XYZ.

Initialization:

- Activated opacity: `0.05`.
- Tangent scale component `i`:
  `clip(2 px / ||J r_i||, 0.20 s_i, 0.60 s_i)`.
- Normal scale: `0.35 s_normal`.
- Rotation and SH: copied from the unchanged anchor.
- Geometry-valid candidates are ranked by the training-block first-order color
  alignment `dot(mean(GT-render), anchor_SH_RGB-mean(render))`.

Of the 200 Receiver-v1 candidates, 32 passed the geometry/cross-view constraints;
the top 20 were selected.  Only one Receiver ID overlaps the v1 K20 selection.
The full XYZ, displacement, opacity, scale and projection evidence is in
`reports/i2_receiver_v2_horns_1201_k20/receiver_v2_initialization.csv` and
`receiver_v2_geometry_evidence.json`.

## Immediate training-view audit

Full-float, dropout-free renders at iteration 1201:

| Arm | PSNR | SSIM | LPIPS | L1 | Selected-block L1 |
|---|---:|---:|---:|---:|---:|
| Release-only | 22.175896 | 0.800569 | 0.231567 | 0.043597465 | 0.116371531 |
| Receiver-v1 | 22.137576 | 0.800389 | 0.231687 | 0.043696779 | 0.115984128 |
| Receiver-v2 | 22.175903 | 0.800565 | 0.231565 | 0.043597542 | 0.116340933 |

Receiver-v2 minus Release-only selected-block L1 is `-3.0598e-5`; outside-block
L1 is `+2.2593e-7`; global L1 is `+7.6989e-8`.  All predeclared risk gates pass.
This is an initialization feasibility audit, not final reconstruction evidence.

## CPU regression

- Receiver-v1/v2 focused suite: **15 passed**.
- All repository tests except the old I1 external-fixture file: **185 passed**.
- Full `tests/` collection: **223 passed, 14 failed**.  All 14 failures are in
  `test_innovation1_phase1.py` because it expects the absent external path
  `output/A3_A3_Horns_3v_Baseline_5k_FrozenPDC/view_manifest.json`.

## 1252 short GPU result

The run completed 50 optimizer updates with 65,876 aligned Gaussian/UID/A0/B0
and optimizer rows.

| Arm | PSNR | SSIM | LPIPS | L1 |
|---|---:|---:|---:|---:|
| Control | 19.811228 | 0.750995 | 0.250286 | 0.06036148 |
| Release-only K20 | 19.917764 | 0.752535 | 0.249213 | 0.05938135 |
| Receiver-v1 K20 | 19.928527 | 0.752577 | 0.249236 | 0.05931219 |
| Receiver-v2 K20 | 19.919268 | 0.752579 | 0.249229 | 0.05937379 |

Receiver-v2 is slightly better than Release-only in PSNR, SSIM and L1, and
slightly worse in LPIPS (`+1.65e-5`).  Relative to v1 it is mixed: lower PSNR
and worse L1, but slightly higher SSIM and lower LPIPS.  These deltas are tiny
and are subject to the existing non-strict RNG-pairing caveat.

## 5k preparation

`tools/run_i2_receiver_v2_5k.py` passed its default read-only preflight.  It
creates no output unless explicitly invoked with `--execute`; no v2 5k training
was started in this task.

