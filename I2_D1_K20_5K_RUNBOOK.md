# I2-D1-K20 5k launch runbook

## Frozen inputs

- Git HEAD: `2fda731ae7390a1bf2ad637aa7dcc7a329b1dba0`
- Source checkpoint: `output/I2_B0_Horns_3v_1201_Growth_z98bpswi/chkpnt1201.pth`
- Source SHA-256: `1d5ee65d9bb1f7b50b2d8ea6bd277e5b52e513b212d0216e3a205e014b428c1f`
- K20 v2 plan SHA-256: `2ca1e203848515a1c596368c03b46b04ae6dde85176c81e847ea732cc64ac9c2`
- View manifest SHA-256: `a43d544c48f4767369ab37cf9ff82106f1d64365cb0e03d129559d297108586b`
- Training views: `DJI_20200223_163017_967.jpg`, `DJI_20200223_163053_863.jpg`, `DJI_20200223_163225_243.jpg`

The launcher performs the full plan/artifact validation and rechecks every hash
before creating an output directory.

## Review-only preflight

```powershell
.\script\run_i2_d1_k20_5k.ps1
```

The default invocation only prints the three resolved commands. It does not
create output directories or start GPU training.

## Launch after approval

Run all arms sequentially:

```powershell
.\script\run_i2_d1_k20_5k.ps1 -Arm All -Execute
```

Or run one arm:

```powershell
.\script\run_i2_d1_k20_5k.ps1 -Arm Control -Execute
.\script\run_i2_d1_k20_5k.ps1 -Arm Release -Execute
.\script\run_i2_d1_k20_5k.ps1 -Arm Reallocation -Execute
```

Each arm starts from the frozen iteration-1201 checkpoint and refuses to reuse
an existing experiment directory. The global final iteration is 5000, giving
3798 optimizer steps because iteration 5000 evaluates/saves without an optimizer
step. Evaluation is scheduled at 1500, 2000, 3000, 4000 and 5000. Checkpoints
are saved at 2000, 3500 and 5000; the final PLY is saved at 5000.

## Outputs

Each experiment directory contains:

- `train.log`: merged stdout/stderr for the complete training process;
- `launch_meta.json`: command, Git HEAD, code/input hashes, configuration,
  elapsed time and randomness caveats;
- `chkpnt2000.pth`, `chkpnt3500.pth`, `chkpnt5000.pth`;
- `point_cloud/iteration_5000/point_cloud.ply`;
- `formal_metrics_5000.json`: aggregate and per-view PSNR, SSIM, LPIPS and L1;
- `evaluation.log`.

After all arms exist, the launcher writes
`output/I2_D1_K20_5K_comparison.json`, including Gaussian counts, training time,
and Release-only/Reallocation deltas.

The formal test metric uses the same in-memory float render, clamp, test camera
population and metric functions as `train.py` evaluation. It does not pass
through PNG. Formal training metrics cover each of the three unique training
views once, whereas `train.py` logs five indexed samples with repeats.

## Randomness and recovery

All fresh arms call `safe_state(0)`, but the checkpoint does not contain RNG
state and equal seeds are not per-UID Dropout pairing. Release-only uses
`max(65876, current_gaussian_count)` draws. Its event log records the first
iteration at which growth exceeds 65,876; strict cross-arm RNG-stream alignment
must not be claimed from that iteration onward. Even before that event, stream
alignment is conditional on the other arms still drawing exactly 65,876 values;
their own densification may invalidate that condition earlier.

The intermediate checkpoints provide model/optimizer/A0/B0 recovery artifacts.
However, restarting a process from one of them resets RNG state because RNG is
not stored in checkpoints. Such a resumed run must be labelled as resumed and
must not be mixed into the formal fresh-from-1201 comparison. For the formal
comparison, rerun a failed arm from the frozen 1201 checkpoint into a new clean
directory.
