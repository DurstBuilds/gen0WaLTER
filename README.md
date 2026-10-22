# gen0WaLTER

![License](https://img.shields.io/badge/License-INSERT__LICENSE-blue)
![Build Status](https://img.shields.io/badge/Build-INSERT__STATUS-lightgrey)
![Version](https://img.shields.io/badge/Version-0.1.0-blue)

**gen0WaLTER** is a simulation package for the full-scale WaLTER robot. The package version is **0.1.0**. WaLTER is a wheeled-leg platform: four hip joints, four continuous knee joints, eight driven wheels, and a waist (`head_joint`) held at zero.

The default learned policy is a state machine on top of the classical full-body PD controller. PPO outputs one number. A positive value latches tumbling: every knee spins about +Y and every wheel holds `tumble.wheel_speed`. A non-positive value leaves the PD speed, yaw, pitch, and roll loops in charge. Each edge is held for 0.5 s. The stall-integral gate stays out of this loop; `StateMachineEnv` writes the latch with `FullBodyPdController.set_tumble_command`.

`--teacher` is the other training path. The student sends the same 16-D joint command the PD controller uses (four hips, four absolute knee angles, eight wheel velocities). `TeacherStudentEnv` stores the PD action from the state the student saw, then steps the student action.

Playback covers the classical PD controller, the smooth-tumble variant, and a state-machine checkpoint. A local browser panel can start one play or train process. How an MJCF file becomes a joint torque is in [`SIMULATION.md`](SIMULATION.md). Masses and link lengths are in [`src/walter_rl/assets/walter_fs/README.md`](src/walter_rl/assets/walter_fs/README.md).

## Key Features

- State-machine PPO: a 1-D tumble toggle, clipped to ±1, with actor and critic MLPs of sizes 256, 128, and 64. The PD controller still writes the 16 joint targets.
- Front and rear height grids (1 m × 1 m, 64 rays each), projected gravity, rated motor torques, the twist command, and 0.5 s of applied tumble memory.
- Active state-machine rewards: body-forward tracking (2.0), yaw-rate tracking (1.0), predictive knee tumble (2.0), uprightness (0.1), stuck-speed penalty (−1.0), toggle rate (−0.1), and limb-torque penalty (−0.01).
- Optional 16-D teacher-student run (`--teacher`) with MLPs of sizes 512, 256, and 128. The match reward weight is currently 0, so that term is registered and skipped.
- Classical full-body PD and a smooth-tumble variant that holds hip height level while the knees spin.
- Four scenes: `flat`, `rough`, `rough-test`, and `stairs`.
- rsl-rl 5.4 `OnPolicyRunner`, TensorBoard logs under `research/logs/training/`, and a localhost panel for one play or train job.
- Checked-in full-scale MJCF (`walter_fs.xml`), with generator motors stripped and CubeMars DC motor actuators attached.
- Viser joystick for forward speed, yaw rate, hip angle, knee angle, Tank Mode, and the tumble integral. Checkpoint playback draws the tumble decision on the motors tab.

# User Manual

## Prerequisites

| Requirement | Version / notes |
| --- | --- |
| Python | `>=3.12,<3.14` (the checked-in environment uses **3.13.13**) |
| [uv](https://docs.astral.sh/uv/) | **0.11.19** or newer (`uv_build>=0.12.1,<0.13`) |
| Operating system | Linux is the supported path. mjlab sets `MUJOCO_GL=egl` on Linux before MuJoCo imports. |
| GPU | Optional NVIDIA GPU. Training and `make_pd_env` select `cuda:0` when `torch.cuda.is_available()`, otherwise `cpu`. |
| Viewer | A browser for `--viewer viser`, or a local display for `--viewer native`. |

Pinned simulation stack (from `mjlab/pyproject.toml`):

| Package | Pin |
| --- | --- |
| torch | `2.14.0` |
| mujoco | `~=3.11.0` |
| mujoco-warp | `~=3.11.0` |
| warp-lang | `>=1.14.0` |
| viser | `>=1.0.27` |
| mjviser | `>=0.0.14` |

RL packages from the root `pyproject.toml`:

| Package | Pin |
| --- | --- |
| rsl-rl-lib | `5.4.2` |
| tensorboard | `>=2.20.0` |
| tensordict | unpinned; the rsl-rl adapter packs observations as a `TensorDict` |

## Installation

Clone the repository and sync the lockfile. `walter_rl` depends on the vendored `mjlab` tree as an editable path source.

```bash
git clone [INSERT REPOSITORY URL HERE]
cd gen0WaLTER
uv python install 3.13
uv sync
```

`uv sync` creates `.venv` and installs `walter_rl`, `mjlab`, `rsl-rl-lib`, and `tensorboard` in editable mode for the local packages. CUDA and CPU PyTorch indexes are declared as optional extras on `mjlab` (`cu128` and `cpu`). The root project does not select either extra; the lockfile resolves the default `torch==2.14.0` wheel.

## Quick Start / Usage

Play the PD controller on flat ground. Viser prints a local URL when the server starts; open that URL in a browser.

```bash
uv run python scripts/play.py --policy pd --task flat
uv run python scripts/play_pd.py --task flat
```

`scripts/play_pd.py` is the same PD scene. `scripts/play.py` also loads a state-machine checkpoint:

```bash
uv run python scripts/play.py --policy checkpoint --task rough-test \
  --checkpoint research/logs/training/walter_fs_state_machine_rough-test/<run>/model_500.pt
```

Smooth tumble uses its own script. The hips hold a flat-ground contact height while the knees spin or settle.

```bash
uv run python scripts/play_smooth_tumble.py --task rough --viewer viser
```

Train the tumble toggle. The script defaults (`--num-envs 8`, `--max-iterations 2`) are a smoke test. They are too short for a policy you would play back. The seed is 42.

```bash
uv run python scripts/train.py --task rough-test --num-envs 2048 --max-iterations 500
```

The 16-D teacher stack:

```bash
uv run python scripts/train.py --teacher --task rough-test --num-envs 2048 --max-iterations 500
```

Continue a checkpoint. `--max-iterations` counts additional updates. The new files go in a fresh timestamp directory under the log root.

```bash
uv run python scripts/train.py --task rough-test --resume research/logs/training/.../model_500.pt --max-iterations 500
```

Scalars land in the run directory. View them with:

```bash
uv run tensorboard --logdir research/logs/training
```

The panel serves play and train controls on localhost and runs one child process at a time:

```bash
uv run python scripts/panel.py
```

Open `http://127.0.0.1:8765`. `--port` changes the port.

Record one rough-test cell (PD teacher, or a 16-D checkpoint on `TeacherStudentEnv`):

```bash
uv run python scripts/eval_cells.py --controller pd --cell flat --difficulty medium
uv run python scripts/eval_cells.py --controller checkpoint --cell pyramid-stairs \
  --difficulty hard --checkpoint research/logs/training/.../model_500.pt
```

In the Viser **Twist** folder during PD or smooth-tumble playback:

1. Check **Enable** so the sliders override the sampled velocity command.
2. Set **lin_vel_x** for forward speed and **ang_vel_z** for yaw rate. Lateral speed is locked at zero.
3. Check **Tank Mode** to apply **hip_angle** and **knee_angle**. It starts unchecked. Unchecked, both commands are zero and the sliders keep their values. **hip_angle** is degrees; positive points the front hips forward and the rear hips backward. **Flip front hip** reverses the front hip and front knee zeros. **knee_angle** is degrees and shifts the knee zero; positive lowers the front inner wheels. With Tank Mode on, the rear zero moves the other way, so positive raises the rear inner wheels. The knee servo uses `tank_actuator_kp` and `tank_actuator_kd` while the box is checked.
4. Leave **Tumbling** checked so the stall integral runs. The motors tab shows **lin vel integral (m)**. Unchecked, the integral is cleared and the knees stay on the roll pose. A state-machine checkpoint ignores this checkbox; the network latch replaces the integral.
5. Leave **Orientation** checked to level pitch and roll. Unchecked, those errors are treated as zero.
6. **Rated torque cap** and **Rated speed cap** hold the DC motors on the continuous ratings. Both start unchecked, so the peak torque envelope and the no-load speed corner stay in use.
7. Keyboard: **I** / **K** step forward speed, **J** / **L** step yaw rate. **Zero** clears the velocity sliders and the pose sliders.

Build a PD environment in Python when you do not want the viewer loop:

```python
from walter_rl.control.full_body_pd.env import make_pd_env
from walter_rl.control.full_body_pd.policy import FullBodyPdPolicy

env = make_pd_env("rough", play=True, num_envs=1)
policy = FullBodyPdPolicy(env)
action = policy(env.reset()[0])
env.close()
```

`policy` ignores the observation tensor. It reads the robot state and the `twist` command.

## Configuration

There is no `.env` file. Behavior comes from CLI flags, `gains.py`, and two environment variables read by mjlab.

### Playback CLI

| Flag | Default | Choices / meaning |
| --- | --- | --- |
| `--policy` | `pd` | `play.py` only. `pd` or `checkpoint`. |
| `--checkpoint` | none | Required with `--policy checkpoint`. A `model_*.pt` file from an rsl-rl run. |
| `--task` | `flat` | `flat`, `rough`, `rough-test`, `stairs` |
| `--num-envs` | `1` | Number of parallel environments |
| `--viewer` | `viser` | `viser` (browser) or `native` (MuJoCo passive viewer) |
| `--payload` | none | Rigid mass welded to the torso top. `rider` is a 75 kg seated human. Used by the PD and smooth-tumble scripts. |

| Task | Scene |
| --- | --- |
| `flat` | Infinite plane. The PD task drops terrain height scans. Training puts a scan back so the observation width matches the other scenes. The state machine then replaces that scan with the front and rear grids. |
| `rough` | mjlab velocity mix with WaLTER caps: pyramid slopes up to 0.45, waves up to 0.12 m, stairs up to 0.12 m. |
| `rough-test` | Same mix on every third difficulty row, with stair step heights scaled by 1.5. The uniform-noise column is a random grid of cubes. Training starts on row 0. Climb rows spawn on an approach, and leaving a tile does not end the episode. |
| `stairs` | Flat (15%) plus pyramid and inverse-pyramid stairs (6–18 cm). Outside rough-test, promotion follows travel along the spawn yaw, and 80% of inverse-stair environments stay on the hardest row. |

Play mode (`play=True`, which the play scripts set) uses a very long episode, disables observation noise and pushes, and turns the terrain curriculum off for the run.

### Training CLI

| Flag | Default | Meaning |
| --- | --- | --- |
| `--task` | `flat` | Same four scenes as playback. |
| `--num-envs` | `8` | Parallel environments. The panel default is 1024. |
| `--max-iterations` | `2` | PPO updates. On `--resume`, this is the number of additional updates. |
| `--teacher` | off | Train the 16-D teacher-student policy. Off selects the tumble toggle. |
| `--teacher-only-reward` | off | Zero every reward except PD action matching. Requires `--teacher`. With the match weight at 0 this leaves the run with no reward. |
| `--lin-vel-x-min` / `--lin-vel-x-max` | `0.25` / `1.25` | Forward command bounds, m/s. Cube-root sampling still averages 1 m/s inside the default span. |
| `--ang-vel-z-min` / `--ang-vel-z-max` | `-1` / `1` | Yaw command bounds, rad/s. 75% of resamples are still zero, and 5% are a full stop. |
| `--device` | GPU if present | `cuda:0` or `cpu`. |
| `--log-root` | `research/logs/training` | Directory for new runs. Older `logs/rsl_rl/` checkpoints still load through `--resume`. |
| `--resume` | none | Checkpoint file. Weights, optimizer, and the iteration counter are restored. |

Log folder names are `walter_fs_state_machine` and `walter_fs_teacher`. Any task other than `flat` appends `_<task>`, for example `walter_fs_state_machine_rough-test`. Checkpoints are `model_<iteration>.pt`, saved every 50 updates.

PPO settings live in `src/walter_rl/rl/ppo.py`: 64 steps per environment, 5 epochs, 4 minibatches, adaptive Adam at `1e-3`, discount `0.99`, GAE `0.95`, entropy coefficient `0.005`, clip `0.2`. The logger is TensorBoard.

### State machine

`StateMachineEnv.step` reads `policy_action[:, 0] > 0` as a request to tumble. A rising edge holds tumble on for 0.5 s. A falling edge holds the stored output negative for 0.5 s so a new tumble cannot start. The PD controller then fills hips, knees, wheels, and a zero waist command. Episode reset clears the latch, the dwell counters, and the tumble spin reference.

The actor and the critic both see:

- `height_scan_front` on the head, yaw-aligned, near edge 0.1 m past the head box
- `height_scan_rear` on the torso, yaw-aligned, near edge 0.1 m past the torso box
- `projected_gravity` (actor noise ±0.05)
- `motor_torques`, actuator force divided by rated torque
- `command`, the twist
- `tumble_applied`, the applied bit over the last 0.5 s, oldest first

Actor scans add uniform noise of ±0.1 before the height is scaled by the 5 m ray length. `StateMachineVecEnv` tells rsl-rl the action dimension is 1. The environment still steps a 16-D PD action.

Reward weights are the constants at the top of `src/walter_rl/tasks/velocity/config/walter_fs/env_cfgs.py`. Zero-weight terms stay in the table and are skipped at runtime. The live state-machine terms are forward tracking, yaw tracking, uprightness, toggle rate, predictive knee tumble, limb torque, and stuck forward speed. Predictive knee tumble pays for forward knee rate when the front scan still shows a step inside about two columns of the head.

### Teacher stack

`TeacherStudentEnv` snapshots `FullBodyPdController.act()` before physics. On a closed forward-scan match gate the stored target is that reactive action. On an open gate the stored target is the same command with tumble forced on. Training steps the student action. `follow_teacher` steps the stored target instead, which is how `eval_cells.py` plays the PD teacher.

`track_teacher` is an exponential match of that stored action. Its weight, `TRACK_TEACHER_WEIGHT`, is `0.0`, so the reward manager skips it. `--teacher-only-reward` zeros every other term and leaves this one at 0.

### Controller gains

Edit `src/walter_rl/control/full_body_pd/gains.py` for the PD inner loop, and `src/walter_rl/control/smooth_tumble/gains.py` for the smooth-tumble player. Each nested dataclass is one loop. Angles are radians, rates are rad/s, and linear speeds are m/s. Hip, knee, and waist actuator gains are applied when the environment is built, so restart the process after changing them. Tank Mode knee gains (`tank_actuator_kp`, `tank_actuator_kd`) switch as soon as the checkbox changes.

| Group | Role | Notable defaults |
| --- | --- | --- |
| `PitchGains` | Level torso pitch with hip splay | `kp=6`, `kd=0.4`, actuator `800 / 100` |
| `RollGains` | Drop the downhill inner wheels | `kp=8`, `kd=0.5`, `q_max=π/2`, actuator `250 / 50`, tank actuator `250 / 50` |
| `VelocityGains` | Track forward speed on all eight wheels | `kp=15`, `full_speed_command=10` m/s, `wheel_radius=0.10`, extra speed clip `50` rad/s |
| `YawRateGains` | Skid-steer from `ang_vel_z` | `kp=5`, half-width `0.32` m, yaw limit `2.5` rad/s |
| `TumbleGains` | Integrate missed forward distance, then spin the knees | `threshold=0.2` m, `release=0.1` m, `cap=0.5` m, `leak=2.5` /s, `min_forward_cmd=0.2` m/s, `min_speed_error=0.6` m/s, `rate=3` rad/s, `wheel_speed=0` rad/s |
| `WaistGains` | Hold `head_joint` at zero | `kp=2000`, `kd=50`, effort `288` N·m |

`sign` is `+1` or `-1`. Flip it if that loop runs away.

`make_pd_env` also accepts `device`, `gains`, `easy_rough` (one 2–4 cm rough tile), `lock_speed`, `lock_heading`, `episode_length_s`, and `payload` (a name from `walter_rl.assets.payloads`, or none).

### Integral tumble

`TumbleGains` is the stall loop used by PD playback and by smooth tumble. The state-machine latch replaces it during training and checkpoint playback. Its state is missed forward distance, in meters. The Viser motors tab draws that state as **lin vel integral (m)**, with the bar scaled to `cap`.

Each control step reads body-frame forward speed, `v_x`:

- When `lin_vel_x` is above `min_forward_cmd` and `lin_vel_x - v_x` is above `min_speed_error`, the state adds that speed error times the step `dt`. The integral grows only from a forward shortfall.
- Otherwise the state leaks toward zero as `lag * exp(-leak * dt)`. `leak` is in 1/s.
- The state is clamped at `cap`, so a long stall cannot leave a large tail. Keep `cap` above `threshold`.

A hysteresis gate reads that state after the update. Tumbling turns on above `threshold` and stays on until the integral falls below `release`. While it is on, every knee reference is captured at the measured angle and then advances at `rate` about +Y. Positive +Y drops the front wheel and lifts the rear. The knee position servo remains the inner loop. Tumble replaces the roll angle and, while the gate is on, the wheel commands.

Every wheel is written to `tumble.wheel_speed` for that interval. The default is 0 rad/s. Positive matches a positive forward command. Speed tracking and skid steering resume as soon as the gate turns off, including while the knees settle.

After release, each knee continues forward at `rate` until it reaches the roll pose on the revolution latched from the spin reference. The latch is taken when tumbling ends, so a shin that passes the pose stays on that turn. Unchecking **Tumbling** clears the integral, the spin reference, and the settle. An episode reset does the same.

### Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `MUJOCO_GL` | `egl` on Linux | Rendering backend. mjlab sets this before MuJoCo imports. Use `osmesa` on machines without EGL, or `glfw` for a windowed GL context. |
| `MJLAB_WARP_QUIET` | unset (`0`) | Set to `1`, `true`, or `yes` to hide Warp kernel compile logs. |

# File Explanations (Directory Structure)

```text
gen0WaLTER/
├── pyproject.toml
├── uv.lock
├── README.md
├── SIMULATION.md
├── RL Manual.md
├── scripts/
│   ├── train.py
│   ├── play.py
│   ├── play_pd.py
│   ├── play_smooth_tumble.py
│   ├── panel.py
│   ├── eval_cells.py
│   └── check_teacher_reward.py
├── src/walter_rl/
│   ├── assets/
│   │   ├── motor_models/
│   │   ├── payloads/
│   │   └── walter_fs/
│   ├── control/
│   │   ├── full_body_pd/
│   │   └── smooth_tumble/
│   ├── panel/
│   ├── rl/
│   │   ├── state_machine_env.py
│   │   ├── teacher_env.py
│   │   ├── vecenv.py
│   │   ├── ppo.py
│   │   └── checkpoint_policy.py
│   ├── tasks/velocity/
│   │   ├── mdp/
│   │   └── config/walter_fs/env_cfgs.py
│   └── viewer/
├── research/
│   ├── journal/
│   ├── literature/
│   └── logs/
└── mjlab/
```

`.venv/` is the local uv environment and is omitted above. `research/logs/` holds training runs and cell clips.

## Component Breakdown

- **`pyproject.toml`** — Package `walter_rl` 0.1.0. Depends on the local `mjlab` checkout, `rsl-rl-lib` 5.4.2, TensorBoard, and `tensordict`.
- **`uv.lock`** — Locked dependency graph, including PyTorch, MuJoCo, Warp, Viser, and rsl-rl.
- **`scripts/train.py`** — PPO entry point. Default env is `StateMachineEnv`. `--teacher` selects `TeacherStudentEnv`.
- **`scripts/play.py`** — PD playback, or a state-machine checkpoint in Viser or the native viewer.
- **`scripts/play_pd.py`** — PD playback only.
- **`scripts/play_smooth_tumble.py`** — Smooth-tumble playback. Hips cancel the knee's ride-height change while tumbling.
- **`scripts/panel.py`** — Browser panel. Play choices are classical PD, smooth tumble, and a checkpoint. Train choices include the teacher flags and a resume path under `research/logs/training` or `logs/rsl_rl`.
- **`scripts/eval_cells.py`** — One rough-test cell, 8 s, forward command 1 m/s. Writes a clip and `frames.csv` for the PD teacher or a 16-D checkpoint.
- **`scripts/check_teacher_reward.py`** — Short rollout that compares the teacher action with zeros on the match term.
- **`SIMULATION.md`** — Path from the MJCF through mjlab and MuJoCo Warp to a DC-motor torque and the PD loops.
- **`src/walter_rl/rl/state_machine_env.py`** — Latches the 1-D toggle, holds each edge for 0.5 s, and steps the PD action.
- **`src/walter_rl/rl/teacher_env.py`** — Stores the scan-conditioned PD target, then steps the student or that target.
- **`src/walter_rl/rl/vecenv.py`** — rsl-rl 5.4 adapter. `StateMachineVecEnv` reports one action.
- **`src/walter_rl/rl/ppo.py`** — `OnPolicyRunner` dict: separate actor and critic, observation normalization, Gaussian policy.
- **`src/walter_rl/rl/checkpoint_policy.py`** — Viewer adapter. Reads the `actor` observation group and reports the stored tumble decision.
- **`src/walter_rl/control/full_body_pd/`** — Pitch, roll, wheel speed, yaw, integral tumble, and the external tumble latch.
- **`src/walter_rl/control/smooth_tumble/`** — Same loops, with level hips during tumble and settle.
- **`src/walter_rl/tasks/velocity/config/walter_fs/env_cfgs.py`** — Scene, command, reward, and observation builders, including `walter_fs_state_machine_env_cfg` and `walter_fs_teacher_env_cfg`.
- **`src/walter_rl/tasks/velocity/mdp/`** — Rewards, finite-safe observations, terrain curricula, and the rough-test drive/climb curriculum.
- **`src/walter_rl/panel/`** — Local HTTP server, the page in `index.html`, allowlisted commands, and a single job supervisor.
- **`src/walter_rl/assets/`** — MJCF, CubeMars motor models, and the `rider` payload. Motor curves are documented in [`src/walter_rl/assets/motor_models/README.md`](src/walter_rl/assets/motor_models/README.md).
- **`mjlab/`** — Vendored simulation library (version 1.6.0 in this tree): manager-based RL env, MuJoCo Warp sim, terrain generator, velocity command, and the Viser and native viewers.

# Developer Notes

## System Architecture

The code splits into four layers.

**Robot description.** `get_spec()` loads `walter_fs.xml` and prepares it for mjlab. `get_walter_fs_robot_cfg()` returns an `EntityCfg` whose articulation is three DC motor groups from `motor_models`: position servos on `.*_thigh_joint` and `.*_thigh_shin_joint`, and a velocity servo on `.*wheel_joint`. Ratings, the torque-speed curve, and the extra joint gears are documented in [`src/walter_rl/assets/motor_models/README.md`](src/walter_rl/assets/motor_models/README.md). Training-time hip and knee stiffness is 20 N·m/rad. The PD environment replaces those with the stiffer gains in `gains.py` and adds a position actuator on `head_joint`. The PD knee action is an absolute angle.

**Scenes.** `env_cfgs.py` starts from mjlab's `make_velocity_env_cfg()` and retargets it to WaLTER. The physics action vector is hip position, knee position, and wheel velocity. A torso ray cast, a small body-height scan, a belly contact sensor, and one contact sensor per wheel pair feed rewards and terminations. `walter_fs_rl_env_cfg` is the shared training scene: simple twist sampling, the tracking rewards, a waist lock, and a terrain scan. The state-machine config replaces the scan and the reward table. The teacher config adds the match term and the predictive tumble term on the torso scan. There is no `mjlab.tasks` entry point; `scripts/train.py` builds the env directly.

**Inner controller.** `FullBodyPdController.act()` fills one action row per environment:

1. Pitch PD on `projected_gravity_b.x` swings the low-end hips outward from vertical, then adds the joystick hip angle.
2. Roll PD on `projected_gravity_b.y` folds the downhill knees until the inner wheel is straight down. Tank Mode flips the rear knee zero.
3. The stall integral, when it is the active gate, spins the knees at `tumble.rate` and holds the wheels at `tumble.wheel_speed`. `set_tumble_command` replaces that gate with an external boolean mask and still runs knee spin and settle.
4. Wheels scale `lin_vel_x` so `full_speed_command` requests the motor's maximum wheel rate, add proportional speed error, and add a differential from the yaw-rate loop. The tumble gate replaces that command with `tumble.wheel_speed`.
5. The waist column is written to zero.

**Learned policy.** `StateMachineEnv` is the default. `TeacherStudentEnv` is the 16-D alternative. `RslRlVecEnvWrapper` presents the mjlab env to rsl-rl 5.4, including timeout bootstrapping. `CheckpointPolicy` matches the viewer protocol (`reset` and `__call__`). Simulation steps run in PyTorch on the device chosen by the script. The native viewer copies one environment's state back to CPU `MjData`. Viser renders through `mjviser`.

## Development Setup

Install the editable checkout and run a short train or a play script:

```bash
uv sync
uv run python scripts/train.py --num-envs 8 --max-iterations 2
uv run python scripts/play_pd.py --task flat --num-envs 1
```

`uv sync` installs both packages in editable mode (`walter_rl` from `src/`, `mjlab` from `mjlab/`). Changes under `src/walter_rl` and `mjlab/src/mjlab` apply on the next process start. Warp compiles kernels on first launch; set `MJLAB_WARP_QUIET=1` to suppress that log.

There is no database and no separate compile step. For a headless PD check that does not open a viewer, construct `make_pd_env(..., play=False)` and call `policy` once, then `env.close()`.

## Testing

This repository has no `tests/` directory and no pytest dependency. Two scripts check pieces of the teacher path by running them:

- `uv run python scripts/check_teacher_reward.py` expects the match term to prefer the teacher action. The reward manager skips the term while `TRACK_TEACHER_WEIGHT` is `0.0`, so the recorded rate stays 0. The script then divides that rate by the same weight and raises `ZeroDivisionError`.
- `uv run python scripts/eval_cells.py` writes a clip for one rough-test cell. It builds `TeacherStudentEnv`, so a state-machine checkpoint (1-D action, smaller MLP) does not load there. `scripts/play.py --policy checkpoint` is the state-machine player and hard-codes the 256/128/64 MLP.

## Contributing & Future Work

There is no `CONTRIBUTING` guide. A practical change is a focused edit plus a play session on `flat` and one uneven task (`rough` or `stairs`):

1. Fork or branch from `[INSERT REPOSITORY URL HERE]`.
2. Keep controller numbers in `gains.py`. Keep scene layout and reward weights in `env_cfgs.py`.
3. Run `uv run python scripts/play.py --policy pd --task flat` and one of `--task rough` or `--task stairs`.
4. Open a pull request against `[INSERT DEFAULT BRANCH HERE]`.

Limits visible from the current tree:

- `TRACK_TEACHER_WEIGHT` is 0, so teacher matching is configured and then skipped. Turning it back on is an edit in `env_cfgs.py`. The state-machine reward table still names `sensor_name="terrain_scan"` for that term; the match function ignores the argument, and the state machine has already replaced that sensor with the front and rear grids.
- `scripts/play.py` plays state-machine checkpoints. Teacher checkpoints belong on the 16-D stack in `eval_cells.py`.
- `src/walter_rl/tasks/__init__.py` and `src/walter_rl/control/__init__.py` still describe a PD-only package. Training imports the PD controller from `walter_rl.rl`.
- The root `pyproject.toml` description still says the package is full-body PD playback.
- `RL Manual.md` is a second manual and still describes the 16-D student as the default, with a teacher match weight of 0.5. This README is the one that matches the code.
- `research/journal/STATUS.md` records the 2026-10-14 teacher study. Later state-machine runs are the checkpoints under `research/logs/training/walter_fs_state_machine_rough-test/`.
- `mjlab/src/mjlab/utils/lab_api/math.py` and `mjlab/src/mjlab/utils/spec.py` each contain an upstream note to double-check a quaternion product and a spec helper. Those are inside the vendored library.
- License, CI, and an automated test suite are not in the tree. Build-status and license badges above are placeholders until those exist.

## License

No `LICENSE` file is present. The project license is **[INSERT LICENSE HERE]**.
