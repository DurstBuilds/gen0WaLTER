"""Headless evaluation for the state-machine tumble policy.

Examples:
  python scripts/eval_state_machine.py --controller oracle --mode probe
  python scripts/eval_state_machine.py --controller checkpoint --mode table \\
    --checkpoint logs/rsl_rl/.../model_500.pt

``table`` rolls every rough-test cell and prints traverse, fall, hard-contact,
time to exit, mid-flight stops, and tumble time on flat ground and slopes.
``probe`` runs the scripted climber on the top rough stair row and compares
its return with always-drive and always-tumble.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from walter_rl.assets.walter_fs.walter_constants import NOMINAL_ROOT_HEIGHT
from walter_rl.rl.ppo import state_machine_ppo_runner_cfg
from walter_rl.rl.state_machine_env import StateMachineEnv
from walter_rl.rl.vecenv import StateMachineVecEnv
from walter_rl.tasks.velocity.config.walter_fs.env_cfgs import (
  walter_fs_state_machine_env_cfg,
)
from walter_rl.tasks.velocity.mdp.rewards import climb_gate

_COMMAND_MPS = 1.0
_APPROACH_M = 3.2
_OBSTACLE_EXIT_M = 3.35
_OPEN_EXIT_M = 3.5
_SLOW_MPS = 0.05
_STOP_WINDOW_S = 1.0
_APPROACH_COLUMNS = ("pyramid_stairs", "random_grid")
_OPEN_COLUMNS = ("flat", "hf_pyramid_slope", "hf_pyramid_slope_inv", "wave_terrain")
_SPAWN_CLEARANCE_M = 0.05


@dataclass
class _Rollout:
  """Rates gathered from one batched episode."""

  traverse: float
  fall: float
  hard_contact: float
  time_to_exit_s: float
  midflight_stop: float
  tumble_on_open: float
  reward_per_s: float
  per_column: list[str]


def main() -> None:
  """Run the requested evaluation and print one line per result."""
  args = _parse_args()
  device = _resolve_device(args.device)
  if args.mode in ("probe", "both"):
    _run_probe(device, args.trials, args.seconds)
  if args.mode in ("table", "both"):
    _run_table(device, args)


def _run_probe(device: str, trials: int, seconds: float) -> None:
  """Score the oracle on the top stair row, then compare three controllers."""
  climbed = _stair_probe(device, trials, seconds)
  print(
    "[probe] top inverted stairs "
    f"traverse={climbed.traverse:.2f} fall={climbed.fall:.2f} "
    f"hard_contact={climbed.hard_contact:.2f} "
    f"time_to_exit_s={climbed.time_to_exit_s:.2f}"
  )
  scores = {
    name: _reward_rate(device, name, None, trials, seconds)
    for name in ("oracle", "drive", "tumble")
  }
  print(
    "[probe] reward/s "
    + " ".join(f"{name}={value:.3f}" for name, value in scores.items())
  )
  if scores["oracle"] <= scores["drive"] or scores["oracle"] <= scores["tumble"]:
    print("[probe] oracle did not out-score both always-drive and always-tumble.")


def _run_table(device: str, args: argparse.Namespace) -> None:
  """Print one metrics line per terrain column on ``--task``."""
  model = _load_policy(args, device) if args.controller == "checkpoint" else None
  result = _evaluate(
    device,
    args.task,
    args.controller,
    model,
    args.trials,
    args.seconds,
    top_row_only=False,
    stair_only=False,
  )
  print(
    f"[table] {args.task} {args.controller} "
    f"traverse={result.traverse:.2f} fall={result.fall:.2f} "
    f"hard_contact={result.hard_contact:.2f} "
    f"time_to_exit_s={result.time_to_exit_s:.2f} "
    f"midflight_stop={result.midflight_stop:.2f} "
    f"tumble_on_open={result.tumble_on_open:.2f} "
    f"reward/s={result.reward_per_s:.3f}"
  )
  for line in result.per_column:
    print(f"[table] {line}")


def _stair_probe(device: str, trials: int, seconds: float) -> _Rollout:
  """Oracle climb from the pit on the hardest inverted-stair row."""
  return _evaluate(
    device,
    "rough",
    "oracle",
    None,
    trials,
    seconds,
    top_row_only=True,
    stair_only=True,
  )


def _reward_rate(
  device: str,
  controller: str,
  model,
  trials: int,
  seconds: float,
) -> float:
  """Mean reward per second of one controller on the rough-test grid."""
  return _evaluate(
    device,
    "rough-test",
    controller,
    model,
    trials,
    seconds,
    top_row_only=False,
    stair_only=False,
  ).reward_per_s


def _evaluate(
  device: str,
  task: str,
  controller: str,
  model,
  trials: int,
  seconds: float,
  *,
  top_row_only: bool,
  stair_only: bool,
) -> _Rollout:
  """Roll ``trials`` episodes on each selected cell and pool the rates."""
  cfg = _eval_cfg(task, seconds)
  cfg.scene.num_envs = _prepare_env_count(task, trials, top_row_only, stair_only)
  env = StateMachineEnv(cfg, device=device)
  try:
    wrapped = StateMachineVecEnv(env, clip_actions=1.0)
    names = _assign_cells(env, trials, top_row_only=top_row_only, stair_only=stair_only)
    obs, _extras = wrapped.reset()
    _place_approaches(env, names)
    policy = model
    if controller == "checkpoint":
      policy = _roll_with_checkpoint(model, wrapped, device)
    return _roll(wrapped, obs, controller, policy, names, seconds)
  finally:
    env.close()


def _eval_cfg(task: str, seconds: float):
  """State-machine scene with a fixed 1 m/s command and no curriculum."""
  cfg = walter_fs_state_machine_env_cfg(play=False, task=task)
  cfg.curriculum = {}
  for name in (
    "skill_progress",
    "skill_spawn",
    "traversal_progress",
    "traversal_spawn",
    "push_robot",
    "randomize_terrain",
  ):
    cfg.events.pop(name, None)
  cfg.episode_length_s = seconds
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.02, 0.02),
    "yaw": (0.0, 0.0),
  }
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (_COMMAND_MPS, _COMMAND_MPS)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.rel_standing_envs = 0.0
  twist.zero_ang_vel_prob = 1.0
  twist.resampling_time_range = (1.0e6, 1.0e6 + 1.0)
  return cfg


def _assign_cells(
  env: StateMachineEnv,
  trials: int,
  *,
  top_row_only: bool,
  stair_only: bool,
) -> list[str]:
  """Give every environment a row and column, repeated ``trials`` times."""
  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_origins is not None
  generator = terrain.cfg.terrain_generator
  assert generator is not None
  names = list(generator.sub_terrains)
  rows = terrain.terrain_origins.shape[0]
  row_ids = [rows - 1] if top_row_only else list(range(rows))
  col_ids = [names.index("pyramid_stairs_inv")] if stair_only else list(range(len(names)))
  slots = [(row, col) for row in row_ids for col in col_ids for _ in range(trials)]
  if len(slots) != env.num_envs:
    raise SystemExit(
      f"Built {env.num_envs} envs but the cell grid needs {len(slots)}. "
      "Pass --trials that matches, or leave num_envs to this script."
    )
  # num_envs is set in _eval_cfg before the env exists. This check runs after.
  levels = torch.tensor([row for row, _col in slots], device=env.device)
  types = torch.tensor([col for _row, col in slots], device=env.device)
  terrain.terrain_levels[:] = levels
  terrain.terrain_types[:] = types
  terrain.env_origins[:] = terrain.terrain_origins[levels, types]
  return [names[col] for _row, col in slots]


def _prepare_env_count(task: str, trials: int, top_row_only: bool, stair_only: bool) -> int:
  """How many environments the cell grid needs."""
  cfg = walter_fs_state_machine_env_cfg(play=False, task=task)
  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None
  rows = 1 if top_row_only else generator.num_rows
  cols = 1 if stair_only else len(generator.sub_terrains)
  return rows * cols * trials


def _place_approaches(env: StateMachineEnv, names: list[str]) -> None:
  """Start pyramid stairs and cubes on the near apron, facing the obstacle."""
  robot = env.scene["robot"]
  terrain = env.scene.terrain
  assert terrain is not None and terrain.env_origins is not None
  origins = terrain.env_origins
  state = robot.data.default_root_state.clone()
  state[:, 0:2] = origins[:, 0:2]
  state[:, 2] = origins[:, 2] + NOMINAL_ROOT_HEIGHT
  for index, name in enumerate(names):
    if name not in _APPROACH_COLUMNS:
      continue
    state[index, 0] = origins[index, 0] - _APPROACH_M
    state[index, 1] = origins[index, 1]
    state[index, 2] = NOMINAL_ROOT_HEIGHT + _SPAWN_CLEARANCE_M
  robot.write_root_state_to_sim(state)
  env.sim.forward()
  env.sim.sense()


def _roll(wrapped, obs, controller: str, model, names: list[str], seconds: float) -> _Rollout:
  """Step until ``seconds`` and pool every environment that is still on its first life."""
  env = wrapped.unwrapped
  n = env.num_envs
  device = env.device
  done = torch.zeros(n, dtype=torch.bool, device=device)
  traversed = torch.zeros(n, dtype=torch.bool, device=device)
  fell = torch.zeros(n, dtype=torch.bool, device=device)
  hit = torch.zeros(n, dtype=torch.bool, device=device)
  exit_time = torch.full((n,), float("nan"), device=device)
  midflight = torch.zeros(n, dtype=torch.bool, device=device)
  slow_time = torch.zeros(n, device=device)
  pending = torch.zeros(n, dtype=torch.bool, device=device)
  previous = torch.zeros(n, dtype=torch.bool, device=device)
  open_steps = torch.zeros(n, device=device)
  tumble_steps = torch.zeros(n, device=device)
  reward_sum = torch.zeros(n, device=device)
  steps = max(1, round(seconds / env.step_dt))
  names_t = names
  for step_index in range(steps):
    action = _action(controller, model, obs, env)
    obs, reward, _dones, _extras = wrapped.step(action)
    terminated = env.reset_terminated.reshape(n)
    truncated = env.reset_time_outs.reshape(n)
    alive = ~done
    reward_sum += torch.where(alive, reward, torch.zeros_like(reward))
    _update_outcomes(
      env,
      names_t,
      alive,
      step_index,
      traversed,
      fell,
      hit,
      exit_time,
      midflight,
      slow_time,
      pending,
      previous,
      open_steps,
      tumble_steps,
    )
    done |= terminated.bool() | truncated.bool()
  return _pool(traversed, fell, hit, exit_time, midflight, open_steps, tumble_steps, reward_sum, seconds, names)


def _action(controller: str, model, obs, env: StateMachineEnv) -> torch.Tensor:
  """One 2-D action row per environment."""
  if controller == "checkpoint":
    return model(obs)
  n = env.num_envs
  hip = torch.full((n,), -1.0, device=env.device)
  if controller == "drive":
    tumble = torch.full((n,), -1.0, device=env.device)
  elif controller == "tumble":
    tumble = torch.ones(n, device=env.device)
  elif controller == "oracle":
    gate = climb_gate(env)
    tumble = torch.where(gate > 0.0, torch.ones(n, device=env.device), -torch.ones(n, device=env.device))
  else:
    raise SystemExit(f"Unknown controller {controller}.")
  return torch.stack((tumble, hip), dim=-1)


def _update_outcomes(
  env,
  names,
  alive,
  step_index,
  traversed,
  fell,
  hit,
  exit_time,
  midflight,
  slow_time,
  pending,
  previous,
  open_steps,
  tumble_steps,
) -> None:
  """Record traverse, failures, and tumble time for environments still on the first life."""
  robot = env.scene["robot"]
  delta = (robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]).abs()
  reach = torch.maximum(delta[:, 0], delta[:, 1])
  radius = _exit_radius(env, names)
  just_out = alive & ~traversed & (reach >= radius)
  traversed |= just_out
  exit_time[just_out] = (step_index + 1) * env.step_dt
  fell |= alive & _fired(env, "fell_over")
  hit |= alive & _fired(env, "torso_contact")
  applied = env._tumble_applied
  released = alive & previous & ~applied & (climb_gate(env) > 0.0)
  slow = robot.data.root_link_lin_vel_b[:, 0].abs() < _SLOW_MPS
  pending[:] = (pending | released) & slow & alive
  slow_time[:] = torch.where(pending, slow_time + env.step_dt, torch.zeros_like(slow_time))
  midflight |= pending & (slow_time >= _STOP_WINDOW_S)
  pending[:] = pending & (slow_time < _STOP_WINDOW_S)
  previous[:] = applied & alive
  on_open = alive & _open_mask(names, env.device)
  open_steps += on_open.float()
  tumble_steps += (on_open & applied).float()


def _exit_radius(env, names: list[str]) -> torch.Tensor:
  """Chebyshev distance that counts as leaving the obstacle."""
  radius = [
    _OBSTACLE_EXIT_M if name in ("pyramid_stairs", "pyramid_stairs_inv", "random_grid") else _OPEN_EXIT_M
    for name in names
  ]
  return torch.tensor(radius, device=env.device)


def _open_mask(names: list[str], device: str) -> torch.Tensor:
  """True on flat, slope, and wave columns."""
  return torch.tensor([name in _OPEN_COLUMNS for name in names], device=device)


def _fired(env, name: str) -> torch.Tensor:
  """Termination flag, or false when the term was removed."""
  if name not in env.termination_manager.active_terms:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return env.termination_manager.get_term(name)


def _pool(
  traversed,
  fell,
  hit,
  exit_time,
  midflight,
  open_steps,
  tumble_steps,
  reward_sum,
  seconds: float,
  names: list[str],
) -> _Rollout:
  """Mean rates, plus a per-column text line stored on the result."""
  exited = exit_time[torch.isfinite(exit_time)]
  time_to_exit = float(exited.mean()) if exited.numel() else float("nan")
  open_total = float(open_steps.sum())
  tumble_on_open = float(tumble_steps.sum() / open_total) if open_total else 0.0
  return _Rollout(
    traverse=float(traversed.float().mean()),
    fall=float(fell.float().mean()),
    hard_contact=float(hit.float().mean()),
    time_to_exit_s=time_to_exit,
    midflight_stop=float(midflight.float().mean()),
    tumble_on_open=tumble_on_open,
    reward_per_s=float(reward_sum.mean() / seconds),
    per_column=_column_lines(
      names, traversed, fell, hit, exit_time, midflight, open_steps, tumble_steps
    ),
  )


def _column_lines(names, traversed, fell, hit, exit_time, midflight, open_steps, tumble_steps) -> list[str]:
  """One summary line for each terrain column."""
  lines = []
  device = traversed.device
  for name in dict.fromkeys(names):
    mask = torch.tensor([item == name for item in names], device=device)
    if not bool(mask.any()):
      continue
    exited = exit_time[mask & torch.isfinite(exit_time)]
    exit_s = float(exited.mean()) if exited.numel() else float("nan")
    open_total = float(open_steps[mask].sum())
    tumble = float(tumble_steps[mask].sum() / open_total) if open_total else 0.0
    lines.append(
      f"{name} traverse={float(traversed[mask].float().mean()):.2f} "
      f"fall={float(fell[mask].float().mean()):.2f} "
      f"hard_contact={float(hit[mask].float().mean()):.2f} "
      f"time_to_exit_s={exit_s:.2f} "
      f"midflight_stop={float(midflight[mask].float().mean()):.2f} "
      f"tumble_fraction={tumble:.2f}"
    )
  return lines


def _load_policy(args: argparse.Namespace, device: str):
  """Deterministic actor from a state-machine checkpoint."""
  if args.checkpoint is None:
    raise SystemExit("--controller checkpoint requires --checkpoint PATH")
  # The runner is built inside _evaluate, where the observation shape is known.
  # This loader only checks the file here. The model is created per scene.
  del device
  from pathlib import Path

  if not Path(args.checkpoint).is_file():
    raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
  return args.checkpoint


def _roll_with_checkpoint(path: str, wrapped, device: str):
  """Build the runner on ``wrapped`` and return its inference policy."""
  agent_cfg = state_machine_ppo_runner_cfg(
    max_iterations=1,
    experiment_name="walter_fs_eval",
  )
  runner = OnPolicyRunner(wrapped, agent_cfg, log_dir=None, device=device)
  runner.load(path)
  return runner.get_inference_policy(device=device)


def _resolve_device(requested: str) -> str:
  """Use the GPU when the user asked for automatic selection and one exists."""
  if requested != "auto":
    return requested
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--controller", choices=("oracle", "drive", "tumble", "checkpoint"), default="oracle")
  parser.add_argument("--mode", choices=("probe", "table", "both"), default="probe")
  parser.add_argument("--task", choices=("rough", "rough-test"), default="rough-test")
  parser.add_argument("--checkpoint", type=str, default=None)
  parser.add_argument("--trials", type=int, default=4)
  parser.add_argument("--seconds", type=float, default=12.0)
  parser.add_argument("--device", type=str, default="auto")
  args = parser.parse_args()
  if args.trials < 1:
    raise SystemExit("--trials must be at least 1.")
  return args


if __name__ == "__main__":
  main()

