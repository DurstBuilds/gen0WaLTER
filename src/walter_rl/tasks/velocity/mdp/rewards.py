"""WaLTER velocity reward terms."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_lin_vel_xy(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward XY velocity tracking without punishing vertical speed.

  Climbing stairs needs nonzero body ``vz``. The factory tracker squares that
  component into the error and fights the climb.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  return torch.exp(-xy_error / std**2)


def knee_tumble(
  env: ManagerBasedRlEnv,
  velocity_scale: float,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sensor_name: str | None = "terrain_scan",
  rough_eps: float = 0.04,
  min_forward_cmd: float = 0.2,
  min_terrain_level: int = 2,
  min_body_vx: float = 0.15,
) -> torch.Tensor:
  """Reward forward-oriented knee winding only when climbing is useful.

  Positive knee rotation (axis +Y) drops the front wheel and lifts the rear
  so the rear wheel can roll up and over a step. Forward climbing therefore
  uses positive ``qdot`` (and negative ``qdot`` when reversing).

  Gated by non-flat stair columns at ``min_terrain_level``+, a minimum
  |forward| command, and actual body-x speed so spin-in-place cannot farm
  the term.
  """
  del sensor_name, rough_eps, min_body_vx  # Kept for config compatibility; unused.
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  cmd_x = command[:, 0]
  pref = torch.sign(cmd_x)
  pref = torch.where(cmd_x.abs() < min_forward_cmd, torch.zeros_like(pref), pref)
  knee_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  aligned = torch.relu(knee_vel * pref.unsqueeze(-1)).mean(dim=-1) / velocity_scale
  # Scale by forward body speed so spin-in-place cannot farm the term.
  body_vx = asset.data.root_link_lin_vel_b[:, 0]
  progress = torch.relu(body_vx * pref).clamp(max=1.0)
  return aligned * progress * _is_hard_stairs(env, min_terrain_level).float()


def knee_waste(
  env: ManagerBasedRlEnv,
  velocity_scale: float,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sensor_name: str = "terrain_scan",
  rough_eps: float = 0.04,
  min_forward_cmd: float = 0.2,
  min_terrain_level: int = 2,
  min_body_vx: float = 0.15,
) -> torch.Tensor:
  """Penalize knee spinning only on the flat terrain column.

  Deliberately narrow: easy stairs (level < ``min_terrain_level``) and idle
  commands on stair columns do **not** count as waste. Prefer missing a flat
  penalty over punishing a robot that is still climbing.
  """
  del sensor_name, rough_eps, command_name, min_forward_cmd, min_terrain_level
  del min_body_vx
  asset: Entity = env.scene[asset_cfg.name]
  spin = asset.data.joint_vel[:, asset_cfg.joint_ids].abs().mean(dim=-1) / velocity_scale
  return spin * _is_flat_column(env).float()


def knee_tumble_reverse(
  env: ManagerBasedRlEnv,
  velocity_scale: float,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sensor_name: str | None = "terrain_scan",
  rough_eps: float = 0.04,
  min_forward_cmd: float = 0.2,
  min_terrain_level: int = 2,
  min_body_vx: float = 0.15,
) -> torch.Tensor:
  """Count wrong-way knee wind (front-lift) on hard stairs.

  Used as a penalty so legs cannot tumble with the front wheel up while
  climbing forward. Same body-vx gate as ``knee_tumble``.
  """
  del sensor_name, rough_eps, min_body_vx
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  cmd_x = command[:, 0]
  pref = torch.sign(cmd_x)
  pref = torch.where(cmd_x.abs() < min_forward_cmd, torch.zeros_like(pref), pref)
  knee_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  wrong = torch.relu(-knee_vel * pref.unsqueeze(-1)).mean(dim=-1) / velocity_scale
  body_vx = asset.data.root_link_lin_vel_b[:, 0]
  progress = torch.relu(body_vx * pref).clamp(max=1.0)
  return wrong * progress * _is_hard_stairs(env, min_terrain_level).float()


def _is_flat_column(env: ManagerBasedRlEnv) -> torch.Tensor:
  """True only for envs assigned to the ``flat`` sub-terrain column."""
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_generator is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
  if "flat" not in names:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return terrain.terrain_types == names.index("flat")


def _is_hard_stairs(env: ManagerBasedRlEnv, min_terrain_level: int) -> torch.Tensor:
  """True on non-flat stair columns at or above ``min_terrain_level``.

  Terrain-scan height std overlaps flat and stairs (~0.06 m both), so the
  curriculum column index is the reliable gate.
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_generator is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  levels = terrain.terrain_levels
  return (~_is_flat_column(env)) & (levels >= min_terrain_level)


def drive_forward(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward positive body-x speed whenever a forward command is active.

  Looser than exact velocity tracking so the policy keeps covering ground on
  inverted stairs instead of stalling while matching a hard setpoint.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  forward = (command[:, 0] > 0.1).float()
  return torch.relu(asset.data.root_link_lin_vel_b[:, 0]) * forward


def climb_vz(
  env: ManagerBasedRlEnv,
  sensor_name: str = "terrain_scan",
  rise_eps: float = 0.04,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward upward world velocity when the height scan sees uneven terrain.

  Encourages gaining elevation on stairs without rewarding hops on flat ground.
  """
  asset: Entity = env.scene[asset_cfg.name]
  up = torch.relu(asset.data.root_link_lin_vel_w[:, 2])
  rough = _terrain_height_std(env, sensor_name)
  return up * (rough > rise_eps).float()


def _terrain_height_std(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-env standard deviation of valid terrain-scan hit heights."""
  sensor = env.scene[sensor_name]
  hit_z = sensor.data.hit_pos_w[..., 2]
  valid = sensor.data.distances >= 0
  fill = torch.where(valid, hit_z, torch.zeros_like(hit_z))
  count = valid.sum(dim=-1).clamp(min=1).to(hit_z.dtype)
  mean = fill.sum(dim=-1) / count
  centered = torch.where(valid, hit_z - mean.unsqueeze(-1), torch.zeros_like(hit_z))
  var = (centered * centered).sum(dim=-1) / count
  return torch.sqrt(var.clamp(min=0.0))


def head_level(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared head-roll error from zero (use a negative reward weight).

  The head hinge is about the forward axis. Zero keeps the head level with
  the torso; any sideways flop increases this term.
  """
  asset: Entity = env.scene[asset_cfg.name]
  roll = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(roll), dim=-1)


def wheel_wheel_collision(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Count wheel pairs that are touching.

  Each ``wheel_pair_*`` sensor watches one unordered pair. The count is the
  penalty signal; the reward weight supplies the sign.
  """
  total: torch.Tensor | None = None
  for name, sensor in env.scene.sensors.items():
    # MuJoCo also registers each contact channel as a builtin sensor whose
    # name starts with the same prefix. Only the contact sensors count pairs.
    if not name.startswith("wheel_pair_") or not isinstance(sensor, ContactSensor):
      continue
    found = sensor.data.found
    assert found is not None
    hits = (found > 0).any(dim=-1).float()
    total = hits if total is None else total + hits
  if total is None:
    return torch.zeros(env.num_envs, device=env.device)
  return total


def track_ride_height(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  height_range: tuple[float, float],
  sensor_name: str | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward torso clearance against the commanded ride height.

  Command 0 is the lowest legal stance and command 1 is thighs vertical down.
  Rough terrain uses the mean downward scan under the torso. Flat ground uses
  the torso origin height, which sits on the ground plane.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  ride = command[:, 3].clamp(0.0, 1.0)
  low, high = height_range
  target = low + ride * (high - low)
  clearance = _torso_clearance(env, sensor_name, asset_cfg)
  return torch.exp(-torch.square(clearance - target) / std**2)


def _torso_clearance(
  env: ManagerBasedRlEnv,
  sensor_name: str | None,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return height of the torso origin above the ground under the body."""
  scanned = _scan_clearance(env, sensor_name)
  if scanned is not None:
    return scanned
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2]


def _scan_clearance(
  env: ManagerBasedRlEnv, sensor_name: str | None
) -> torch.Tensor | None:
  """Mean hit clearance, ignoring rays that miss."""
  if sensor_name is None or sensor_name not in env.scene.sensors:
    return None
  sensor = env.scene[sensor_name]
  distances = sensor.data.distances
  frame_z = sensor.data.frame_pos_w[:, 0, 2]
  clearance = frame_z.unsqueeze(-1) - sensor.data.hit_pos_w[..., 2]
  valid = distances >= 0
  clearance = torch.where(valid, clearance, torch.zeros_like(clearance))
  count = valid.sum(dim=-1).clamp(min=1).to(clearance.dtype)
  mean = clearance.sum(dim=-1) / count
  return torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
