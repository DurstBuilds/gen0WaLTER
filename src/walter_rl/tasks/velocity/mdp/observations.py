"""WaLTER-specific observation terms."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.observations import builtin_sensor, height_scan, joint_vel_rel
from mjlab.managers.scene_entity_config import SceneEntityCfg

from walter_rl.assets.motor_models import get_motor

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_HIP_RATED_TORQUE_NM = get_motor("ak70_10_kv100").rated_torque_nm
_KNEE_RATED_TORQUE_NM = get_motor("ak80_64_kv80").rated_torque_nm
_WHEEL_RATED_TORQUE_NM = get_motor("ak10_9_v2_kv60").rated_torque_nm
_TORQUE_SCALE_ATTR = "_motor_torque_obs_scale"


def finite_projected_gravity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Projected gravity with non-finite samples replaced by zero."""
  from mjlab.envs.mdp.observations import projected_gravity

  gravity = projected_gravity(env, asset_cfg=asset_cfg)
  return torch.nan_to_num(gravity, nan=0.0, posinf=0.0, neginf=0.0)


def finite_joint_pos_rel(
  env: ManagerBasedRlEnv,
  biased: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Relative joint position with non-finite samples replaced by zero."""
  from mjlab.envs.mdp.observations import joint_pos_rel

  position = joint_pos_rel(env, biased=biased, asset_cfg=asset_cfg)
  return torch.nan_to_num(position, nan=0.0, posinf=0.0, neginf=0.0)


def finite_builtin_sensor(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """IMU reading with non-finite samples replaced by zero."""
  return torch.nan_to_num(builtin_sensor(env, sensor_name), nan=0.0, posinf=0.0, neginf=0.0)


def finite_joint_vel_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Joint velocity with non-finite samples clamped to a bounded range."""
  velocity = joint_vel_rel(env, asset_cfg=asset_cfg)
  return torch.nan_to_num(velocity, nan=0.0, posinf=40.0, neginf=-40.0)


def finite_height_scan(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Terrain height scan with missed or broken rays replaced by zero."""
  heights = height_scan(env, sensor_name=sensor_name)
  return torch.nan_to_num(heights, nan=0.0, posinf=1.0, neginf=0.0)


def non_finite_root(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate environments whose root state has left the finite range.

  A blown-up contact solve otherwise poisons the whole rollout with NaNs.
  Resetting those environments keeps training alive.
  """
  asset: Entity = env.scene[asset_cfg.name]
  pose_ok = torch.isfinite(asset.data.root_link_pos_w).all(dim=-1)
  vel_ok = torch.isfinite(asset.data.root_link_lin_vel_w).all(dim=-1)
  return ~(pose_ok & vel_ok)


def joint_pos_sincos(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Sine and cosine of selected joint positions.

  Continuous knees can wind through full turns. Encoding the angle this way
  keeps the observation bounded while still telling the policy which way the
  leg is pointing.
  """
  asset: Entity = env.scene[asset_cfg.name]
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return torch.nan_to_num(
    torch.cat((torch.sin(joint_pos), torch.cos(joint_pos)), dim=-1),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
  )


def motor_torques(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Actuator forces scaled by each motor's rated torque.

  ``actuator_force`` is one scalar per hip, knee, and wheel. Dividing by
  the CubeMars rated torque keeps the observation near ``[-1, 1]``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  force = asset.data.actuator_force
  scale = _rated_torque_scale(env, asset, force)
  return torch.nan_to_num(force / scale, nan=0.0, posinf=1.0, neginf=-1.0)


def _rated_torque_scale(
  env: ManagerBasedRlEnv,
  asset: Entity,
  force: torch.Tensor,
) -> torch.Tensor:
  """Rated torque for each actuator, cached on the environment."""
  scale = getattr(env, _TORQUE_SCALE_ATTR, None)
  if isinstance(scale, torch.Tensor) and scale.shape[-1] == force.shape[-1]:
    return scale
  values = [_rated_torque_for_actuator(name) for name in asset.actuator_names]
  scale = torch.tensor(values, device=force.device, dtype=force.dtype)
  setattr(env, _TORQUE_SCALE_ATTR, scale)
  return scale


def _rated_torque_for_actuator(name: str) -> float:
  """Datasheet rated torque for a hip, knee, or wheel actuator name."""
  lowered = name.lower()
  if "wheel" in lowered:
    return _WHEEL_RATED_TORQUE_NM
  if "knee" in lowered or "shin" in lowered:
    return _KNEE_RATED_TORQUE_NM
  return _HIP_RATED_TORQUE_NM


def tumble_applied(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Tumble mode applied to the PD this step, including the minimum on-time.

  Shape is ``(num_envs, 1)``. A history on this term is the recent tumble
  decisions. Reset environments report 0.
  """
  applied = getattr(env, "_tumble_applied", None)
  if not isinstance(applied, torch.Tensor):
    return torch.zeros(env.num_envs, 1, device=env.device)
  return applied.to(dtype=torch.float32).reshape(env.num_envs, 1)
