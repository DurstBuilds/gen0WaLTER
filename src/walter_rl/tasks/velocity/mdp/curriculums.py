"""WaLTER terrain curriculum helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")
_MIN_CMD_SPEED = 0.1


def terrain_levels_hold_max(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  """Promote/demote by commanded-direction progress; hold at the hardest row.

  Uses displacement along the velocity command, not isotropic distance from
  spawn, so circling/tumbling in place cannot farm promotions. Clearing the
  top row keeps the env on the hardest difficulty instead of re-rolling.
  """
  asset: Entity = env.scene[asset_cfg.name]
  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None
  command = env.command_manager.get_command(command_name)
  assert command is not None

  delta_xy = asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
  cmd_xy = command[env_ids, :2]
  cmd_speed = torch.linalg.norm(cmd_xy, dim=1)
  has_cmd = cmd_speed > _MIN_CMD_SPEED
  # Signed meters traveled along the commanded planar direction.
  progress = (delta_xy * cmd_xy).sum(dim=-1) / cmd_speed.clamp(min=1e-6)

  promote_dist = terrain_generator.size[0] / 4  # 2 m on 8 m tiles; size/2 was too hard early
  move_up = has_cmd & (progress > promote_dist)
  # Fail to cover a quarter of the commanded path length → demote (all columns).
  move_down = has_cmd & (progress < cmd_speed * env.max_episode_length_s * 0.25)
  move_down *= ~move_up

  if env.common_step_counter == 0:
    move_up = torch.zeros_like(move_up)
    move_down = torch.zeros_like(move_down)

  assert terrain.env_origins is not None and terrain.terrain_origins is not None
  levels = terrain.terrain_levels
  levels[env_ids] = levels[env_ids] + move_up.long() - move_down.long()
  hard_row = terrain.max_terrain_level - 1
  levels[env_ids] = torch.clamp(levels[env_ids], min=0, max=hard_row)
  terrain.env_origins[env_ids] = terrain.terrain_origins[
    levels[env_ids], terrain.terrain_types[env_ids]
  ]

  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels.float()),
    "max": torch.max(levels.float()),
  }
  sub_names = list(terrain_generator.sub_terrains.keys())
  num_cols = terrain.terrain_origins.shape[1]
  if num_cols == len(sub_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask].float())
  return result
