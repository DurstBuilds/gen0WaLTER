"""Fixed-column traversal curriculum for the rough training grid.

Each environment keeps the column it was spawned on. A row promotes when the
root reaches the exit radius, and demotes on a fall, a hard torso hit, or a
long episode that never left the inner half of that radius. Clearing the top
row re-rolls a random row so the easy tiles stay in the mix.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from walter_rl.assets.walter_fs.walter_constants import NOMINAL_ROOT_HEIGHT

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg

_STAIR_COLUMNS = ("pyramid_stairs", "pyramid_stairs_inv")
_OBSTACLE_COLUMNS = ("pyramid_stairs", "pyramid_stairs_inv", "random_grid")
_HEIGHTFIELD_COLUMNS = ("hf_pyramid_slope", "hf_pyramid_slope_inv", "wave_terrain")

_SPAWN_HALF_M = 0.3
_STAIR_YAW_NOISE_RAD = 0.35
_OBSTACLE_EXIT_M = 3.35
_OPEN_EXIT_M = 3.5
_STUCK_RADIUS_FRACTION = 0.5
_STUCK_EPISODE_FRACTION = 0.5
_ROOT_CLEARANCE_M = 0.02
_HEIGHTFIELD_CLEARANCE_M = 0.08
_FALL_TERM = "fell_over"
_TORSO_TERM = "torso_contact"


class TraversalTerrainCurriculum(ManagerTermBase):
  """Promote and demote terrain rows from whether the tile was crossed."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    del cfg
    n = env.num_envs
    self.farthest = torch.zeros(n, device=env.device)
    self.steps = torch.zeros(n, dtype=torch.long, device=env.device)
    self._reached = torch.zeros(n, dtype=torch.bool, device=env.device)
    self._demoted = torch.zeros(n, dtype=torch.bool, device=env.device)

  def accumulate(self) -> None:
    """Keep the farthest Chebyshev distance from the tile center.

    ``steps`` counts policy steps since the last verdict. The runner may
    preload ``episode_length_buf``, so that buffer is not the time lived.
    """
    self.steps += 1
    robot: Entity = self._env.scene["robot"]
    terrain = self._env.scene.terrain
    assert terrain is not None and terrain.env_origins is not None
    delta = (robot.data.root_link_pos_w[:, :2] - terrain.env_origins[:, :2]).abs()
    reach = torch.maximum(delta[:, 0], delta[:, 1])
    self.farthest = torch.maximum(self.farthest, reach)

  def spawn(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    """Place ``env_ids`` at the tile center for the row chosen at this reset."""
    robot: Entity = env.scene["robot"]
    terrain = env.scene.terrain
    assert terrain is not None and terrain.terrain_origins is not None
    levels = terrain.terrain_levels[env_ids]
    types = terrain.terrain_types[env_ids]
    origins = terrain.terrain_origins[levels, types]
    pos, yaw = _spawn_pose(terrain, origins, types)
    state = torch.zeros(env_ids.shape[0], 13, device=env.device)
    state[:, 0:3] = pos
    state[:, 3:7] = quat_from_euler_xyz(
      torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
    )
    robot.write_root_state_to_sim(state, env_ids=env_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Clear the distance tracker after the verdict has been applied."""
    ids = slice(None) if env_ids is None else env_ids
    self.farthest[ids] = 0.0
    self.steps[ids] = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    **params,
  ) -> dict[str, torch.Tensor]:
    """Update rows for finished episodes and return per-column stats."""
    del params
    judged = env_ids[:0]
    if env.common_step_counter != 0 and env_ids.numel() > 0:
      _apply_rows(self, env, env_ids)
      judged = env_ids
    return _stats(self, env, judged)


def accumulate_traversal_progress(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  term_name: str = "terrain_levels",
) -> None:
  """Step event that records distance from the tile center."""
  del env_ids
  env.curriculum_manager.get_term_cfg(term_name).func.accumulate()


def spawn_traversal_pose(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  term_name: str = "terrain_levels",
) -> None:
  """Reset event that places the robot at the center of its tile."""
  env.curriculum_manager.get_term_cfg(term_name).func.spawn(env, env_ids)


def _apply_rows(
  term: TraversalTerrainCurriculum,
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
) -> None:
  """Promote a crossing, demote a failure, and leave the column alone."""
  terrain = env.scene.terrain
  assert terrain is not None
  exit_radius = _exit_radius(terrain)[env_ids]
  reached = term.farthest[env_ids] >= exit_radius
  failed = _failed(term, env, env_ids, exit_radius) & ~reached
  term._reached[env_ids] = reached
  term._demoted[env_ids] = failed
  terrain.update_env_origins(env_ids, reached.long(), failed.long())


def _failed(
  term: TraversalTerrainCurriculum,
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  exit_radius: torch.Tensor,
) -> torch.Tensor:
  """True on a fall, a hard torso hit, or a long episode stuck near the center."""
  fell = _termination(env, _FALL_TERM, env_ids)
  hit = _termination(env, _TORSO_TERM, env_ids)
  long_enough = term.steps[env_ids] >= (
    _STUCK_EPISODE_FRACTION * env.max_episode_length
  )
  stuck = long_enough & (term.farthest[env_ids] < _STUCK_RADIUS_FRACTION * exit_radius)
  return fell | hit | stuck


def _termination(
  env: ManagerBasedRlEnv,
  name: str,
  env_ids: torch.Tensor,
) -> torch.Tensor:
  """Done mask for one termination, or false when that term is absent."""
  if name not in env.termination_manager.active_terms:
    return torch.zeros(env_ids.shape[0], dtype=torch.bool, device=env.device)
  return env.termination_manager.get_term(name)[env_ids]


def _exit_radius(terrain) -> torch.Tensor:
  """Meters from the tile center that count as leaving the obstacle."""
  types = terrain.terrain_types
  radius = torch.full(types.shape, _OPEN_EXIT_M, device=types.device)
  return torch.where(_named(terrain, types, _OBSTACLE_COLUMNS), _OBSTACLE_EXIT_M, radius)


def _spawn_pose(terrain, origins: torch.Tensor, types: torch.Tensor):
  """World position and yaw. Stairs face a cardinal heading."""
  device = origins.device
  n = origins.shape[0]
  noise = (torch.rand(n, 2, device=device) - 0.5) * (2.0 * _SPAWN_HALF_M)
  xy = origins[:, :2] + noise
  yaw = _spawn_yaw(terrain, types)
  z = origins[:, 2] + NOMINAL_ROOT_HEIGHT + _clearance(terrain, types)
  return torch.cat([xy, z.unsqueeze(-1)], dim=-1), yaw


def _spawn_yaw(terrain, types: torch.Tensor) -> torch.Tensor:
  """Cardinal yaw on stairs, uniform yaw on every other column."""
  n = types.shape[0]
  device = types.device
  uniform = (torch.rand(n, device=device) - 0.5) * (2.0 * math.pi)
  cardinal = torch.randint(0, 4, (n,), device=device).float() * (0.5 * math.pi)
  jitter = (torch.rand(n, device=device) - 0.5) * (2.0 * _STAIR_YAW_NOISE_RAD)
  stairs = _named(terrain, types, _STAIR_COLUMNS)
  return torch.where(stairs, cardinal + jitter, uniform)


def _clearance(terrain, types: torch.Tensor) -> torch.Tensor:
  """Extra root height so a heightfield spawn is not born in the surface."""
  clearance = torch.full(types.shape, _ROOT_CLEARANCE_M, device=types.device)
  on_field = _named(terrain, types, _HEIGHTFIELD_COLUMNS)
  extra = _ROOT_CLEARANCE_M + _HEIGHTFIELD_CLEARANCE_M
  return torch.where(on_field, extra, clearance)


def _stats(
  term: TraversalTerrainCurriculum,
  env: ManagerBasedRlEnv,
  judged: torch.Tensor,
) -> dict[str, torch.Tensor]:
  """Mean row, traverse rate, demote rate, and env count for each column."""
  terrain = env.scene.terrain
  assert terrain is not None
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }
  generator = terrain.cfg.terrain_generator
  if generator is None or terrain.terrain_origins is None:
    return result
  names = list(generator.sub_terrains.keys())
  if terrain.terrain_origins.shape[1] != len(names):
    return result
  for index, name in enumerate(names):
    _column_stats(term, terrain, result, index, name, judged)
  return result


def _column_stats(term, terrain, result, index: int, name: str, judged: torch.Tensor) -> None:
  """Add one column's level, population, and this-step verdict rates."""
  mask = terrain.terrain_types == index
  if not bool(mask.any()):
    return
  result[name] = torch.mean(terrain.terrain_levels[mask].float())
  result[f"{name}_count"] = mask.sum().float()
  if judged.numel() == 0:
    return
  finished = torch.zeros_like(mask)
  finished[judged] = mask[judged]
  if not bool(finished.any()):
    return
  result[f"{name}_traverse"] = term._reached[finished].float().mean()
  result[f"{name}_demote"] = term._demoted[finished].float().mean()


def _named(terrain, types: torch.Tensor, names: tuple[str, ...]) -> torch.Tensor:
  """True where ``types`` is one of ``names``."""
  mask = torch.zeros(types.shape, dtype=torch.bool, device=types.device)
  have = list(terrain.cfg.terrain_generator.sub_terrains.keys())
  for name in names:
    if name in have:
      mask |= types == have.index(name)
  return mask
