"""Drive and climb curriculum for the rough-test terrain grid.

Drive rows are flat enough to roll. Climb rows are the tall stair and cube
faces. Promotion follows the mode of the spawn, and each climb spawn is an
approach at the feature instead of the cell origin.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from walter_rl.assets.walter_fs.walter_constants import NOMINAL_ROOT_HEIGHT
from walter_rl.tasks.velocity.mdp.rewards import forward_match_step

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg

_KNEE_PATTERN = r".*_knee"
_STAIRS = "pyramid_stairs"
_STAIRS_INV = "pyramid_stairs_inv"
_CUBES = "random_grid"
_HEIGHTFIELDS = ("hf_pyramid_slope", "hf_pyramid_slope_inv", "wave_terrain")
_CLIMB_NAMES = (_STAIRS, _STAIRS_INV, _CUBES)

# Geometry of the 8 m rough-test tiles. See the pyramid and cube generators.
_TILE_HALF_M = 4.0
_STAIR_RISER_R_M = 3.0
_PIT_WALL_R_M = 1.5
# Root origin to the front face. The back torso is the root; the long
# segment hangs forward of the waist.
_TORSO_HALF_X_M = 0.5554
_APPROACH_M = 0.75
_YAW_NOISE_RAD = 0.15
_LATERAL_NOISE_M = 0.15
_DRIVE_NOISE_M = 0.2
_ROOT_CLEARANCE_M = 0.02
_HEIGHTFIELD_CLEARANCE_M = 0.08
_ROOT_Z_M = NOMINAL_ROOT_HEIGHT + _ROOT_CLEARANCE_M

_CLIMB_ROW = 2
_DRIVE_TOP_ROW = 1
_ROLLABLE_STEP_M = 0.06
# The stall-integral teacher cruises about 0.30 m/s under a 1 m/s command.
_DRIVE_MARGIN_MPS = 0.40
_DRIVE_DISTANCE_M = 4.0
_MIN_CMD_MPS = 0.1
_QUIET_KNEE_RADPS = 0.75
_TUMBLE_KNEE_RADPS = 1.5
_PREDICTIVE_ALONG_M = 0.55
_CLIMB_CLEAR_M = 0.05
_AWAY_PATH_M = 2.0
_AWAY_ALONG_M = 0.3
_PATCH_MARGIN_M = 1.0
# Share of drive graduates that are not already on a climb column.
_CLIMB_STAIR_FRACTION = 0.50
_CLIMB_CUBE_FRACTION = 0.35

_PROMOTE = 1
_HOLD = 0
_DEMOTE = -1


class SkillTerrainCurriculum(ManagerTermBase):
  """Advance drive and climb rows from how the episode was driven.

  Drive promotes on 4 m of on-patch rolling inside 0.40 m/s of the command
  with quiet knees. A tumble on ground below the rollable step demotes.
  Climb promotes when a forward tumble starts while the step is still ahead,
  and the body then gains height. Cruising away from the faced riser demotes
  back to drive row 0. Clearing the top climb row returns to drive row 0.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._command_name = str(cfg.params["command_name"])
    robot: Entity = env.scene["robot"]
    knee_ids, _names = robot.find_joints(_KNEE_PATTERN)
    self._knee_ids = torch.tensor(knee_ids, dtype=torch.long, device=env.device)
    n = env.num_envs
    self.approach_yaw = torch.zeros(n, device=env.device)
    self.spawn_xy = torch.zeros(n, 2, device=env.device)
    self.drive_travel = torch.zeros(n, device=env.device)
    self.path = torch.zeros(n, device=env.device)
    self.wasted_tumble = torch.zeros(n, dtype=torch.bool, device=env.device)
    self.predictive = torch.zeros(n, dtype=torch.bool, device=env.device)
    self.height_gain = torch.zeros(n, device=env.device)
    self._z_latch = torch.zeros(n, device=env.device)
    self.last_verdict = torch.zeros(n, dtype=torch.long, device=env.device)
    self.last_predictive = torch.zeros(n, dtype=torch.bool, device=env.device)
    self.last_height = torch.zeros(n, device=env.device)
    self.last_travel = torch.zeros(n, device=env.device)
    self.last_wasted = torch.zeros(n, dtype=torch.bool, device=env.device)
    terrain = env.scene.terrain
    assert terrain is not None
    # The column assigned at startup. Climb graduates return here.
    self._home_column = terrain.terrain_types.clone()

  def accumulate(self, dt: float) -> None:
    """Update drive distance and climb flags from this step."""
    robot: Entity = self._env.scene["robot"]
    command = self._env.command_manager.get_command(self._command_name)
    assert command is not None
    speed = torch.linalg.norm(robot.data.root_link_lin_vel_w[:, :2], dim=-1)
    self.path += speed * dt
    step = _forward_step(self._env)
    self._accumulate_drive(robot, command, speed, dt, step)
    self._accumulate_climb(robot, step)

  def spawn(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    """Place ``env_ids`` for the row and column chosen at this reset."""
    robot: Entity = env.scene["robot"]
    terrain = env.scene.terrain
    assert terrain is not None and terrain.terrain_origins is not None
    levels = terrain.terrain_levels[env_ids]
    types = terrain.terrain_types[env_ids]
    origins = terrain.terrain_origins[levels, types]
    pos, yaw = _spawn_pose(terrain, origins, types, levels)
    self.approach_yaw[env_ids] = yaw
    self.spawn_xy[env_ids] = pos[:, :2]
    state = torch.zeros(env_ids.shape[0], 13, device=env.device)
    state[:, 0:3] = pos
    state[:, 3:7] = quat_from_euler_xyz(
      torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
    )
    robot.write_root_state_to_sim(state, env_ids=env_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Clear episode accumulators after the verdict has been applied."""
    ids = slice(None) if env_ids is None else env_ids
    self.drive_travel[ids] = 0.0
    self.path[ids] = 0.0
    self.wasted_tumble[ids] = False
    self.predictive[ids] = False
    self.height_gain[ids] = 0.0
    self._z_latch[ids] = 0.0

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, **params):
    del params
    if env.common_step_counter != 0:
      verdict = _verdict(self, env_ids)
      self.last_verdict[env_ids] = verdict
      self.last_predictive[env_ids] = self.predictive[env_ids]
      self.last_height[env_ids] = self.height_gain[env_ids]
      self.last_travel[env_ids] = self.drive_travel[env_ids]
      self.last_wasted[env_ids] = self.wasted_tumble[env_ids]
      _apply_verdict(self, env, env_ids, verdict)
    stats = _stats(env)
    stats["drive_travel"] = torch.mean(self.drive_travel[env_ids])
    return stats

  def _accumulate_drive(self, robot, command, speed, dt: float, step) -> None:
    """Count rolling meters, and flag a tumble on ground that should be rolled."""
    actual = robot.data.root_link_lin_vel_b[:, :2]
    error = torch.linalg.norm(actual - command[:, :2], dim=-1)
    knee = robot.data.joint_vel[:, self._knee_ids]
    quiet = knee.abs().mean(dim=-1) <= _QUIET_KNEE_RADPS
    tumbling = knee.abs().mean(dim=-1) >= _TUMBLE_KNEE_RADPS
    moving = command[:, 0].abs() > _MIN_CMD_MPS
    on_patch = _on_patch(self._env, robot)
    count = on_patch & moving & quiet & (error <= _DRIVE_MARGIN_MPS)
    drive = ~_is_climb(self._env)
    self.drive_travel += torch.where(drive & count, speed * dt, 0.0)
    flat = step < _ROLLABLE_STEP_M
    self.wasted_tumble |= drive & on_patch & tumbling & flat

  def _accumulate_climb(self, robot, step) -> None:
    """Latch a tumble that starts before the riser, then track the rise."""
    climb = _is_climb(self._env)
    knee = robot.data.joint_vel[:, self._knee_ids]
    tumbling = torch.relu(knee).mean(dim=-1) >= _TUMBLE_KNEE_RADPS
    along = _along_heading(self, robot)
    ahead = step >= _ROLLABLE_STEP_M
    early = climb & tumbling & ahead & (along < _PREDICTIVE_ALONG_M) & ~self.predictive
    self._z_latch = torch.where(early, robot.data.root_link_pos_w[:, 2], self._z_latch)
    self.predictive |= early
    risen = robot.data.root_link_pos_w[:, 2] - self._z_latch
    gain = torch.where(self.predictive, risen, torch.zeros_like(risen))
    self.height_gain = torch.maximum(self.height_gain, gain)


def _forward_step(env) -> torch.Tensor:
  """Largest height jump ahead of the robot."""
  return forward_match_step(env)


def accumulate_skill_progress(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  term_name: str = "terrain_levels",
) -> None:
  """Step the rough-test skill curriculum."""
  del env_ids
  env.curriculum_manager.get_term_cfg(term_name).func.accumulate(env.step_dt)


def spawn_skill_pose(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  term_name: str = "terrain_levels",
) -> None:
  """Reset-event wrapper around ``SkillTerrainCurriculum.spawn``."""
  env.curriculum_manager.get_term_cfg(term_name).func.spawn(env, env_ids)


def _spawn_pose(terrain, origins, types, levels):
  """World position and yaw for each resetting environment."""
  device = origins.device
  n = origins.shape[0]
  climb_up = _named(terrain, types, (_STAIRS, _CUBES)) & (levels >= _CLIMB_ROW)
  climb_out = _named(terrain, types, (_STAIRS_INV,)) & (levels >= _CLIMB_ROW)
  drive = ~(climb_up | climb_out)
  phi = _cardinal_yaw(n, device)
  yaw = torch.where(climb_out, phi, phi + math.pi)
  yaw = torch.where(drive, _uniform_yaw(n, device), yaw)
  radius = _approach_radius(terrain, types, climb_out)
  xy = _approach_xy(origins[:, :2], phi, radius)
  drive_xy = origins[:, :2] + _centered_uniform(n, 2, _DRIVE_NOISE_M, device)
  xy = torch.where(drive.unsqueeze(-1), drive_xy, xy)
  z = _spawn_z(terrain, origins, types, climb_up, climb_out)
  pos = torch.cat([xy, z.unsqueeze(-1)], dim=-1)
  return pos, _wrap_pi(yaw)


def _approach_radius(terrain, types, climb_out: torch.Tensor) -> torch.Tensor:
  """Distance from the cell center to the approach point."""
  stair = _STAIR_RISER_R_M + _TORSO_HALF_X_M + _APPROACH_M
  cube = _cube_face_radius(terrain) + _TORSO_HALF_X_M + _APPROACH_M
  pit = _PIT_WALL_R_M - _TORSO_HALF_X_M - _APPROACH_M
  radius = torch.full(types.shape, stair, device=types.device)
  radius = torch.where(_named(terrain, types, (_CUBES,)), cube, radius)
  return torch.where(climb_out, pit, radius)


def _cube_face_radius(terrain) -> float:
  """Distance from the cell center to the outer face of the first cube.

  Matches ``BoxRandomGridTerrainCfg``, which rebuilds the apron from however
  many cells fit inside the requested border.
  """
  sub = terrain.cfg.terrain_generator.sub_terrains[_CUBES]
  size = float(sub.size[0])
  boxes = int((size - 2.0 * sub.border_width) / sub.grid_width)
  leftover = size - boxes * sub.grid_width
  return 0.5 * size - 0.5 * leftover


def _approach_xy(center, phi, radius):
  """Point on the approach ring, plus a small sideways offset."""
  out = torch.stack((torch.cos(phi), torch.sin(phi)), dim=-1)
  lat = torch.stack((-out[:, 1], out[:, 0]), dim=-1)
  side = _centered_uniform(phi.shape[0], 1, _LATERAL_NOISE_M, phi.device)
  return center + radius.unsqueeze(-1) * out + side * lat


def _spawn_z(terrain, origins, types, climb_up, climb_out):
  """Root height above the surface the spawn is standing on."""
  ground = torch.zeros(types.shape, device=types.device)
  floor = origins[:, 2]
  extra = torch.where(
    _named(terrain, types, _HEIGHTFIELDS), _HEIGHTFIELD_CLEARANCE_M, 0.0
  )
  z = floor + _ROOT_Z_M + extra
  z = torch.where(climb_up, ground + _ROOT_Z_M, z)
  return torch.where(climb_out, floor + _ROOT_Z_M, z)


def _verdict(term: SkillTerrainCurriculum, env_ids: torch.Tensor) -> torch.Tensor:
  """1 promote, 0 hold, -1 demote for each resetting environment."""
  climb = _is_climb(term._env)[env_ids]
  drive = ~climb
  travel = term.drive_travel[env_ids]
  wasted = term.wasted_tumble[env_ids]
  cleared = term.predictive[env_ids] & (term.height_gain[env_ids] >= _CLIMB_CLEAR_M)
  along = _along_heading(term, term._env.scene["robot"])[env_ids]
  away = (along < _AWAY_ALONG_M) & (term.path[env_ids] > _AWAY_PATH_M)
  promote = (drive & (travel >= _DRIVE_DISTANCE_M) & ~wasted) | (climb & cleared)
  demote = (drive & wasted) | (climb & away & ~cleared)
  code = torch.zeros(env_ids.shape[0], dtype=torch.long, device=env_ids.device)
  code = torch.where(promote, _PROMOTE, code)
  return torch.where(demote, _DEMOTE, code)


def _apply_verdict(
  term: SkillTerrainCurriculum,
  env,
  env_ids: torch.Tensor,
  verdict: torch.Tensor,
) -> None:
  """Move rows. A finished or demoted climb returns to its home column."""
  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_origins is not None
  levels = terrain.terrain_levels
  types = terrain.terrain_types
  row = levels[env_ids].clone()
  col = types[env_ids].clone()
  climb = _is_climb(env)[env_ids]
  row, col = _next_drive(terrain, row, col, verdict, ~climb)
  row, col = _next_climb(row, col, verdict, climb, term._home_column[env_ids])
  levels[env_ids] = row
  types[env_ids] = col
  terrain.env_origins[env_ids] = terrain.terrain_origins[row, col]


def _next_drive(terrain, row, col, verdict, drive):
  """Promote within rows 0–1, then enter climb row 2."""
  promote = drive & (verdict == _PROMOTE)
  demote = drive & (verdict == _DEMOTE)
  stay_drive = promote & (row < _DRIVE_TOP_ROW)
  enter = promote & (row >= _DRIVE_TOP_ROW)
  row = torch.where(stay_drive, row + 1, row)
  row = torch.where(demote, (row - 1).clamp(min=0), row)
  row = torch.where(enter, torch.full_like(row, _CLIMB_ROW), row)
  needs_column = enter & ~_named(terrain, col, _CLIMB_NAMES)
  return row, torch.where(needs_column, _climb_column(terrain, col), col)


def _next_climb(row, col, verdict, climb, home):
  """Step up climb rows. Finishing or failing returns to the home column."""
  promote = climb & (verdict == _PROMOTE)
  demote = climb & (verdict == _DEMOTE)
  # Climb rows are 2 and 3. Clearing 3 returns to the drive pool.
  finished = promote & (row >= _CLIMB_ROW + 1)
  step = promote & ~finished
  returning = finished | demote
  row = torch.where(step, row + 1, row)
  row = torch.where(returning, torch.zeros_like(row), row)
  col = torch.where(returning, home, col)
  return row, col


def _climb_column(terrain, col: torch.Tensor) -> torch.Tensor:
  """Assign stairs, cubes, or a few inverted stairs."""
  draw = torch.rand(col.shape, device=col.device)
  stairs = _index(terrain, _STAIRS)
  cubes = _index(terrain, _CUBES)
  pits = _index(terrain, _STAIRS_INV)
  chosen = torch.full_like(col, stairs)
  chosen = torch.where(draw > _CLIMB_STAIR_FRACTION, cubes, chosen)
  return torch.where(draw > _CLIMB_STAIR_FRACTION + _CLIMB_CUBE_FRACTION, pits, chosen)


def _is_climb(env) -> torch.Tensor:
  """True on stair or cube columns at row 2 or higher."""
  terrain = env.scene.terrain
  assert terrain is not None
  named = _named(terrain, terrain.terrain_types, _CLIMB_NAMES)
  return named & (terrain.terrain_levels >= _CLIMB_ROW)


def _named(terrain, types: torch.Tensor, names: tuple[str, ...]) -> torch.Tensor:
  """True where ``types`` is one of ``names``."""
  mask = torch.zeros(types.shape, dtype=torch.bool, device=types.device)
  have = list(terrain.cfg.terrain_generator.sub_terrains.keys())
  for name in names:
    if name in have:
      mask |= types == have.index(name)
  return mask


def _index(terrain, name: str) -> int:
  names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
  return names.index(name)


def _on_patch(env, robot) -> torch.Tensor:
  """True while the root is inside the tile, plus the approach margin."""
  terrain = env.scene.terrain
  center = terrain.env_origins[:, :2]
  delta = (robot.data.root_link_pos_w[:, :2] - center).abs()
  limit = _TILE_HALF_M + _PATCH_MARGIN_M
  return (delta[:, 0] <= limit) & (delta[:, 1] <= limit)


def _along_heading(term: SkillTerrainCurriculum, robot) -> torch.Tensor:
  """Meters from the spawn point along the approach yaw."""
  delta = robot.data.root_link_pos_w[:, :2] - term.spawn_xy
  yaw = term.approach_yaw
  return torch.cos(yaw) * delta[:, 0] + torch.sin(yaw) * delta[:, 1]


def _cardinal_yaw(n: int, device) -> torch.Tensor:
  base = torch.randint(0, 4, (n,), device=device).float() * (0.5 * math.pi)
  noise = _centered_uniform(n, 1, _YAW_NOISE_RAD, device).squeeze(-1)
  return base + noise


def _uniform_yaw(n: int, device) -> torch.Tensor:
  return (torch.rand(n, device=device) - 0.5) * (2.0 * math.pi)


def _centered_uniform(n: int, width: int, half: float, device) -> torch.Tensor:
  return (torch.rand(n, width, device=device) - 0.5) * (2.0 * half)


def _wrap_pi(yaw: torch.Tensor) -> torch.Tensor:
  return torch.atan2(torch.sin(yaw), torch.cos(yaw))


def _stats(env) -> dict[str, torch.Tensor]:
  """Mean terrain row, overall and per column."""
  terrain = env.scene.terrain
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
    "climb": _is_climb(env).float().mean(),
  }
  generator = terrain.cfg.terrain_generator
  if generator is None or terrain.terrain_origins is None:
    return result
  names = list(generator.sub_terrains.keys())
  if terrain.terrain_origins.shape[1] != len(names):
    return result
  for index, name in enumerate(names):
    mask = terrain.terrain_types == index
    if mask.any():
      result[name] = torch.mean(levels[mask])
  return result
