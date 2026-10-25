"""Velocity env whose policy toggles tumbling and sets the tank hip."""

from __future__ import annotations

import math

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

from walter_rl.control.full_body_pd.controller import FullBodyPdController

_MIN_TUMBLE_ON_S = 0.5
_POLICY_ACTION_DIM = 2
_HIP_COLUMN = 1
_MAX_TANK_HIP_DEG = 30.0
_MAX_TANK_HIP_RAD = math.radians(_MAX_TANK_HIP_DEG)
# Physics stays at 200 Hz. The PD retargets every 4 substeps (50 Hz).
_PD_DECIMATION = 4
_PD_CONTROL_DT = 0.02


class StateMachineEnv(ManagerBasedRlEnv):
  """Apply PD joint commands. The policy action is tumble plus tank hip.

  ``policy_action[:, 0] > 0`` latches tumbling. Efficient wheel drive runs
  when the toggle is off. Each edge is held for ``_MIN_TUMBLE_ON_S``: tumble
  stays on after a rising edge, and after a falling edge the stored output
  is forced negative so a new tumble cannot start. Column 1 is the tank hip
  on ``[-1, 1]``, mapped to 0–30 degrees. The latch updates once per policy
  step. The PD retargets five times inside that step. Tank Mode stays on
  and the tank knee command stays at zero. The stall-integral gate is not
  used.
  """

  def __init__(
    self,
    cfg: ManagerBasedRlEnvCfg,
    device: str,
    render_mode: str | None = None,
  ) -> None:
    super().__init__(cfg, device=device, render_mode=render_mode)
    if self.cfg.decimation % _PD_DECIMATION != 0:
      raise ValueError(
        f"Policy decimation {self.cfg.decimation} must be a multiple of "
        f"the PD decimation {_PD_DECIMATION}."
      )
    self._pd_ticks_per_policy_step = self.cfg.decimation // _PD_DECIMATION
    self._teacher = FullBodyPdController(self, control_dt=_PD_CONTROL_DT)
    self.policy_action = torch.zeros(
      self.num_envs, _POLICY_ACTION_DIM, device=self.device
    )
    self.prev_policy_action = torch.zeros_like(self.policy_action)
    self._tumble_hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._tumble_off_hold = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._tumble_applied = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._min_tumble_steps = max(1, math.ceil(_MIN_TUMBLE_ON_S / self.step_dt))
    self._apply_tank_pose()

  @property
  def teacher(self) -> FullBodyPdController:
    """PD controller that writes hips, knees, and wheels each step."""
    return self._teacher

  def _apply_action_and_simulate(self, action: torch.Tensor) -> None:
    """Latch the policy command once, then run the PD at its own rate."""
    tumble = self._latch_policy_command(action)
    self._teacher.skip_fresh_clear = False
    for _ in range(self._pd_ticks_per_policy_step):
      self._teacher.set_tumble_command(tumble)
      self.action_manager.process_action(self._teacher.act())
      for _ in range(_PD_DECIMATION):
        self._physics_substep()
      self._teacher.skip_fresh_clear = True

  def _latch_policy_command(self, action: torch.Tensor) -> torch.Tensor:
    """Store the 2-D policy action and return the latched tumble mask."""
    command = action.reshape(self.num_envs, _POLICY_ACTION_DIM)
    command = command.to(device=self.device)
    self._reset_toggle_history()
    self.prev_policy_action = self.policy_action.clone()
    self.policy_action = command.detach().clone()
    self._write_hip_offset()
    self._apply_tank_pose()
    return self._tumble_command()

  def _write_hip_offset(self) -> None:
    """Map the hip sample from ``[-1, 1]`` onto ``[0, 1]``.

    The Beta policy uses the whole interval. Zero is 15 degrees and the
    ends are 0 and 30 degrees.
    """
    hip = (self.policy_action[:, _HIP_COLUMN] + 1.0) * 0.5
    self.policy_action[:, _HIP_COLUMN] = hip.clamp(0.0, 1.0)

  def _apply_tank_pose(self) -> None:
    """Hold Tank Mode on, freeze the knee command, and write the hip offset.

    Column 1 is 0 at 0 degrees and 1 at 30 degrees. Rewriting the pose here
    runs before the PD read, so a later play-slider update cannot win.
    """
    twist = self.command_manager.get_term("twist")
    twist.tank_mode = True
    twist.knee_angle.zero_()
    twist.hip_angle[:] = self.policy_action[:, _HIP_COLUMN] * _MAX_TANK_HIP_RAD

  def _tumble_command(self) -> torch.Tensor:
    """Hold each tumble edge for the minimum dwell."""
    self._lock_off_output()
    applied = self._apply_on_dwell()
    self._begin_off_dwell(applied)
    self._tumble_applied = applied
    return applied

  def _lock_off_output(self) -> None:
    """Force the stored toggle negative while the off dwell is running."""
    locked = self._tumble_off_hold > 0
    signed = -self.policy_action[:, 0].abs()
    self.policy_action[:, 0] = torch.where(locked, signed, self.policy_action[:, 0])

  def _apply_on_dwell(self) -> torch.Tensor:
    """Keep tumble on for the minimum on-time after each rising edge."""
    requested = self.policy_action[:, 0] > 0.0
    rising = requested & ~self._tumble_applied
    self._tumble_hold = torch.where(
      rising,
      torch.full_like(self._tumble_hold, self._min_tumble_steps),
      self._tumble_hold,
    )
    applied = requested | (self._tumble_hold > 0)
    self._tumble_hold = (self._tumble_hold - 1).clamp(min=0)
    return applied

  def _begin_off_dwell(self, applied: torch.Tensor) -> None:
    """Start the off dwell on the step tumble releases."""
    falling = self._tumble_applied & ~applied
    self._tumble_off_hold = torch.where(
      falling,
      torch.full_like(self._tumble_off_hold, self._min_tumble_steps),
      self._tumble_off_hold,
    )
    self._tumble_off_hold = (self._tumble_off_hold - 1).clamp(min=0)

  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    """Clear the tumble latch before the post-reset observation is built."""
    super()._reset_idx(env_ids)
    self._clear_tumble_latch(slice(None) if env_ids is None else env_ids)

  def _reset_toggle_history(self) -> None:
    """Zero the stored action on environments that just reset."""
    fresh = self.episode_length_buf == 0
    self.prev_policy_action[fresh] = 0.0
    self.policy_action[fresh] = 0.0
    self._clear_tumble_latch(fresh)

  def _clear_tumble_latch(self, env_ids: torch.Tensor | slice) -> None:
    """Drop on-hold, off-hold, and the applied bit for ``env_ids``."""
    self._tumble_hold[env_ids] = 0
    self._tumble_off_hold[env_ids] = 0
    self._tumble_applied[env_ids] = False
