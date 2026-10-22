"""Velocity env whose policy only toggles tumbling on the PD controller."""

from __future__ import annotations

import math

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.types import VecEnvStepReturn

from walter_rl.control.full_body_pd.controller import FullBodyPdController

_MIN_TUMBLE_ON_S = 0.5


class StateMachineEnv(ManagerBasedRlEnv):
  """Apply PD joint commands. The policy action is a 1-D tumble toggle.

  ``policy_action[:, 0] > 0`` latches tumbling. Efficient wheel drive runs
  when the toggle is off. Each edge is held for ``_MIN_TUMBLE_ON_S``: tumble
  stays on after a rising edge, and after a falling edge the stored output
  is forced negative so a new tumble cannot start. The stall-integral gate
  is not used.
  """

  def __init__(
    self,
    cfg: ManagerBasedRlEnvCfg,
    device: str,
    render_mode: str | None = None,
  ) -> None:
    super().__init__(cfg, device=device, render_mode=render_mode)
    self._teacher = FullBodyPdController(self)
    self.policy_action = torch.zeros(self.num_envs, 1, device=self.device)
    self.prev_policy_action = torch.zeros_like(self.policy_action)
    self._tumble_hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._tumble_off_hold = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._tumble_applied = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._min_tumble_steps = max(1, math.ceil(_MIN_TUMBLE_ON_S / self.step_dt))

  @property
  def teacher(self) -> FullBodyPdController:
    """PD controller that writes hips, knees, and wheels each step."""
    return self._teacher

  def step(self, action: torch.Tensor) -> VecEnvStepReturn:
    """Latch tumbling from the scalar action, then step the PD command."""
    toggle = action.reshape(self.num_envs, 1).to(device=self.device)
    self._reset_toggle_history()
    self.prev_policy_action = self.policy_action.clone()
    self.policy_action = toggle.detach().clone()
    self._teacher.set_tumble_command(self._tumble_command())
    return super().step(self._teacher.act())

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
    """Zero the stored toggle on environments that just reset."""
    fresh = self.episode_length_buf == 0
    self.prev_policy_action[fresh] = 0.0
    self.policy_action[fresh] = 0.0
    self._clear_tumble_latch(fresh)

  def _clear_tumble_latch(self, env_ids: torch.Tensor | slice) -> None:
    """Drop on-hold, off-hold, and the applied bit for ``env_ids``."""
    self._tumble_hold[env_ids] = 0
    self._tumble_off_hold[env_ids] = 0
    self._tumble_applied[env_ids] = False
