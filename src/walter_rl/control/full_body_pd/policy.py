"""Viewer adapter for the full-body PD controller."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv

from walter_rl.control.full_body_pd.controller import FullBodyPdController
from walter_rl.control.full_body_pd.gains import FullBodyPdGains


class FullBodyPdPolicy:
  """Callable the Viser and native viewers can step.

  The observation vector is ignored. The loops read the robot and the
  ``twist`` command, which is what the Viser joystick writes.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    gains: FullBodyPdGains | None = None,
  ) -> None:
    self._controller = FullBodyPdController(env.unwrapped, gains)

  @property
  def controller(self) -> FullBodyPdController:
    """Controller that produces the action vector."""
    return self._controller

  def reset(self) -> None:
    """Reset the controller, including the tumble integral."""
    self._controller.reset()

  def control_errors(self, env_idx: int) -> tuple[tuple[str, float, float], ...]:
    """Current loop errors for the Viser motors tab."""
    return self._controller.control_errors(env_idx)

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    """Return the PD action. ``obs`` is unused."""
    del obs
    return self._controller.act()
