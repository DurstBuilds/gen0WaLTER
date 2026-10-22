"""Load an rsl-rl checkpoint for the mjlab viewer."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv


class CheckpointPolicy:
  """Viewer policy that runs a trained actor on the actor observation group."""

  def __init__(
    self,
    model: torch.nn.Module,
    env: ManagerBasedRlEnv | None = None,
  ) -> None:
    self._model = model
    self._env = env

  def reset(self) -> None:
    """No recurrent state to clear."""

  def control_errors(self, env_idx: int) -> tuple[tuple[str, float, float], ...]:
    """Tumble toggle applied on ``env_idx``, clipped to ±1.

    Positive requests tumble. The off dwell stores the negative of the
    network value, and that locked number is what this reports.
    """
    return (("tumble decision", _tumble_decision(self._env, env_idx), 1.0),)

  def __call__(self, obs: dict | TensorDict) -> torch.Tensor:
    """Return the deterministic action for the current observations."""
    return self._model(_as_tensordict(obs))


def _tumble_decision(env: ManagerBasedRlEnv | None, env_idx: int) -> float:
  """Stored tumble toggle for one environment, or 0 when the env has none."""
  if env is None:
    return 0.0
  action = getattr(env.unwrapped, "policy_action", None)
  if not isinstance(action, torch.Tensor):
    return 0.0
  return float(action[env_idx, 0].detach().cpu())


def _as_tensordict(obs: dict | TensorDict) -> TensorDict:
  """Wrap a plain observation dict. The actor reads the ``actor`` group."""
  if isinstance(obs, TensorDict):
    return obs
  actor = obs["actor"]
  return TensorDict(obs, batch_size=[actor.shape[0]])
