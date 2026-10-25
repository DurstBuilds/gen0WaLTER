"""PPO runner settings for rsl-rl 5.4."""

from __future__ import annotations

_ACTOR_HIDDEN = (512, 256, 128)
_TEACHER_STEPS = 64
_TEACHER_ENTROPY = 0.005

# Shared by training, playback, and checkpoint loading.
STATE_MACHINE_ACTOR_HIDDEN = (256, 128, 64)
STATE_MACHINE_CRITIC_HIDDEN = (512, 256, 128)
_STATE_MACHINE_STEPS = 24
_STATE_MACHINE_ENTROPY = 0.003


def walter_ppo_runner_cfg(
  *,
  max_iterations: int,
  experiment_name: str = "walter_fs_velocity",
  save_interval: int = 50,
  clip_actions: float | None = None,
  hidden_dims: tuple[int, ...] = _ACTOR_HIDDEN,
  critic_hidden_dims: tuple[int, ...] | None = None,
  num_steps_per_env: int = _TEACHER_STEPS,
  entropy_coef: float = _TEACHER_ENTROPY,
  distribution: str = "gaussian",
) -> dict:
  """Build the dict ``OnPolicyRunner`` reads for the WaLTER velocity task.

  Actor and critic are separate MLPs. Metrics go to TensorBoard under the
  run log directory. ``rnd_cfg`` is present and empty because the rsl-rl 5.4
  runner indexes that key every iteration.
  """
  critic_dims = hidden_dims if critic_hidden_dims is None else critic_hidden_dims
  return {
    "seed": 42,
    "num_steps_per_env": num_steps_per_env,
    "max_iterations": max_iterations,
    "save_interval": save_interval,
    "experiment_name": experiment_name,
    "run_name": "",
    "logger": "tensorboard",
    "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
    "clip_actions": clip_actions,
    "upload_model": False,
    "check_for_nan": True,
    "actor": _mlp_cfg(
      stochastic=True, hidden_dims=hidden_dims, distribution=distribution
    ),
    "critic": _mlp_cfg(stochastic=False, hidden_dims=critic_dims),
    "algorithm": {
      "class_name": "PPO",
      "num_learning_epochs": 5,
      "num_mini_batches": 4,
      "learning_rate": 1.0e-3,
      "schedule": "adaptive",
      "gamma": 0.99,
      "lam": 0.95,
      "entropy_coef": entropy_coef,
      "desired_kl": 0.01,
      "max_grad_norm": 1.0,
      "value_loss_coef": 1.0,
      "use_clipped_value_loss": True,
      "clip_param": 0.2,
      "normalize_advantage_per_mini_batch": False,
      "optimizer": "adam",
      "share_cnn_encoders": False,
      "rnd_cfg": None,
      "symmetry_cfg": None,
    },
  }


def teacher_ppo_runner_cfg(
  *,
  max_iterations: int,
  experiment_name: str = "walter_fs_teacher",
  save_interval: int = 50,
  clip_actions: float | None = None,
) -> dict:
  """PPO config for the 17-D teacher student with state-machine MLP widths."""
  return walter_ppo_runner_cfg(
    max_iterations=max_iterations,
    experiment_name=experiment_name,
    save_interval=save_interval,
    clip_actions=clip_actions,
    hidden_dims=STATE_MACHINE_ACTOR_HIDDEN,
    critic_hidden_dims=STATE_MACHINE_CRITIC_HIDDEN,
  )


def state_machine_ppo_runner_cfg(
  *,
  max_iterations: int,
  experiment_name: str = "walter_fs_state_machine",
  save_interval: int = 50,
  clip_actions: float | None = 1.0,
) -> dict:
  """PPO config for the 10 Hz tumble toggle and tank hip.

  The actor is a Beta distribution on ``[-1, 1]``, so the policy cannot
  walk the std upward the way a clipped Gaussian does. A rollout step is
  2.4 s at 10 Hz.
  """
  return walter_ppo_runner_cfg(
    max_iterations=max_iterations,
    experiment_name=experiment_name,
    save_interval=save_interval,
    clip_actions=clip_actions,
    hidden_dims=STATE_MACHINE_ACTOR_HIDDEN,
    critic_hidden_dims=STATE_MACHINE_CRITIC_HIDDEN,
    num_steps_per_env=_STATE_MACHINE_STEPS,
    entropy_coef=_STATE_MACHINE_ENTROPY,
    distribution="beta",
  )


def _mlp_cfg(
  *,
  stochastic: bool,
  hidden_dims: tuple[int, ...],
  distribution: str = "gaussian",
) -> dict:
  """One MLPModel entry. The critic has no action distribution."""
  cfg = {
    "class_name": "MLPModel",
    "hidden_dims": hidden_dims,
    "activation": "elu",
    "obs_normalization": True,
  }
  if stochastic:
    cfg["distribution_cfg"] = _distribution_cfg(distribution)
  return cfg


def _distribution_cfg(distribution: str) -> dict:
  """Action distribution for the actor."""
  if distribution == "beta":
    return {
      "class_name": "BetaDistribution",
      "action_range": (-1.0, 1.0),
    }
  if distribution == "gaussian":
    return {
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    }
  raise ValueError(f"Unknown action distribution: {distribution}")
