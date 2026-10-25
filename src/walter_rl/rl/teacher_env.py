"""Environment that scores the student against the full-body PD teacher."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.types import VecEnvStepReturn

from walter_rl.control.full_body_pd.controller import FullBodyPdController
from walter_rl.control.full_body_pd.gains import TumbleGains
from walter_rl.tasks.velocity.mdp.rewards import forward_match_step, match_gate_from_step

# Policy vector: hips, wheels, drive_state, knee positions.
POLICY_ACTION_DIM = 17
DRIVE_STATE_THRESHOLD = 0.5
_DRIVE_STATE_COLUMN = 12
_HIP_POLICY = slice(0, 4)
_WHEEL_POLICY = slice(4, 12)
_KNEE_POLICY = slice(13, 17)
# Plant vector stays hip_pos, knee_pos, wheel_vel.
_HIP_PLANT = slice(0, 4)
_KNEE_PLANT = slice(4, 8)
_WHEEL_PLANT = slice(8, 16)
_PLANT_ACTION_DIM = 16
_ACTION_TERM_ORDER = ("hip_pos", "knee_pos", "wheel_vel")
_HIP_NAMES = ("br_hip", "bl_hip", "fr_hip", "fl_hip")
_KNEE_NAMES = ("br_knee", "bl_knee", "fr_knee", "fl_knee")
_WHEEL_NAMES = (
  "br_axle1",
  "br_axle2",
  "bl_axle1",
  "bl_axle2",
  "fr_axle1",
  "fr_axle2",
  "fl_axle1",
  "fl_axle2",
)
_TERM_TARGET_NAMES = {
  "hip_pos": _HIP_NAMES,
  "knee_pos": _KNEE_NAMES,
  "wheel_vel": _WHEEL_NAMES,
}


class TeacherStudentEnv(ManagerBasedRlEnv):
  """Velocity env that records the PD action before each physics step.

  Rewards run after the step, so calling the teacher there would score the
  next state. ``teacher_action`` is taken from the same state the student
  saw. A flat forward scan stores the reactive PD action. A step that turns
  the match gate off stores that action with tumble forced on.

  The student outputs 17 numbers: four hips, eight wheels, ``drive_state``,
  and four knee positions. ``step`` decodes that into the 16-D plant command.
  Inside ±0.5 the knee columns are the position actions. Outside that, every
  knee integrates one shared reference at ``drive_state * TumbleGains.rate``.
  ``follow_teacher`` steps the stored 16-D PD target and skips the decode.
  """

  def __init__(
    self,
    cfg: ManagerBasedRlEnvCfg,
    device: str,
    render_mode: str | None = None,
  ) -> None:
    super().__init__(cfg, device=device, render_mode=render_mode)
    self.follow_teacher = False
    self._teacher = FullBodyPdController(self)
    self._assert_plant_layout()
    action_dim = self.action_manager.total_action_dim
    self.teacher_action = torch.zeros(self.num_envs, action_dim, device=self.device)
    self.policy_action = torch.zeros(
      self.num_envs, POLICY_ACTION_DIM, device=self.device
    )
    self._tumble_applied = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._spin_mask = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._knee_ids = self._knee_joint_ids()
    self._knee_ref = torch.zeros(
      self.num_envs, self._knee_ids.numel(), device=self.device
    )
    self._knee_scale = _knee_action_scale(self)
    self._knee_rate = TumbleGains().rate

  @property
  def teacher(self) -> FullBodyPdController:
    """Reactive PD controller snapshotted at the start of ``step``."""
    return self._teacher

  def step(self, action: torch.Tensor) -> VecEnvStepReturn:
    """Snapshot the teacher target, then step the applied action.

    PD playback applies the reactive 16-D action and skips the decode.
    Training stores the reactive action on a closed scan and the forced
    tumble on an open scan, and steps the decoded 16-D plant command.
    """
    reactive = self._teacher.act()
    if self.follow_teacher:
      self.teacher_action = reactive.detach()
      self._tumble_applied = self._teacher.tumbling.detach().clone()
      return super().step(self.teacher_action)
    forced = self._teacher.overlay_forced_tumble(
      reactive, self._scan_forces_tumble()
    )
    self.teacher_action = forced.detach()
    return super().step(self._decode_policy_action(action))

  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    """Clear spin state so the next tumble reseeds from the measured angle."""
    super()._reset_idx(env_ids)
    rows = slice(None) if env_ids is None else env_ids
    self._clear_drive_state(rows)

  def _decode_policy_action(self, action: torch.Tensor) -> torch.Tensor:
    """Copy hips and wheels, and switch knees on ``drive_state``."""
    if action.ndim != 2 or action.shape != (self.num_envs, POLICY_ACTION_DIM):
      raise ValueError(
        f"Teacher policy action must have shape ({self.num_envs}, "
        f"{POLICY_ACTION_DIM}), got {tuple(action.shape)}."
      )
    command = action.to(device=self.device)
    self.policy_action = command.detach().clone()
    plant = torch.zeros(
      self.num_envs, _PLANT_ACTION_DIM, device=self.device
    )
    plant[:, _HIP_PLANT] = command[:, _HIP_POLICY]
    plant[:, _WHEEL_PLANT] = command[:, _WHEEL_POLICY]
    spinning = self._write_knees(plant, command)
    self._spin_mask = spinning
    self._tumble_applied = spinning
    return plant

  def _write_knees(
    self, plant: torch.Tensor, command: torch.Tensor
  ) -> torch.Tensor:
    """Write knee positions, or the shared integrated reference."""
    drive = command[:, _DRIVE_STATE_COLUMN]
    spinning = drive.abs() > DRIVE_STATE_THRESHOLD
    holding = ~spinning
    plant[holding, _KNEE_PLANT] = command[holding, _KNEE_POLICY]
    if bool(spinning.any()):
      self._integrate_knee_spin(spinning, drive)
      plant[spinning, _KNEE_PLANT] = (
        self._knee_ref[spinning] / self._knee_scale
      )
    return spinning

  def _integrate_knee_spin(
    self, spinning: torch.Tensor, drive: torch.Tensor
  ) -> None:
    """Seed each knee on entry, then add the same signed rate step."""
    just_on = spinning & ~self._spin_mask
    if bool(just_on.any()):
      measured = self.scene["robot"].data.joint_pos[just_on][:, self._knee_ids]
      self._knee_ref[just_on] = measured
    increment = drive.unsqueeze(-1) * self._knee_rate * self.step_dt
    self._knee_ref[spinning] = self._knee_ref[spinning] + increment[spinning]

  def _clear_drive_state(self, env_ids: torch.Tensor | slice) -> None:
    """Zero the spin latch, knee reference, and stored policy action."""
    self._spin_mask[env_ids] = False
    self._knee_ref[env_ids] = 0.0
    self.policy_action[env_ids] = 0.0
    self._tumble_applied[env_ids] = False

  def _assert_plant_layout(self) -> None:
    """Require the action-manager joint order the policy vector is built on."""
    found_terms = tuple(self.action_manager.active_terms)
    if found_terms != _ACTION_TERM_ORDER:
      raise RuntimeError(
        f"Action terms {found_terms} do not match {_ACTION_TERM_ORDER}."
      )
    if self.action_manager.total_action_dim != _PLANT_ACTION_DIM:
      raise RuntimeError(
        "Plant action dim "
        f"{self.action_manager.total_action_dim} is not {_PLANT_ACTION_DIM}."
      )
    for term_name, wanted in _TERM_TARGET_NAMES.items():
      found = tuple(self.action_manager.get_term(term_name).target_names)
      if found != wanted:
        raise RuntimeError(
          f"{term_name} targets {found} do not match {wanted}."
        )

  def _knee_joint_ids(self) -> torch.Tensor:
    """Joint indices in the same order as the knee action columns."""
    ids, matched = self.scene["robot"].find_joints(
      _KNEE_NAMES, preserve_order=True
    )
    if tuple(matched) != _KNEE_NAMES:
      raise RuntimeError(f"Expected knee joints {_KNEE_NAMES}, found {matched}.")
    return torch.tensor(ids, dtype=torch.long, device=self.device)

  def _scan_forces_tumble(self) -> torch.Tensor:
    """True where the forward step has turned the teacher-match gate off.

    The gate is 0 at a step of 0.08 m or more. Flatter scans, including a
    hard slope still inside the match ramp, do not force a tumble.
    """
    match = match_gate_from_step(forward_match_step(self))
    return match <= 0.0


def _knee_action_scale(env: ManagerBasedRlEnv) -> float:
  """Radians of knee angle per unit of the knee action column."""
  scale = env.action_manager.get_term("knee_pos").scale
  if not isinstance(scale, float):
    raise TypeError("Knee action scale must be one float for every knee.")
  return scale
