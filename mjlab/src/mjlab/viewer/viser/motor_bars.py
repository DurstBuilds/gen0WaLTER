"""Live torque and speed bars for every joint motor in the Viser sidebar."""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
import viser

from mjlab.actuator.actuator import TransmissionType
from mjlab.utils.mujoco import dof_width

_BAR_HEIGHT_PX = 18
_LABEL_WIDTH_PX = 168


@dataclass(frozen=True)
class JointMotor:
  """One 1-DoF joint motor and the tensors that hold its state."""

  label: str
  entity_name: str
  dof_index: int
  actuator: Any
  target_index: int


def list_joint_motors(scene: Any) -> list[JointMotor]:
  """Return every 1-DoF joint motor on the scene, in actuator order."""
  entities = [
    (name, entity)
    for name, entity in scene.entities.items()
    if any(_is_joint_actuator(actuator) for actuator in entity.actuators)
  ]
  prefix_entity = len(entities) > 1
  motors: list[JointMotor] = []
  for name, entity in entities:
    motors.extend(_entity_motors(name, entity, prefix_entity))
  return motors


def bar_fractions(values: np.ndarray, limits: np.ndarray) -> np.ndarray:
  """Map each value to a 0–1 bar length.

  A finite positive limit is that motor's full scale. Motors without one
  use the largest absolute value in the section, so a single unscaled
  motor does not sit at full width while others are larger.
  """
  fractions = np.zeros(values.shape, dtype=np.float64)
  has_limit = np.isfinite(limits) & (limits > 0.0)
  fractions[has_limit] = np.abs(values[has_limit]) / limits[has_limit]
  if np.any(~has_limit):
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    fractions[~has_limit] = np.abs(values[~has_limit]) / max(peak, 1e-9)
  return np.clip(fractions, 0.0, 1.0)


class SignedBarPanel:
  """Horizontal bars whose length is the absolute value and color is the sign."""

  def __init__(
    self,
    server: viser.ViserServer,
    labels: list[str],
    label_width: int = _LABEL_WIDTH_PX,
  ) -> None:
    """Create an empty HTML bar list in the current GUI folder."""
    self._labels = labels
    self._label_width = label_width
    self._html = server.gui.add_html(_waiting_markup())

  def update(self, values: np.ndarray, limits: np.ndarray) -> None:
    """Replace the bars with the latest values."""
    fractions = bar_fractions(values, limits)
    self._html.content = _bars_markup(
      self._labels, values, fractions, self._label_width
    )

  def cleanup(self) -> None:
    """Remove the HTML element from the GUI."""
    self._html.remove()


_CONTROL_ERROR_LABEL_WIDTH_PX = 200


class ControlErrorPanel:
  """Signed bars for the loop errors of one environment."""

  def __init__(self, server: viser.ViserServer, labels: list[str]) -> None:
    """Create the bar list in the current GUI folder."""
    self._bars = SignedBarPanel(
      server, labels, label_width=_CONTROL_ERROR_LABEL_WIDTH_PX
    )

  def update(self, rows: tuple[tuple[str, float, float], ...]) -> None:
    """Redraw from ``(name, error, full_scale)`` rows."""
    values = np.array([value for _name, value, _scale in rows], dtype=np.float64)
    limits = np.array([scale for _name, _value, scale in rows], dtype=np.float64)
    self._bars.update(values, limits)

  def cleanup(self) -> None:
    """Remove the bar list from the GUI."""
    self._bars.cleanup()


class MotorSidebar:
  """Torque and speed sections for the motors on one scene."""

  def __init__(
    self, server: viser.ViserServer, scene: Any, motors: list[JointMotor]
  ) -> None:
    """Add a Torque folder and a Speed folder under the current GUI context."""
    self._scene = scene
    self._motors = motors
    labels = [motor.label for motor in motors]
    with server.gui.add_folder("Torque (N·m)"):
      self._torque = SignedBarPanel(server, labels)
    with server.gui.add_folder("Speed (rad/s)"):
      self._speed = SignedBarPanel(server, labels)

  def update(self, env_idx: int) -> None:
    """Read the selected environment and redraw both sections."""
    torques, speeds, torque_limits, speed_limits = _sample(
      self._scene, self._motors, env_idx
    )
    self._torque.update(torques, torque_limits)
    self._speed.update(speeds, speed_limits)

  def cleanup(self) -> None:
    """Remove both bar panels from the GUI."""
    self._torque.cleanup()
    self._speed.cleanup()


def _is_joint_actuator(actuator: Any) -> bool:
  return actuator.transmission_type == TransmissionType.JOINT


def _entity_motors(name: str, entity: Any, prefix_entity: bool) -> list[JointMotor]:
  dof_of_joint = _scalar_dof_ids(entity)
  motors: list[JointMotor] = []
  for actuator in entity.actuators:
    if not _is_joint_actuator(actuator):
      continue
    motors.extend(_actuator_motors(name, actuator, dof_of_joint, prefix_entity))
  return motors


def _actuator_motors(
  entity_name: str,
  actuator: Any,
  dof_of_joint: list[int | None],
  prefix_entity: bool,
) -> list[JointMotor]:
  motors: list[JointMotor] = []
  joint_ids = actuator.target_ids.tolist()
  for target_index, joint_index in enumerate(joint_ids):
    dof_index = dof_of_joint[int(joint_index)]
    if dof_index is None:
      continue
    joint_name = actuator.target_names[target_index]
    label = f"{entity_name}/{joint_name}" if prefix_entity else joint_name
    motors.append(
      JointMotor(label, entity_name, dof_index, actuator, target_index)
    )
  return motors


def _scalar_dof_ids(entity: Any) -> list[int | None]:
  """Index into ``joint_vel`` for each non-free joint, or None if it is not 1-DoF."""
  joints = entity.spec.joints
  if joints and joints[0].type == mujoco.mjtJoint.mjJNT_FREE:
    joints = joints[1:]
  ids: list[int | None] = []
  cursor = 0
  for joint in joints:
    width = dof_width(joint.type)
    ids.append(cursor if width == 1 else None)
    cursor += width
  return ids


def _sample(
  scene: Any, motors: list[JointMotor], env_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  count = len(motors)
  torques = np.zeros(count, dtype=np.float64)
  speeds = np.zeros(count, dtype=np.float64)
  torque_limits = np.full(count, math.nan, dtype=np.float64)
  speed_limits = np.full(count, math.nan, dtype=np.float64)
  for entity_name, slots in _slots_by_entity(motors).items():
    _fill_entity(
      scene[entity_name],
      motors,
      slots,
      env_idx,
      torques,
      speeds,
      torque_limits,
      speed_limits,
    )
  return torques, speeds, torque_limits, speed_limits


def _slots_by_entity(motors: list[JointMotor]) -> dict[str, list[int]]:
  grouped: dict[str, list[int]] = {}
  for index, motor in enumerate(motors):
    grouped.setdefault(motor.entity_name, []).append(index)
  return grouped


def _fill_entity(
  entity: Any,
  motors: list[JointMotor],
  slots: list[int],
  env_idx: int,
  torques: np.ndarray,
  speeds: np.ndarray,
  torque_limits: np.ndarray,
  speed_limits: np.ndarray,
) -> None:
  dof_ids = [motors[slot].dof_index for slot in slots]
  torque_row = entity.data.qfrc_actuator[env_idx, dof_ids].detach().cpu().numpy()
  speed_row = entity.data.joint_vel[env_idx, dof_ids].detach().cpu().numpy()
  for column, slot in enumerate(slots):
    motor = motors[slot]
    torques[slot] = float(torque_row[column])
    speeds[slot] = float(speed_row[column])
    torque_limits[slot] = _torque_limit(motor.actuator, env_idx, motor.target_index)
    speed_limits[slot] = _speed_limit(motor.actuator, env_idx, motor.target_index)


def _torque_limit(actuator: Any, env_idx: int, target_index: int) -> float:
  peak = _tensor_limit(actuator, env_idx, target_index, "saturation_effort")
  if math.isfinite(peak):
    return peak
  continuous = _tensor_limit(actuator, env_idx, target_index, "force_limit")
  if math.isfinite(continuous):
    return continuous
  return _cfg_limit(actuator, "effort_limit")


def _speed_limit(actuator: Any, env_idx: int, target_index: int) -> float:
  limit = _tensor_limit(actuator, env_idx, target_index, "velocity_limit_motor")
  if math.isfinite(limit):
    return limit
  return _cfg_limit(actuator, "velocity_limit")


def _tensor_limit(actuator: Any, env_idx: int, target_index: int, attr: str) -> float:
  values = getattr(actuator, attr, None)
  if values is None:
    return math.nan
  return _positive_finite(float(values[env_idx, target_index]))


def _cfg_limit(actuator: Any, attr: str) -> float:
  value = getattr(actuator.cfg, attr, None)
  if value is None:
    return math.nan
  return _positive_finite(float(value))


def _positive_finite(value: float) -> float:
  if math.isfinite(value) and value > 0.0:
    return value
  return math.nan


def _waiting_markup() -> str:
  return '<div style="padding:0.5em;color:#999;font-size:0.85em;">Waiting for data…</div>'


def _bars_markup(
  labels: list[str],
  values: np.ndarray,
  fractions: np.ndarray,
  label_width: int = _LABEL_WIDTH_PX,
) -> str:
  rows = [
    _bar_row(label, float(value), float(fraction), label_width)
    for label, value, fraction in zip(labels, values, fractions, strict=True)
  ]
  return (
    '<div style="padding:0.3em 0.5em;font-family:monospace;">' + "".join(rows) + "</div>"
  )


def _bar_row(label: str, value: float, fraction: float, label_width: int) -> str:
  pct = fraction * 100.0
  color = "#4caf50" if value >= 0.0 else "#f44336"
  text_color = "#fff" if pct > 25.0 else "#ccc"
  safe_label = html.escape(label, quote=True)
  return (
    '<div style="display:flex;align-items:center;margin:2px 0;">'
    f'<span style="min-width:{label_width}px;font-size:0.78em;text-align:right;'
    "padding-right:6px;color:#ddd;white-space:nowrap;overflow:hidden;"
    f'text-overflow:ellipsis;" title="{safe_label}">{safe_label}</span>'
    '<div style="flex:1;background:#333;border-radius:3px;'
    f'height:{_BAR_HEIGHT_PX}px;position:relative;overflow:hidden;">'
    f'<div style="width:{pct:.1f}%;height:100%;background:{color};'
    'border-radius:3px;"></div>'
    f'<span style="position:absolute;right:4px;top:0;line-height:{_BAR_HEIGHT_PX}px;'
    f'font-size:0.72em;color:{text_color};">{_format_value(value)}</span>'
    "</div></div>"
  )


def _format_value(value: float) -> str:
  if not math.isfinite(value):
    return "—"
  if abs(value) < 1e-3 and value != 0.0:
    return f"{value:.2e}"
  return f"{value:.3f}"
