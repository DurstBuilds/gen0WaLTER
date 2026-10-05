"""Datasheet motor models and output-side torque-speed curves.

Torque and speed ratings are gearbox output. KV, Kt, Ke, phase resistance,
phase inductance, and rotor inertia are motor-side. The curve does not multiply
output torque by the gear ratio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

BUS_VOLTAGES_V: tuple[float, float] = (24.0, 48.0)
DEFAULT_BUS_VOLTAGE_V = 48.0
TorqueEnvelope = Literal["continuous", "peak"]

_RPM_TO_RAD_S = math.tau / 60.0


def rpm_to_rad_s(speed_rpm: float) -> float:
  """Convert a rotational speed from rpm to rad/s."""
  return speed_rpm * _RPM_TO_RAD_S


@dataclass(frozen=True)
class DcMotorParams:
  """Numbers consumed by mjlab's DC motor torque-speed clip.

  ``saturation_effort`` is the stall intercept of the voltage line, in N·m.
  ``velocity_limit`` is the no-load output speed, in rad/s. ``effort_limit`` is
  the torque cap for the selected envelope, in N·m.
  """

  saturation_effort: float
  velocity_limit: float
  effort_limit: float


@dataclass(frozen=True)
class MotorModel:
  """One actuator datasheet and the curve that limits its output torque."""

  motor_id: str
  manufacturer: str
  model: str
  source_url: str
  gear_ratio: float
  rated_torque_nm: float
  peak_torque_nm: float
  rated_speed_rpm: tuple[float, float]
  """Output speed at rated torque, ``(24 V, 48 V)``."""
  no_load_speed_rpm: tuple[float, float]
  """Output no-load speed, ``(24 V, 48 V)``."""
  kv_rpm_per_v: float
  kt_nm_per_a: float
  ke_v_per_krpm: float
  phase_resistance_ohm: float
  phase_inductance_h: float
  rotor_inertia_kg_m2: float
  rated_current_a: float
  peak_current_a: float
  bus_voltage_v: float = DEFAULT_BUS_VOLTAGE_V

  def __post_init__(self) -> None:
    object.__setattr__(self, "bus_voltage_v", float(self.bus_voltage_v))
    _validate_motor(self)

  def at_voltage(self, voltage_v: float) -> MotorModel:
    """Return this model using the datasheet speeds at ``voltage_v``."""
    voltage = float(voltage_v)
    _require_bus_voltage(voltage)
    if voltage == self.bus_voltage_v:
      return self
    return replace(self, bus_voltage_v=voltage)

  def rated_speed_rad_s(self) -> float:
    """Output speed at rated torque for the selected bus voltage."""
    return rpm_to_rad_s(self._speed(self.rated_speed_rpm))

  def no_load_speed_rad_s(self) -> float:
    """Output no-load speed for the selected bus voltage."""
    return rpm_to_rad_s(self._speed(self.no_load_speed_rpm))

  def stall_intercept_nm(self) -> float:
    """Stall torque of the line through rated speed and no-load speed."""
    no_load_rpm = self._speed(self.no_load_speed_rpm)
    rated_rpm = self._speed(self.rated_speed_rpm)
    return self.rated_torque_nm * no_load_rpm / (no_load_rpm - rated_rpm)

  def torque_limit(
    self,
    speed_rad_s: float,
    envelope: TorqueEnvelope = "continuous",
  ) -> float:
    """Positive output torque available at ``speed_rad_s``.

    Continuous holds rated torque until rated speed, then follows the voltage
    line to zero at no-load speed. Peak follows that same line, capped at the
    published peak torque.
    """
    _require_envelope(envelope)
    speed = abs(speed_rad_s)
    no_load = self.no_load_speed_rad_s()
    if speed >= no_load:
      return 0.0
    line = self.stall_intercept_nm() * (1.0 - speed / no_load)
    cap = _envelope_cap(self, envelope)
    return min(cap, line)

  def dc_motor_params(
    self,
    envelope: TorqueEnvelope = "continuous",
  ) -> DcMotorParams:
    """Pack the curve for mjlab's DC motor clip."""
    _require_envelope(envelope)
    return DcMotorParams(
      saturation_effort=self.stall_intercept_nm(),
      velocity_limit=self.no_load_speed_rad_s(),
      effort_limit=_envelope_cap(self, envelope),
    )

  def at_external_gear(
    self,
    external_gear_ratio: float,
    envelope: TorqueEnvelope = "continuous",
  ) -> DcMotorParams:
    """Reflect this curve through an extra gearbox onto the joint.

    Datasheet torque and speed are already after the motor's internal gear.
    Joint torque is that output times ``external_gear_ratio``. Joint speed is
    that output divided by it. The effort cap stays on the envelope and does
    not exceed the voltage-line stall intercept.
    """
    if external_gear_ratio <= 0:
      raise ValueError(
        f"External gear ratio must be positive, got {external_gear_ratio}."
      )
    params = self.dc_motor_params(envelope=envelope)
    gear = external_gear_ratio
    return DcMotorParams(
      saturation_effort=params.saturation_effort * gear,
      velocity_limit=params.velocity_limit / gear,
      effort_limit=min(params.effort_limit, params.saturation_effort) * gear,
    )

  def _speed(self, pair: tuple[float, float]) -> float:
    return pair[_voltage_index(self.bus_voltage_v)]


def _envelope_cap(motor: MotorModel, envelope: TorqueEnvelope) -> float:
  if envelope == "continuous":
    return motor.rated_torque_nm
  return motor.peak_torque_nm


def _require_bus_voltage(voltage_v: float) -> None:
  if voltage_v not in BUS_VOLTAGES_V:
    raise ValueError(f"Bus voltage must be one of {BUS_VOLTAGES_V}, got {voltage_v}.")


def _voltage_index(voltage_v: float) -> int:
  _require_bus_voltage(voltage_v)
  return BUS_VOLTAGES_V.index(voltage_v)


def _require_envelope(envelope: str) -> None:
  if envelope not in ("continuous", "peak"):
    raise ValueError(f"Envelope must be 'continuous' or 'peak', got {envelope!r}.")


def _validate_motor(motor: MotorModel) -> None:
  _require_bus_voltage(motor.bus_voltage_v)
  _validate_speed_pairs(motor)
  if motor.rated_torque_nm <= 0 or motor.peak_torque_nm < motor.rated_torque_nm:
    raise ValueError(f"{motor.motor_id} needs 0 < rated torque <= peak torque.")
  if motor.gear_ratio <= 0:
    raise ValueError(f"{motor.motor_id} gear ratio must be positive.")


def _validate_speed_pairs(motor: MotorModel) -> None:
  if len(motor.rated_speed_rpm) != 2 or len(motor.no_load_speed_rpm) != 2:
    raise ValueError(f"{motor.motor_id} speed pairs must be (24 V, 48 V).")
  pairs = zip(motor.rated_speed_rpm, motor.no_load_speed_rpm, strict=True)
  for rated_rpm, no_load_rpm in pairs:
    if rated_rpm <= 0 or no_load_rpm <= rated_rpm:
      raise ValueError(
        f"{motor.motor_id} needs 0 < rated speed < no-load speed, "
        f"got {rated_rpm} rpm and {no_load_rpm} rpm."
      )
