"""Actuator datasheets and output-side torque-speed curves.

Each motor lives in its own folder and exports ``MOTOR``. ``get_motor`` and
``assign`` resolve those folders by id, for example
``assign({"hip": "ak70_10_kv100"})``.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path

from walter_rl.assets.motor_models.model import DcMotorParams, MotorModel


def _discover_motors() -> dict[str, MotorModel]:
  motors: dict[str, MotorModel] = {}
  for child in sorted(Path(__file__).parent.iterdir()):
    if not _is_motor_package(child):
      continue
    motor = _load_motor(child.name)
    if motor.motor_id != child.name:
      raise ValueError(f"{child.name} exports motor_id {motor.motor_id!r}.")
    motors[motor.motor_id] = motor
  return motors


def _is_motor_package(path: Path) -> bool:
  return (
    path.is_dir()
    and (path / "__init__.py").is_file()
    and not path.name.startswith("_")
  )


def _load_motor(folder_name: str) -> MotorModel:
  module = importlib.import_module(f"{__name__}.{folder_name}")
  motor = getattr(module, "MOTOR", None)
  if not isinstance(motor, MotorModel):
    raise TypeError(f"{folder_name} must export MOTOR: MotorModel.")
  return motor


MOTORS: dict[str, MotorModel] = _discover_motors()


def get_motor(motor_id: str) -> MotorModel:
  """Return the datasheet registered under ``motor_id``."""
  try:
    return MOTORS[motor_id]
  except KeyError:
    known = ", ".join(sorted(MOTORS))
    raise KeyError(f"Unknown motor {motor_id!r}. Known motors: {known}.") from None


def assign(roles: Mapping[str, str]) -> dict[str, MotorModel]:
  """Resolve ``{"hip": "ak70_10_kv100"}`` to motor models.

  Raises ``KeyError`` when a motor id is not registered.
  """
  return {role: get_motor(motor_id) for role, motor_id in roles.items()}


__all__ = [
  "MOTORS",
  "DcMotorParams",
  "MotorModel",
  "assign",
  "get_motor",
]
