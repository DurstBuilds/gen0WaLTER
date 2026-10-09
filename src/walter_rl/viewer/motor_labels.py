"""Human-readable motor names for the Viser sidebar."""

from __future__ import annotations

import re

_LEG = re.compile(
  r"(?P<section>head|torso)_(?P<side>left|right)_(?P<role>.+)_joint\Z"
)


def motor_display_label(joint_name: str) -> str:
  """Map ``walter_fs`` joint names to short motor labels.

  Examples:
    ``torso_left_thigh_joint`` -> ``Hip (Torso Left)``
    ``head_left_thigh_shin_joint`` -> ``Knee (Head Left)``
    ``torso_right_shin_rear_wheel_joint`` -> ``Rear Wheel (Torso Right)``
  """
  match = _LEG.match(joint_name)
  if match is None:
    return _fallback_label(joint_name)

  section = match.group("section").title()
  side = match.group("side").title()
  location = f"{section} {side}"
  role = match.group("role")

  if role == "thigh":
    return f"Hip ({location})"
  if role == "thigh_shin":
    return f"Knee ({location})"
  wheel = re.fullmatch(r"shin_(?P<which>front|rear)_wheel", role)
  if wheel is not None:
    which = wheel.group("which").title()
    return f"{which} Wheel ({location})"

  return _fallback_label(joint_name)


def _fallback_label(joint_name: str) -> str:
  name = joint_name.removesuffix("_joint")
  return name.replace("_", " ").title()
