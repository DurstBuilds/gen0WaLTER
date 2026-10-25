"""Human-readable motor names for the Viser sidebar."""

from __future__ import annotations

import re

_LEG = re.compile(r"(?P<leg>fr|fl|br|bl)_(?P<role>hip|knee|axle[12])\Z")
_LEG_LABEL = {
  "fr": "Front Right",
  "fl": "Front Left",
  "br": "Back Right",
  "bl": "Back Left",
}
_ROLE_LABEL = {
  "hip": "Hip",
  "knee": "Knee",
  "axle1": "Wheel 1",
  "axle2": "Wheel 2",
}


def motor_display_label(joint_name: str) -> str:
  """Map hardware joint names to short motor labels.

  Examples:
    ``fr_hip`` -> ``Hip (Front Right)``
    ``bl_knee`` -> ``Knee (Back Left)``
    ``fl_axle2`` -> ``Wheel 2 (Front Left)``
  """
  match = _LEG.match(joint_name)
  if match is None:
    return _fallback_label(joint_name)
  role = _ROLE_LABEL[match.group("role")]
  location = _LEG_LABEL[match.group("leg")]
  return f"{role} ({location})"


def _fallback_label(joint_name: str) -> str:
  name = joint_name.removesuffix("_joint")
  return name.replace("_", " ").title()
