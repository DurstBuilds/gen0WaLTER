"""Rigid masses welded onto the robot.

Each payload is an MJCF asset plus one registry entry. Geometry and mass live
in the XML. The mount pose lives here, in the parent body frame. A payload
body has no joint, so MuJoCo welds its mass into the parent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco

_PAYLOAD_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Payload:
  """A rigid mass welded to a named robot body."""

  name: str
  xml_path: Path
  parent_body: str
  pos: tuple[float, float, float]
  quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


# Mount the rider on the head. The top face is z = 0.09 m and +x faces forward.
_PAYLOADS: dict[str, Payload] = {
  "rider": Payload(
    name="rider",
    xml_path=_PAYLOAD_DIR / "rider.xml",
    parent_body="head",
    pos=(0.1, 0.0, 0.09),
  ),
}


def available_payloads() -> tuple[str, ...]:
  """Return registered payload names, in registry order."""
  return tuple(_PAYLOADS)


def attach_payload(spec: mujoco.MjSpec, name: str) -> None:
  """Weld ``name`` onto its parent body.

  Raises:
    KeyError: ``name`` is not a registered payload.
  """
  payload = _payload(name)
  child = mujoco.MjSpec.from_file(str(payload.xml_path))
  frame = spec.body(payload.parent_body).add_frame(
    pos=list(payload.pos),
    quat=list(payload.quat),
  )
  spec.attach(child=child, prefix="", frame=frame)


def _payload(name: str) -> Payload:
  """Return one registry entry, or raise ``KeyError`` with the known names."""
  try:
    return _PAYLOADS[name]
  except KeyError:
    known = ", ".join(_PAYLOADS)
    raise KeyError(f"Unknown payload {name!r}. Known payloads: {known}.") from None
