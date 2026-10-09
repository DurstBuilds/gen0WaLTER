"""Viser helpers for Walter play scripts."""

from __future__ import annotations

from mjlab.viewer import ViserPlayViewer

from walter_rl.viewer.motor_labels import motor_display_label


class WalterViserPlayViewer(ViserPlayViewer):
  """Viser play viewer with Walter motor labels on the Motors tab."""

  def __init__(self, *args, **kwargs) -> None:
    kwargs.setdefault("motor_joint_label", motor_display_label)
    super().__init__(*args, **kwargs)


__all__ = ["WalterViserPlayViewer", "motor_display_label"]
