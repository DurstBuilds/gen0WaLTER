"""Smooth-tumble controller for the Walter robot."""

from walter_rl.control.smooth_tumble.gains import SmoothTumbleGains, default_gains
from walter_rl.control.smooth_tumble.policy import SmoothTumblePolicy

__all__ = ["SmoothTumbleGains", "SmoothTumblePolicy", "default_gains"]
