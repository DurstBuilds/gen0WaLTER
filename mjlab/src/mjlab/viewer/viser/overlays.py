"""Overlay managers for Viser viewer orchestration.

These managers intentionally coordinate *when* higher-level updates happen
(env switches, paused/running updates, etc.) while leaving low-level render
handle lifecycle ownership inside :mod:`scene.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import mujoco
import viser

from mjlab.sensor import CameraSensor
from mjlab.viewer.viser.camera_viewer import ViserCameraViewer
from mjlab.viewer.viser.motor_bars import (
  ControlErrorPanel,
  MotorSidebar,
  list_joint_motors,
)
from mjlab.viewer.viser.reward_bar_panel import RewardBarPanel
from mjlab.viewer.viser.term_plotter import ViserTermPlotter


class _EnvProtocol(Protocol):
  @property
  def unwrapped(self) -> Any: ...


class _SceneProtocol(Protocol):
  env_idx: int
  debug_visualization_enabled: bool
  needs_update: bool

  @property
  def show_contact_points(self) -> bool: ...
  @property
  def show_contact_forces(self) -> bool: ...

  def clear_debug_all(self) -> None: ...
  def clear(self) -> None: ...


@dataclass
class ViserTermOverlays:
  """Manage reward, metrics, and motor readout tabs for the Viser viewer."""

  server: viser.ViserServer
  env: _EnvProtocol
  scene: _SceneProtocol
  frame_time: float
  reward_bar_max_terms: int = 20
  policy_getter: Callable[[], Any] | None = None
  motor_joint_label: Callable[[str], str] | None = None
  reward_plotter: ViserTermPlotter | None = None
  reward_bar_panel: RewardBarPanel | None = None
  metrics_plotter: ViserTermPlotter | None = None
  motor_sidebar: MotorSidebar | None = None
  control_errors: ControlErrorPanel | None = None

  def setup_tabs(self, tabs: Any) -> None:
    """Create reward, metrics, and motor tabs based on the environment."""
    if hasattr(self.env.unwrapped, "reward_manager"):
      term_names = [
        name
        for name, _ in self.env.unwrapped.reward_manager.get_active_iterable_terms(
          self.scene.env_idx
        )
      ]
      if term_names:
        with tabs.add_tab("Rewards", icon=viser.Icon.CHART_LINE):
          self.reward_bar_panel = RewardBarPanel(
            self.server,
            term_names,
            update_dt=self.frame_time,
            max_terms=self.reward_bar_max_terms,
          )
          self.reward_plotter = ViserTermPlotter(
            self.server, term_names, name="Reward", env_idx=self.scene.env_idx
          )

    if hasattr(self.env.unwrapped, "metrics_manager"):
      term_names = [
        name
        for name, _ in self.env.unwrapped.metrics_manager.get_active_iterable_terms(
          self.scene.env_idx
        )
      ]
      if term_names:
        with tabs.add_tab("Metrics", icon=viser.Icon.CHART_BAR):
          self.metrics_plotter = ViserTermPlotter(
            self.server, term_names, name="Metric", env_idx=self.scene.env_idx
          )

    self._setup_motor_tab(tabs)

  def _setup_motor_tab(self, tabs: Any) -> None:
    """Add torque, speed, and control-error readouts when the scene has them."""
    sim_scene = getattr(self.env.unwrapped, "scene", None)
    motors = []
    if sim_scene is not None and hasattr(sim_scene, "entities"):
      motors = list_joint_motors(sim_scene, joint_label=self.motor_joint_label)
    if not motors and not _has_control_errors(self._policy()):
      return
    with tabs.add_tab("Motors", icon=viser.Icon.ENGINE):
      if motors:
        self.motor_sidebar = MotorSidebar(self.server, sim_scene, motors)
      self._setup_control_errors()

  def _policy(self) -> Any:
    """Policy currently driving the viewer, if one was provided."""
    if self.policy_getter is None:
      return None
    return self.policy_getter()

  def _setup_control_errors(self) -> None:
    """Add the loop-error folder when the policy reports them."""
    policy = self._policy()
    if not _has_control_errors(policy):
      return
    sample = policy.control_errors(self.scene.env_idx)
    labels = [name for name, _value, _scale in sample]
    with self.server.gui.add_folder("Control Errors"):
      self.control_errors = ControlErrorPanel(self.server, labels)

  def on_env_switch(self) -> None:
    """Clear histories when active environment changes."""
    env_idx = self.scene.env_idx
    if self.reward_plotter:
      self.reward_plotter.clear_histories()
      self.reward_plotter.update_env_idx(env_idx)
    if self.reward_bar_panel:
      self.reward_bar_panel.clear_histories()
    if self.metrics_plotter:
      self.metrics_plotter.clear_histories()
      self.metrics_plotter.update_env_idx(env_idx)

  def update(self, paused: bool) -> None:
    """Update term plots from the selected environment."""
    if (
      self.reward_plotter is not None or self.reward_bar_panel is not None
    ) and not paused:
      terms = list(
        self.env.unwrapped.reward_manager.get_active_iterable_terms(self.scene.env_idx)
      )
      if self.reward_plotter is not None:
        self.reward_plotter.update(terms)
      if self.reward_bar_panel is not None:
        self.reward_bar_panel.update(terms)

    if self.metrics_plotter is not None and not paused:
      terms = list(
        self.env.unwrapped.metrics_manager.get_active_iterable_terms(self.scene.env_idx)
      )
      self.metrics_plotter.update(terms)

    if self.motor_sidebar is not None:
      self.motor_sidebar.update(self.scene.env_idx)
    self._update_control_errors()

  def clear_histories(self) -> None:
    """Clear all overlay histories."""
    self.on_env_switch()

  def cleanup(self) -> None:
    """Cleanup plotter resources."""
    if self.reward_plotter:
      self.reward_plotter.cleanup()
    if self.reward_bar_panel:
      self.reward_bar_panel.cleanup()
    if self.metrics_plotter:
      self.metrics_plotter.cleanup()
    if self.motor_sidebar:
      self.motor_sidebar.cleanup()
    if self.control_errors:
      self.control_errors.cleanup()

  def _update_control_errors(self) -> None:
    """Refresh loop errors for the selected environment."""
    if self.control_errors is None:
      return
    policy = self._policy()
    if not _has_control_errors(policy):
      return
    self.control_errors.update(policy.control_errors(self.scene.env_idx))


def _has_control_errors(policy: Any) -> bool:
  """Whether ``policy`` can report one environment's loop errors."""
  return callable(getattr(policy, "control_errors", None))


@dataclass
class ViserCameraOverlays:
  """Manage camera feed widgets and updates for Viser viewer."""

  server: viser.ViserServer
  env: _EnvProtocol
  mj_model: mujoco.MjModel
  camera_viewers: list[ViserCameraViewer] | None = None

  @property
  def has_cameras(self) -> bool:
    """Whether the environment has any camera sensors."""
    return any(
      isinstance(s, CameraSensor) for s in self.env.unwrapped.scene.sensors.values()
    )

  def setup_controls(self) -> None:
    """Create camera feed controls under the active GUI folder."""
    camera_sensors = [
      sensor
      for sensor in self.env.unwrapped.scene.sensors.values()
      if isinstance(sensor, CameraSensor)
    ]
    if not camera_sensors:
      self.camera_viewers = []
      return

    self.camera_viewers = [
      ViserCameraViewer(self.server, sensor, self.mj_model) for sensor in camera_sensors
    ]

  def update(self, sim_data: Any, env_idx: int, scene_offset: Any) -> None:
    """Push latest camera images/frustums to GUI."""
    if not self.camera_viewers:
      return
    for camera_viewer in self.camera_viewers:
      camera_viewer.update(sim_data, env_idx, scene_offset)

  def cleanup(self) -> None:
    """Cleanup all camera feed widgets."""
    if not self.camera_viewers:
      return
    for camera_viewer in self.camera_viewers:
      camera_viewer.cleanup()


@dataclass
class ViserDebugOverlays:
  """Manage debug visualization queueing and env-switch behavior."""

  env: _EnvProtocol
  scene: _SceneProtocol

  def on_env_switch(self) -> None:
    """Reset debug visuals when switching selected environment."""
    if self.scene.debug_visualization_enabled:
      self.scene.clear_debug_all()

  def queue(self) -> None:
    """Queue environment debug visualizers for the current frame."""
    if self.scene.debug_visualization_enabled and hasattr(
      self.env.unwrapped, "update_visualizers"
    ):
      self.scene.clear()  # Clear queued arrows from previous frame.
      self.env.unwrapped.update_visualizers(self.scene)


@dataclass
class ViserContactOverlays:
  """Manage contact-visualization orchestration from the viewer layer.

  Note: contact mesh creation/update/removal stays in ``ViserMujocoScene``.
  This manager only requests scene refreshes at the right times.
  """

  scene: _SceneProtocol

  def is_enabled(self) -> bool:
    """Whether any contact visualization is currently enabled."""
    return self.scene.show_contact_points or self.scene.show_contact_forces

  def on_env_switch(self) -> None:
    """Request a scene refresh when switching environments with contacts enabled."""
    if self.is_enabled():
      self.scene.needs_update = True
