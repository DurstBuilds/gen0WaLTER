"""Gains for the Walter full-body PD controller.

Edit this file to tune the controller. Each nested group is one loop.
Angles are radians, angular rates are rad/s, and linear speeds are m/s.
A larger proportional gain pulls harder toward the setpoint and is more
likely to oscillate. A larger derivative gain resists how fast the error
is changing. ``sign`` is +1 or -1; flip it if that loop runs away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PitchGains:
  """Level the torso pitch with the hip joints.

  ``kp`` is outward hip splay, in radians, per unit of ``projected_gravity_b.x``
  (positive when the nose is down). ``kd`` is splay per rad/s of pitch rate.
  A level body commands 0, which is a vertical thigh. Only the low end swings,
  and it swings away from the body.

  ``actuator_kp`` and ``actuator_kd`` are the MuJoCo hip servo, in N·m/rad and
  N·m·s/rad. They override the soft training gains on the PD env so wheel
  torque cannot fold a thigh off vertical. They apply on the next launch.
  """

  kp: float = 6
  kd: float = 0.4
  sign: float = 1.0
  actuator_kp: float = 800.0
  actuator_kd: float = 100.0


@dataclass
class RollGains:
  """Level the torso roll by extending the downhill legs.

  ``kp`` is how far the inner wheel drops, in radians, per unit of
  ``projected_gravity_b.y`` (positive when the left side is down). ``kd``
  damps roll rate. ``q_max`` is the shin angle from horizontal at which the
  inner wheel is straight down. ``pi / 2`` is that vertical stop. Thigh pitch
  and the knee slider move the zero, and both count against this limit, so
  the extra fold is shorter than ``q_max`` and the joint command is smaller.
  Rear knees go positive and front knees go negative so both push with the
  wheel toward the body, on the left and the right. The flat pose is shifted
  by the thigh pitch, so a hip angle leaves the shin level.

  ``actuator_kp`` and ``actuator_kd`` are the MuJoCo knee servo, in N·m/rad
  and N·m·s/rad, used while Tank Mode is off. They override the soft training
  gains on the PD env and apply on the next launch.

  ``tank_actuator_kp`` and ``tank_actuator_kd`` replace that servo while Tank
  Mode is checked, so the leg yields instead of holding the knee angle.
  Those two switch as soon as the checkbox changes.
  """

  kp: float = 8.0
  kd: float = 0.5
  q_max: float = math.pi / 2.0
  actuator_kp: float = 250.0
  actuator_kd: float = 50.0
  tank_actuator_kp: float = 250.0
  tank_actuator_kd: float = 50.0
  sign: float = 1.0


@dataclass
class VelocityGains:
  """Track forward speed with all eight wheels.

  ``full_speed_command`` is the forward command, in m/s, that requests the
  wheel motor's maximum speed. That maximum is the no-load speed, or the
  rated speed while the rated speed cap is on. Smaller commands scale down
  from it. ``kp`` adds wheel rate, in rad/s, per m/s of speed error.
  ``wheel_speed_limit`` is an extra clip in rad/s and cannot raise the
  motor's own limit. ``wheel_radius`` turns a yaw rate into a left/right
  wheel difference.
  """

  kp: float = 15.0
  wheel_radius: float = 0.11
  wheel_speed_limit: float = 50.0
  full_speed_command: float = 10.0


@dataclass
class YawRateGains:
  """Skid-steer the wheels to the commanded yaw rate.

  The setpoint is the twist command's angular velocity, in rad/s. That is
  the value the Viser ``ang_vel_z`` slider writes. Zero on the slider
  commands zero yaw rate.

  ``kp`` is dimensionless. The wheel yaw command is
  ``ang_vel_z + kp * (ang_vel_z - measured_yaw_rate)``. ``yaw_rate_limit``
  caps that command so the wheels do not break traction.
  ``track_half_width`` is half the distance between the left and right
  wheel centers. A positive yaw rate speeds the right wheels and slows
  the left wheels.
  """

  kp: float = 5.0
  track_half_width: float = 0.295
  yaw_rate_limit: float = 2.5
  sign: float = 1.0


@dataclass
class TumbleGains:
  """Spin every knee about +Y after the body falls behind a forward command.

  The integrator adds positive ``v_cmd_x - v_x`` only while the forward
  command is above ``min_forward_cmd`` and that speed error exceeds
  ``min_speed_error``. A smaller shortfall does not start integration. Its
  state is meters of missed distance, clamped to ``cap`` so a long stall
  cannot leave a large tail. Tumbling starts above ``threshold`` and stops
  below ``release``. Any other time the integral leaks toward zero at
  ``leak`` (1/s). ``cap`` should stay above ``threshold``.

  ``rate`` is the knee angular velocity, in rad/s. Positive is +Y on every
  knee axis, which drops the front wheel and lifts the rear.

  ``wheel_speed`` is the wheel rate, in rad/s, written to every wheel while
  tumbling. Positive matches a positive forward command. The velocity and
  yaw loops resume once tumbling ends, including during the knee settle.
  """

  threshold: float = 0.2
  release: float = 0.1
  cap: float = 0.5
  leak: float = 2.5
  min_forward_cmd: float = 0.2
  min_speed_error: float = 0.6
  rate: float = 3.0
  wheel_speed: float = 0.0


@dataclass
class WaistGains:
  """Hold ``waist`` at zero.

  These are the MuJoCo position-actuator gains, in N·m/rad and N·m·s/rad.
  ``effort_limit`` is the torque cap, in N·m. They are read when the
  environment is built, so a change applies on the next launch.
  """

  kp: float = 2000.0
  kd: float = 50.0
  effort_limit: float = 288.0


@dataclass
class FullBodyPdGains:
  """All six loops. Construct one and pass it to the controller."""

  pitch: PitchGains = field(default_factory=PitchGains)
  roll: RollGains = field(default_factory=RollGains)
  velocity: VelocityGains = field(default_factory=VelocityGains)
  yaw: YawRateGains = field(default_factory=YawRateGains)
  tumble: TumbleGains = field(default_factory=TumbleGains)
  waist: WaistGains = field(default_factory=WaistGains)


def default_gains() -> FullBodyPdGains:
  """Return the gains currently written in this file."""
  return FullBodyPdGains()


def format_gains(gains: FullBodyPdGains) -> str:
  """Format the tunable numbers on a few lines for rollout logs."""
  pitch = gains.pitch
  roll = gains.roll
  velocity = gains.velocity
  yaw = gains.yaw
  tumble = gains.tumble
  waist = gains.waist
  return "\n".join(
    (
      f"pitch kp={pitch.kp} kd={pitch.kd} sign={pitch.sign} "
      f"actuator_kp={pitch.actuator_kp} actuator_kd={pitch.actuator_kd}",
      f"roll kp={roll.kp} kd={roll.kd} q_max={roll.q_max} sign={roll.sign} "
      f"actuator_kp={roll.actuator_kp} actuator_kd={roll.actuator_kd} "
      f"tank_actuator_kp={roll.tank_actuator_kp} "
      f"tank_actuator_kd={roll.tank_actuator_kd}",
      f"velocity kp={velocity.kp} radius={velocity.wheel_radius} "
      f"limit={velocity.wheel_speed_limit} "
      f"full_cmd={velocity.full_speed_command}",
      f"yaw kp={yaw.kp} limit={yaw.yaw_rate_limit} "
      f"track_half={yaw.track_half_width} sign={yaw.sign}",
      f"tumble threshold={tumble.threshold} release={tumble.release} "
      f"cap={tumble.cap} leak={tumble.leak} "
      f"min_cmd={tumble.min_forward_cmd} min_error={tumble.min_speed_error} "
      f"rate={tumble.rate} wheel_speed={tumble.wheel_speed}",
      f"waist kp={waist.kp} kd={waist.kd} effort={waist.effort_limit}",
    )
  )
