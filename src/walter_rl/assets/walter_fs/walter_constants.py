"""Full-scale WaLTER articulation, collisions, and nominal pose.

The MJCF is produced by ``generate-walter --fs`` and checked in. Bare generator
motors are removed here. mjlab DC motors apply the CubeMars torque-speed curves
through the external joint gears, with the FS position and velocity gains.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import mujoco
import numpy as np

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from walter_rl.assets.motor_models import get_motor

WALTER_FS_XML: Path = Path(__file__).parent / "walter_fs.xml"
assert WALTER_FS_XML.exists()

# Extra reduction after the actuator gearbox. Datasheet torque and speed are
# already output-side, so these ratios are applied on top.
HIP_GEAR_RATIO = 33
KNEE_GEAR_RATIO = 1.6
WHEEL_GEAR_RATIO = 1.0

# Joint-side PD from the FS YAML. These hold the zero pose while the wheels drive.
HIP_STIFFNESS = 20.0
HIP_DAMPING = 10.0
KNEE_STIFFNESS = 20.0
KNEE_DAMPING = 10.0
# Velocity gain (Nms/rad). Full wheel effort at about a 7 rad/s tracking error.
WHEEL_VELOCITY_GAIN = 15.0

# Action scales. Knee scale is radians added every physics step (dt = 5 ms),
# so 0.03 corresponds to about 6 rad/s at action = 1.
KNEE_ACTION_SCALE = 0.03
WHEEL_ACTION_SCALE = 30.0

# Settled torso height with every hinge at zero: wheel radius 0.1 m, shin
# hung 0.18 m below the hip, so the wheel centers sit on the ground at z = 0.28.
NOMINAL_ROOT_HEIGHT = 0.28

# Hips are straight down at zero. ±70 deg is the mechanical stop, and the
# position action uses that same range so a unit action can reach it.
HIP_ANGLE_LIMIT = math.radians(70.0)
_LEG_DROP_M = 0.18
_WHEEL_RADIUS_M = 0.10
HIP_ACTION_SCALE = HIP_ANGLE_LIMIT
# Clearance of the torso origin with the wheels planted: radius plus the
# vertical component of the leg drop.
RIDE_HEIGHT_MAX = NOMINAL_ROOT_HEIGHT
RIDE_HEIGHT_MIN = _WHEEL_RADIUS_M + _LEG_DROP_M * math.cos(HIP_ANGLE_LIMIT)

_WHEEL_GEOM = r".*_wheel_geom"
_BUMP_GEOM = r".*_bump.*"
_BODY_GEOM = r".*_geom"
_BUMP_RADIUS = 0.04

# Generator wheel solref is [-80800, -112] (stiffness plus negative damping).
# Override to a passive time-constant / damping-ratio pair.
_WHEEL_SOLREF = (0.02, 1.0)
_WHEEL_FRICTION = (1.0, 0.005, 0.0001)


def get_spec() -> mujoco.MjSpec:
  """Load the FS MJCF and prepare it for mjlab actuation."""
  spec = mujoco.MjSpec.from_file(str(WALTER_FS_XML))
  _name_body_geoms(spec)
  _limit_hip_joints(spec)
  _stabilize_wheel_contacts(spec)
  _add_contact_bumps(spec)
  _strip_generator_actuators(spec)
  _add_imu(spec)
  return spec


def _limit_hip_joints(spec: mujoco.MjSpec) -> None:
  """Stop each hip ±70 deg from hanging straight down.

  Knee and wheel hinges stay unlimited so a leg assembly can wind through a step.
  """
  for joint in spec.joints:
    if not joint.name.endswith("_thigh_joint"):
      continue
    joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    joint.range[:] = np.array([-HIP_ANGLE_LIMIT, HIP_ANGLE_LIMIT])


def get_walter_fs_robot_cfg() -> EntityCfg:
  """Return a fresh full-scale WaLTER entity config."""
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, NOMINAL_ROOT_HEIGHT),
      joint_pos={".*": 0.0},
      joint_vel={".*": 0.0},
    ),
    collisions=(_COLLISION,),
    spec_fn=get_spec,
    articulation=WALTER_FS_ARTICULATION,
  )


def _name_body_geoms(spec: mujoco.MjSpec) -> None:
  """Name the torso and head boxes so contact sensors can match them."""
  for body in spec.bodies:
    for geom in body.geoms:
      if not geom.name:
        geom.name = f"{body.name}_geom"


def _stabilize_wheel_contacts(spec: mujoco.MjSpec) -> None:
  """Replace the generator's negative wheel damping with a passive solref."""
  for body in spec.bodies:
    for geom in body.geoms:
      if "wheel" in geom.name:
        geom.solref = np.array(_WHEEL_SOLREF)


def _strip_generator_actuators(spec: mujoco.MjSpec) -> None:
  """Drop bare motors so mjlab can attach position and velocity actuators."""
  for actuator in list(spec.actuators):
    spec.delete(actuator)


def _add_contact_bumps(spec: mujoco.MjSpec) -> None:
  """Add small spheres so body contact does not flood heightfield collisions.

  The torso box and leg capsules each generate dozens of heightfield contacts
  when they penetrate, and MuJoCo drops the pair once that count hits 50.
  Wheels keep their ellipsoids. Belly and leg bumps are the only other
  colliders, and torso/head bumps are what the fall sensor watches.
  """
  _add_belly_bumps(spec.body("torso"), ( -0.15, 0.15))
  _add_belly_bumps(spec.body("head"), (-0.08, 0.08))
  for body in spec.bodies:
    if body.name.endswith("_thigh"):
      _add_sphere(body, f"{body.name}_bump", [0.0, 0.0, -0.09])
    elif body.name.endswith("_shin"):
      _add_sphere(body, f"{body.name}_bump", [0.0, 0.0, 0.0])


def _add_belly_bumps(body: mujoco.MjsBody, xs: tuple[float, float]) -> None:
  for index, x in enumerate(xs):
    _add_sphere(body, f"{body.name}_bump_{index}", [x, 0.0, -0.05])


def _add_sphere(body: mujoco.MjsBody, name: str, pos: list[float]) -> None:
  body.add_geom(
    name=name,
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[_BUMP_RADIUS, 0.0, 0.0],
    pos=pos,
    density=0.0,
  )


def _add_imu(spec: mujoco.MjSpec) -> None:
  """Add the gyro and velocimeter the velocity task reads as robot/imu_*."""
  spec.body("torso").add_site(name="imu", pos=[0.0, 0.0, 0.0], size=[0.01, 0.01, 0.01])
  spec.add_sensor(
    name="imu_ang_vel",
    type=mujoco.mjtSensor.mjSENS_GYRO,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu",
  )
  spec.add_sensor(
    name="imu_lin_vel",
    type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu",
  )


_COLLISION = CollisionCfg(
  geom_names_expr=(_WHEEL_GEOM, _BUMP_GEOM, _BODY_GEOM),
  # Boxes and capsules stay visual. Wheels and small bumps hit the terrain
  # (contype 1) but not each other (conaffinity 0).
  # Bit 1 is terrain. Bit 2 is wheel-to-wheel only. Wheels keep terrain
  # contact through contype, and conaffinity 2 lets them hit each other
  # without also hitting the belly and leg bumps.
  contype={_WHEEL_GEOM: 3, _BUMP_GEOM: 1, _BODY_GEOM: 0},
  conaffinity={_WHEEL_GEOM: 2, _BUMP_GEOM: 0, _BODY_GEOM: 0},
  condim={_WHEEL_GEOM: 6, _BUMP_GEOM: 3, _BODY_GEOM: 3},
  priority={_WHEEL_GEOM: 1, _BUMP_GEOM: 1, _BODY_GEOM: 0},
  friction={
    _WHEEL_GEOM: _WHEEL_FRICTION,
    _BUMP_GEOM: (0.8,),
    _BODY_GEOM: (0.6,),
  },
  solref={
    _WHEEL_GEOM: _WHEEL_SOLREF,
    _BUMP_GEOM: (0.02, 1.0),
    _BODY_GEOM: (0.02, 1.0),
  },
)


# Joint pattern, motor id, external gear, stiffness, damping. Hip pattern is
# only the thigh joint; the knee name is ``thigh_shin_joint``.
_DC_MOTORS = (
  (".*_thigh_joint", "ak70_10_kv100", HIP_GEAR_RATIO, HIP_STIFFNESS, HIP_DAMPING),
  (
    ".*_thigh_shin_joint",
    "ak80_64_kv80",
    KNEE_GEAR_RATIO,
    KNEE_STIFFNESS,
    KNEE_DAMPING,
  ),
  (".*wheel_joint", "ak10_9_v2_kv60", WHEEL_GEAR_RATIO, 0.0, WHEEL_VELOCITY_GAIN),
)


def continuous_effort_limit(target_names: tuple[str, ...] | list[str]) -> float | None:
  """Rated joint torque for these DC-motor targets, in N·m.

  Returns None when none of the names is a Walter hip, knee, or wheel.
  """
  drive = _matched_drive(target_names)
  if drive is None:
    return None
  motor_id, gear = drive
  curve = get_motor(motor_id).at_external_gear(gear, envelope="continuous")
  return curve.effort_limit


def rated_speed_limit(target_names: tuple[str, ...] | list[str]) -> float | None:
  """Rated joint speed for these DC-motor targets, in rad/s.

  Returns None when none of the names is a Walter hip, knee, or wheel.
  """
  drive = _matched_drive(target_names)
  if drive is None:
    return None
  motor_id, gear = drive
  return get_motor(motor_id).rated_speed_rad_s() / gear


def _matched_drive(target_names: tuple[str, ...] | list[str]) -> tuple[str, float] | None:
  """Motor id and external gear for the first matching target."""
  for name in target_names:
    spec = _dc_motor_spec(name)
    if spec is None:
      continue
    _, motor_id, gear, _, _ = spec
    return motor_id, gear
  return None


def _dc_motor_spec(joint_name: str) -> tuple | None:
  """Return the DC motor row whose joint pattern matches ``joint_name``."""
  for spec in _DC_MOTORS:
    if re.fullmatch(spec[0], joint_name):
      return spec
  return None


def _geared_motor(
  target_names_expr: tuple[str, ...],
  motor_id: str,
  external_gear_ratio: float,
  *,
  stiffness: float,
  damping: float,
) -> DcMotorActuatorCfg:
  """PD servo whose torque-speed curve includes the external gear.

  The peak envelope matches the old joint effort caps (peak torque times
  this gear). Wheel servos pass stiffness 0 so only the velocity gain acts.
  """
  curve = get_motor(motor_id).at_external_gear(
    external_gear_ratio,
    envelope="peak",
  )
  return DcMotorActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=stiffness,
    damping=damping,
    saturation_effort=curve.saturation_effort,
    velocity_limit=curve.velocity_limit,
    effort_limit=curve.effort_limit,
  )


WALTER_FS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=tuple(
    _geared_motor((expr,), motor_id, gear, stiffness=stiffness, damping=damping)
    for expr, motor_id, gear, stiffness, damping in _DC_MOTORS
  ),
  soft_joint_pos_limit_factor=0.9,
)
