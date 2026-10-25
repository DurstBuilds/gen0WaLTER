# Full-scale WaLTER

Checked-in MJCF for the full-scale hardware model (`walter_fs.xml`), taken from [WaLTER_Modeling](https://github.com/Optimal-Robotics-Lab/WaLTER_Modeling). Leg and wheel shapes are the CAD meshes in `assets/meshes/visual/`. The wheel mesh uses `assets/textures/wheel.png`. Torsos are still boxes. `walter_constants.py` loads the file and attaches the CubeMars actuators from [`motor_models`](../motor_models/README.md).

Each link mass is the hardware config mass, spread uniformly through that body's visual shape. MuJoCo integrates the closed meshes exactly. Contact bumps added in `walter_constants.py` have zero density. Joint `armature` is the same reflected rotor inertia as before, not body mass. Meshes and torso boxes do not collide. Wheels collide as ellipsoids, and the belly and legs collide as small spheres.

**+x is forward.** The root is the long rear **torso**; the short **head** box is ahead at the waist. Rear legs attach to the torso; front legs attach to the head. All four legs are identical.

## Mass

Total mass is **101.46 kg**.

| Group | Count | Each (kg) | Subtotal (kg) | Share |
| --- | --- | --- | --- | --- |
| Head (short box) | 1 | 18.79 | 18.79 | 18.5% |
| Torso (long box) | 1 | 25.91 | 25.91 | 25.5% |
| Thigh | 4 | 5.25 | 21.00 | 20.7% |
| Shin | 4 | 5.76 | 23.04 | 22.7% |
| Wheel | 8 | 1.59 | 12.72 | 12.5% |
| **Robot** | | | **101.46** | **100%** |

One leg (thigh, shin, and both wheels) is **14.19 kg**. Four legs are **56.76 kg**.

## Lengths

At zero, every hip and knee hangs so the wheel centers sit **0.180975 m** below the hip. With a wheel radius of 0.11 m, the torso origin is **0.290975 m** above the ground.

| Link | Size |
| --- | --- |
| Hip to knee | 0.180975 m down and 0.057673 m outboard |
| Torso box (root) | 0.46 × 0.30 × 0.18 m, centered on the root |
| Head box | 0.30 × 0.30 × 0.18 m, centered 0.4054 m forward of the torso origin |
| Waist | on the head body, 0.15 m aft of the head center (toward the torso) |
| Shin, axle to axle | 0.2274 m between the wheel centers |
| Wheel contact | ellipsoid radius 0.11 m, half-width 0.05 m, 0.04 m outboard of the axle |
| Wheel mesh | radius 0.109 m, about 0.08 m wide |

Rear hips on the torso are 0.108 m aft of the torso center. Front hips on the head are 0.027 m forward of the head center. Left and right hips are 0.30 m apart.

## Motors

Each leg uses the same three actuator types. Torque and speed already include the motor gearbox. The extra gear in `walter_constants.py` is applied on top. Curve details are in the [motor model README](../motor_models/README.md). The hardware generator's own gear motors are not used.

| Joint | Count per leg | Motor | Extra gear |
| --- | --- | --- | --- |
| Hip | 1 | CubeMars AK70-10 KV100 | 33 |
| Knee | 1 | CubeMars AK80-64 KV80 | 1.6 |
| Wheel | 2 | CubeMars AK10-9 V2.0 KV60 | 1 |

The robot therefore has four hip motors, four knee motors, and eight wheel motors. The waist is not one of these actuators. The PD environment holds it with a separate position servo.
