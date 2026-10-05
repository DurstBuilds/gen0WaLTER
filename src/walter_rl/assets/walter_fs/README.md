# Full-scale WaLTER

Checked-in MJCF for the full-scale robot (`walter_fs.xml`). `walter_constants.py` loads it, strips the generator motors, and attaches the CubeMars actuators from [`motor_models`](../motor_models/README.md).

Masses below are the geom `mass` values in the MJCF. Contact bumps added in `walter_constants.py` have zero density. Joint `armature` is reflected rotor inertia, not body mass.

All four legs are identical. The head is the front body; the torso carries the rear legs.

## Mass

Total mass is **90.3 kg**.

| Group | Count | Each (kg) | Subtotal (kg) | Share |
| --- | --- | --- | --- | --- |
| Torso | 1 | 36 | 36.0 | 39.9% |
| Head | 1 | 18 | 18.0 | 19.9% |
| Thigh | 4 | 3.375 | 13.5 | 15.0% |
| Shin | 4 | 1.2 | 4.8 | 5.3% |
| Wheel | 8 | 2.25 | 18.0 | 19.9% |
| **Robot** | | | **90.3** | **100%** |

One leg (thigh, shin, and both wheels) is **9.075 kg**, 10.0% of the robot. Four legs are **36.3 kg**, 40.2%.

| Leg part | Mass (kg) | Share of the leg |
| --- | --- | --- |
| Thigh | 3.375 | 37.2% |
| Shin | 1.2 | 13.2% |
| Wheels (2 × 2.25) | 4.5 | 49.6% |
| **Leg** | **9.075** | **100%** |

## Lengths

At zero, every hip and knee hangs so the wheel centers sit **0.18 m** below the hip. With a wheel radius of 0.10 m, the torso origin is **0.28 m** above the ground.

| Link | Size |
| --- | --- |
| Hip to knee | 0.18 m down and 0.12 m outboard (0.216 m straight) |
| Thigh capsule | radius 0.04 m, cylinder length 0.18 m |
| Shin, axle to axle | 0.216 m between the front and rear wheel centers |
| Shin capsule | radius 0.04 m, cylinder length 0.216 m along that axle line |
| Wheel | diameter 0.20 m (radius 0.10 m), width 0.10 m |

The torso box is 0.46 × 0.216 × 0.18 m. The head box is 0.30 × 0.216 × 0.18 m. Hips are ±0.108 m from the body centerline. Left and right wheel centers are 0.636 m apart.

## Motors

Each leg uses the same three actuator types. Torque and speed already include the motor gearbox. The extra gear in `walter_constants.py` is applied on top. Curve details are in the [motor model README](../motor_models/README.md).

| Joint | Count per leg | Motor | Extra gear |
| --- | --- | --- | --- |
| Hip | 1 | CubeMars AK70-10 KV100 | 33 |
| Knee | 1 | CubeMars AK80-64 KV80 | 1.6 |
| Wheel | 2 | CubeMars AK10-9 V2.0 KV60 | 1 |

The robot therefore has four hip motors, four knee motors, and eight wheel motors. The waist (`head_joint`) is not one of these actuators. The PD environment holds it with a separate position servo.
