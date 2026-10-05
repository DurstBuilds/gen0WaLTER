# Motor models

Datasheets and output-side torque-speed curves for actuators used by WaLTER and later robots. Each motor is a folder that exports `MOTOR`. The package loads every such folder by id.

Torque and speed ratings are **gearbox output**. KV, Kt, Ke, phase resistance, phase inductance, and rotor inertia are **motor-side**. The curve does not multiply output torque by the motor's internal gear ratio again.

Curves use the **48 V** column. Both 24 V and 48 V speeds are stored. `at_voltage(24)` selects the other column.

## Catalog

| Id | Model | Internal gear | Rated torque | Peak torque | Rated speed (24/48 V) | No-load speed (24/48 V) |
| --- | --- | --- | --- | --- | --- | --- |
| `ak70_10_kv100` | [AK70-10 KV100](https://www.cubemars.com/product/ak70-10-kv100-robotic-actuator.html) | 10:1 | 8.3 N·m | 24.8 N·m | 148 / 310 rpm | 240 / 480 rpm |
| `ak10_9_v2_kv60` | [AK10-9 V2.0 KV60](https://www.cubemars.com/product/ak10-9-v2-0-kv60-robotic-actuator.html) | 9:1 | 18 N·m | 48 N·m | 109 / 228 rpm | 160 / 320 rpm |
| `ak80_64_kv80` | [AK80-64 KV80](https://www.cubemars.com/product/ak80-64-kv80-robotic-actuator.html) | 64:1 | 48 N·m | 120 N·m | 23 / 48 rpm | 37 / 75 rpm |

AK70-10's published 24.8 N·m peak sits slightly above the 48 V voltage line (about 23.4 N·m at stall). The peak envelope follows that line. The published peak is still stored on the model.

## Lookup

```python
from walter_rl.assets.motor_models import assign, get_motor

hip = get_motor("ak70_10_kv100")
hip.torque_limit(speed_rad_s)                  # continuous, 48 V
hip.torque_limit(speed_rad_s, envelope="peak")
hip.dc_motor_params()                          # saturation, velocity_limit, effort_limit
hip.at_voltage(24)

assign({"hip": "ak70_10_kv100", "knee": "ak80_64_kv80"})
```

`get_motor` raises `KeyError` for an unknown id.

## Torque-speed curve

One voltage line passes through `(rated_speed, rated_torque)` and `(no_load_speed, 0)`. Its stall intercept is:

```text
rated_torque * no_load / (no_load - rated_speed)
```

- **Continuous:** rated torque until rated speed, then that line down to 0 at no-load speed.
- **Peak:** the same line, capped at the published peak torque. The cap binds only when peak is below the intercept.

`dc_motor_params()` feeds mjlab's `DcMotorActuatorCfg`:

| Field | Meaning |
| --- | --- |
| `saturation_effort` | Stall intercept of the voltage line (N·m) |
| `velocity_limit` | No-load output speed (rad/s) |
| `effort_limit` | Rated torque for `"continuous"`, peak torque for `"peak"` |

## Extra gear

`at_external_gear(ratio)` applies a reduction **after** the actuator. Datasheet torque and speed already include the internal gearbox.

- Joint torque = actuator torque × `ratio`
- Joint speed = actuator speed / `ratio`

WaLTER does this in `walter_fs/walter_constants.py` with the peak envelope:

| Joint | Motor | Extra gear |
| --- | --- | --- |
| Hip | `ak70_10_kv100` | 33 |
| Knee | `ak80_64_kv80` | 1.6 |
| Wheel | `ak10_9_v2_kv60` | 1 |

## Add a motor

1. Create `src/walter_rl/assets/motor_models/<motor_id>/__init__.py`.
2. Export `MOTOR = MotorModel(...)`. `motor_id` must match the folder name.
3. Fill both voltage columns from the official datasheet. Do not halve 48 V speeds to invent 24 V speeds.
4. Store SI values with the datasheet figure visible: `272e-3` ohm, `113e-6` H, `753.4788e-7` kg·m² for 753.4788 g·cm².

The registry picks up the new folder on import. No edit to `__init__.py` is required.
