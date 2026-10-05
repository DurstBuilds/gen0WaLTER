---
name: generate-motor-model
description: Look up actuator datasheets and add output-side torque-speed assets under src/walter_rl/assets/motor_models. Use when the user invokes generate-motor-model with one motor or a list of motors, or asks for rated and peak torque, a torque-speed curve, dc_motor_params, or a joint role assigned to a motor id.
disable-model-invocation: true
---

# Generate motor model

Add one folder per named actuator and report its torque-speed curve. The curve math already lives on `MotorModel`. Do not reimplement it and do not edit `src/walter_rl/assets/walter_fs/walter_constants.py`.

## Input

Accept one motor or a list, plus optional role assignments. Examples:

- `AK70-10 KV100`
- `AK10-9 V2.0 KV60, AK80-64 KV80`
- `hip -> ak70_10_kv100`

A role assignment only appears in the reply as an `assign({...})` example. Do not write a default joint map.

## Workflow

1. Turn each requested name into a `motor_id`: lowercase, digits and underscores only. `AK70-10 KV100` becomes `ak70_10_kv100`. `AK10-9 V2.0 KV60` becomes `ak10_9_v2_kv60`. The folder name and `MOTOR.motor_id` must match.
2. If that id is already registered, do not rewrite it. Load it and report the curve.
3. Otherwise search the manufacturer's official product page. Record the source URL.
4. Stop and ask the user when a name is not a real SKU, several SKUs match, or any required field below is missing. Name the missing fields. Do not substitute a nearby model, halve 48 V speeds to invent 24 V speeds, or fill constants from a different voltage column.
5. Add `src/walter_rl/assets/motor_models/<motor_id>/__init__.py` exporting `MOTOR`. The package discovers folders that export `MOTOR`; do not edit the registry.
6. Run the check below, then reply with the curve outcome.

## Datasheet

Torque and rpm are gearbox output. Do not multiply them by `gear_ratio`. KV, Kt, Ke, phase resistance, phase inductance, and rotor inertia are motor-side.

Store both 24 V and 48 V output speeds. The reported curve uses 48 V. `at_voltage(24)` selects the other column. Supported bus voltages are only 24 and 48.

Required before writing a model:

- gear ratio
- rated torque (N·m) and peak torque (N·m)
- rated speed and no-load speed at 24 V and at 48 V (rpm)
- KV (rpm/V), Kt (N·m/A), Ke (V/krpm)
- phase-to-phase resistance and inductance
- rotor inertia
- rated current and peak current (A)

Write SI in the constructor, keeping the datasheet figure visible:

- milliohms: `272e-3`
- microhenries: `113e-6`
- g·cm²: `753.4788e-7` (multiply by `1e-7` for kg·m²)

Use the published no-load speed as max speed. Ignore a marketing max-speed line that disagrees with it.

If the spec table and a summary row disagree, keep the detailed specification and mention the conflict in the module docstring only when it changes a stored number.

## File

```python
"""Manufacturer MODEL actuator datasheet.

Source: https://example.com/product
"""

from walter_rl.assets.motor_models.model import MotorModel

MOTOR = MotorModel(
  motor_id="ak70_10_kv100",
  manufacturer="CubeMars",
  model="AK70-10 KV100",
  source_url="https://example.com/product",
  gear_ratio=10.0,
  rated_torque_nm=8.3,
  peak_torque_nm=24.8,
  rated_speed_rpm=(148.0, 310.0),
  no_load_speed_rpm=(240.0, 480.0),
  kv_rpm_per_v=100.0,
  kt_nm_per_a=0.123,
  ke_v_per_krpm=11.2,
  phase_resistance_ohm=272e-3,
  phase_inductance_h=113e-6,
  rotor_inertia_kg_m2=753.4788e-7,
  rated_current_a=7.2,
  peak_current_a=23.2,
)
```

When published peak torque is above the 48 V stall intercept, say so in the module docstring. Still store the published peak. See `ak70_10_kv100`.

## Curve

`MotorModel` builds one voltage line through `(rated_speed, rated_torque)` and `(no_load_speed, 0)`:

```text
stall_intercept = rated_torque * no_load / (no_load - rated_speed)
```

- Continuous: rated torque until rated speed, then the line to 0 at no-load speed.
- Peak: that same line, capped at the published peak. The cap binds only when peak is below the intercept.

`dc_motor_params()` feeds mjlab `dc_motor_clip`: `saturation_effort` is the stall intercept, `velocity_limit` is no-load speed in rad/s, `effort_limit` is rated torque for `"continuous"` and peak torque for `"peak"`.

## Check

After adding a motor, run from the repo root:

```bash
uv run python -c "
from walter_rl.assets.motor_models import get_motor
m = get_motor('MOTOR_ID')
no_load, rated = m.no_load_speed_rpm[1], m.rated_speed_rpm[1]
expected = m.rated_torque_nm * no_load / (no_load - rated)
assert abs(m.stall_intercept_nm() - expected) < 1e-9
assert m.torque_limit(0.0) == m.rated_torque_nm
assert abs(m.torque_limit(m.rated_speed_rad_s()) - m.rated_torque_nm) < 1e-6
assert m.torque_limit(m.no_load_speed_rad_s()) == 0.0
print(m.motor_id, round(m.stall_intercept_nm(), 3), m.dc_motor_params())
"
```

Repeat per new id. Fix the asset if the check fails.

## Reply

For each motor, report:

- `motor_id`, model name, source URL
- rated torque, peak torque
- 48 V rated speed and no-load speed
- stall intercept (N·m)
- whether the peak cap binds; if it does not, give the peak-envelope stall torque
- `dc_motor_params()` for the continuous envelope

Then show:

```python
from walter_rl.assets.motor_models import assign, get_motor

motor = get_motor("ak70_10_kv100")
motor.torque_limit(speed_rad_s)
motor.torque_limit(speed_rad_s, envelope="peak")
motor.dc_motor_params()
motor.at_voltage(24)

assign({"hip": "ak70_10_kv100"})
```

Use the ids from this run. Include `assign` only for roles the user named.
