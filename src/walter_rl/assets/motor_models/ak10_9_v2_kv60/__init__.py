"""CubeMars AK10-9 V2.0 KV60 actuator datasheet.

Source: https://www.cubemars.com/product/ak10-9-v2-0-kv60-robotic-actuator.html

Max speed is the published no-load speed. The page's 26 rad/s marketing line is
not used.
"""

from walter_rl.assets.motor_models.model import MotorModel

MOTOR = MotorModel(
  motor_id="ak10_9_v2_kv60",
  manufacturer="CubeMars",
  model="AK10-9 V2.0 KV60",
  source_url="https://www.cubemars.com/product/ak10-9-v2-0-kv60-robotic-actuator.html",
  gear_ratio=9.0,
  rated_torque_nm=18.0,
  peak_torque_nm=48.0,
  rated_speed_rpm=(109.0, 228.0),
  no_load_speed_rpm=(160.0, 320.0),
  kv_rpm_per_v=60.0,
  kt_nm_per_a=0.198,
  ke_v_per_krpm=17.2,
  phase_resistance_ohm=195e-3,
  phase_inductance_h=181e-6,
  rotor_inertia_kg_m2=1002e-7,
  rated_current_a=10.6,
  peak_current_a=29.8,
)
