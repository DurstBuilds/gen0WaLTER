"""CubeMars AK80-64 KV80 actuator datasheet.

Source: https://www.cubemars.com/product/ak80-64-kv80-robotic-actuator.html

This SKU is sold as KV80. The product page is not branded V2.
"""

from walter_rl.assets.motor_models.model import MotorModel

MOTOR = MotorModel(
  motor_id="ak80_64_kv80",
  manufacturer="CubeMars",
  model="AK80-64 KV80",
  source_url="https://www.cubemars.com/product/ak80-64-kv80-robotic-actuator.html",
  gear_ratio=64.0,
  rated_torque_nm=48.0,
  peak_torque_nm=120.0,
  rated_speed_rpm=(23.0, 48.0),
  no_load_speed_rpm=(37.0, 75.0),
  kv_rpm_per_v=80.0,
  kt_nm_per_a=0.136,
  ke_v_per_krpm=13.7,
  phase_resistance_ohm=220e-3,
  phase_inductance_h=133.5e-6,
  rotor_inertia_kg_m2=564.5e-7,
  rated_current_a=7.0,
  peak_current_a=19.0,
)
