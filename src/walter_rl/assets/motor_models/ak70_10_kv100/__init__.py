"""CubeMars AK70-10 KV100 actuator datasheet.

Source: https://www.cubemars.com/product/ak70-10-kv100-robotic-actuator.html

The published peak of 24.8 N·m sits slightly above the 48 V voltage line implied
by 8.3 N·m at 310 rpm and zero torque at 480 rpm. The peak envelope therefore
follows that line (about 23.4 N·m at stall) and still stores 24.8 N·m.
"""

from walter_rl.assets.motor_models.model import MotorModel

MOTOR = MotorModel(
  motor_id="ak70_10_kv100",
  manufacturer="CubeMars",
  model="AK70-10 KV100",
  source_url="https://www.cubemars.com/product/ak70-10-kv100-robotic-actuator.html",
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
