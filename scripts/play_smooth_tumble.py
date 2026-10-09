"""Play the Walter smooth-tumble controller in Viser or the native viewer.

Examples:
  python scripts/play_smooth_tumble.py --task flat
  python scripts/play_smooth_tumble.py --task rough --viewer viser
  python scripts/play_smooth_tumble.py --task rough-test --viewer viser
  python scripts/play_smooth_tumble.py --task flat --payload rider
"""

from __future__ import annotations

import argparse

from mjlab.viewer import NativeMujocoViewer

from walter_rl.viewer import WalterViserPlayViewer

from walter_rl.assets.payloads import available_payloads
from walter_rl.control.full_body_pd.env import TaskName, make_pd_env
from walter_rl.control.smooth_tumble import SmoothTumblePolicy

_TASKS = ("flat", "rough", "rough-test", "stairs")


def main() -> None:
  """Launch a play environment and run the smooth-tumble policy."""
  args = _parse_args()
  task: TaskName = args.task
  env = make_pd_env(
    task,
    play=True,
    num_envs=args.num_envs,
    payload=args.payload,
  )
  policy = SmoothTumblePolicy(env)
  viewer = WalterViserPlayViewer if args.viewer == "viser" else NativeMujocoViewer
  try:
    viewer(env, policy).run()
  finally:
    env.close()


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=_TASKS, default="flat")
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--viewer", choices=("viser", "native"), default="viser")
  parser.add_argument(
    "--payload",
    choices=available_payloads(),
    default=None,
    help="Rigid mass welded to the top of the robot.",
  )
  return parser.parse_args()


if __name__ == "__main__":
  main()
