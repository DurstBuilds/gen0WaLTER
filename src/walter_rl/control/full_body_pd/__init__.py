"""Full-body PD controller for the Walter robot."""

from walter_rl.control.full_body_pd.gains import FullBodyPdGains, default_gains
from walter_rl.control.full_body_pd.policy import FullBodyPdPolicy

__all__ = ["FullBodyPdGains", "FullBodyPdPolicy", "default_gains"]
