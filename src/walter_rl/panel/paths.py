"""Locations the panel reads and the processes it starts."""

from __future__ import annotations

from pathlib import Path

_CHECKPOINT_LOG_ROOTS = (
  Path("research/logs/training"),
  Path("logs/rsl_rl"),
)


def repo_root() -> Path:
  """Return the gen0WaLTER checkout that contains this package."""
  return Path(__file__).resolve().parents[3]


def checkpoint_log_roots(root: Path) -> tuple[Path, ...]:
  """Directories that hold ``<experiment>/<timestamp>/model_*.pt`` runs."""
  return tuple((root / rel).resolve() for rel in _CHECKPOINT_LOG_ROOTS)


def is_checkpoint_path(root: Path, path: Path) -> bool:
  """Return whether ``path`` lies under a known training log root."""
  resolved = path.resolve()
  for logs in checkpoint_log_roots(root):
    try:
      resolved.relative_to(logs)
      return True
    except ValueError:
      continue
  return False
