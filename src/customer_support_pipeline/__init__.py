"""
Customer Support Call Intelligence Pipeline Package.
"""

# Apply compatibility shims before importing submodules
from .core import shims  # noqa: F401

__version__ = "1.0.0"
__all__ = ["process_single_call", "process_audio_folder", "unpack_archive"]


def __getattr__(name: str):
    if name == "process_single_call":
        from .orchestration.pipeline import process_single_call
        return process_single_call
    if name in ("process_audio_folder", "unpack_archive"):
        from .orchestration import batch_runner
        return getattr(batch_runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
