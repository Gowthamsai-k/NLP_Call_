"""
Customer Support Call Intelligence Pipeline Package.
"""

from .orchestration.pipeline import process_single_call
from .orchestration.batch_runner import process_audio_folder, unpack_archive

__version__ = "1.0.0"
__all__ = ["process_single_call", "process_audio_folder", "unpack_archive"]
