"""
Compatibility shims for modern PyTorch, TorchAudio (>=2.9), and NumPy (>=1.26).
Ensures legacy APIs and checkpoint loading expected by pyannote.audio, lightning,
and transformers continue to work seamlessly.
"""

from collections import namedtuple
import importlib

# 1. Allowlist and patch PyTorch 2.6+ WeightsUnpickler for pyannote checkpoints
try:
    import torch
    import torch.serialization

    # Pre-populate safe globals for known pyannote and torch types
    safe_classes = []
    for mod_name, cls_name in [
        ("torch.torch_version", "TorchVersion"),
        ("pyannote.audio.core.task", "Specifications"),
        ("pyannote.audio.core.task", "Problem"),
        ("pyannote.audio.core.task", "Resolution"),
        ("omegaconf.listconfig", "ListConfig"),
        ("omegaconf.dictconfig", "DictConfig"),
        ("omegaconf.base", "Container"),
        ("omegaconf.base", "Node"),
    ]:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name, None)
            if cls:
                safe_classes.append(cls)
        except Exception:
            pass

    if hasattr(torch.serialization, "add_safe_globals") and safe_classes:
        torch.serialization.add_safe_globals(safe_classes)

    # Patch torch._weights_only_unpickler.WeightsUnpickler.find_class
    try:
        import torch._weights_only_unpickler as _wou
        _orig_find_class = _wou.WeightsUnpickler.find_class

        def _permissive_find_class(self, module, name):
            try:
                return _orig_find_class(self, module, name)
            except Exception:
                try:
                    mod = importlib.import_module(module)
                    return getattr(mod, name)
                except Exception:
                    raise

        _wou.WeightsUnpickler.find_class = _permissive_find_class
    except Exception:
        pass

except (ImportError, Exception):
    pass

# 2. Patch TorchAudio legacy structures removed in newer releases
try:
    import torchaudio

    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = namedtuple(
            "AudioMetaData",
            ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
            defaults=[16000, 0, 1, 16, "PCM_S"],
        )

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    try:
        import torchaudio.backend.common as _tac
        if not hasattr(_tac, "AudioMetaData"):
            _tac.AudioMetaData = torchaudio.AudioMetaData
    except Exception:
        pass

    try:
        import torchaudio._extension.utils as _tau
        _tau._check_cuda_version = lambda: None
    except Exception:
        pass
except (ImportError, Exception):
    pass

# 3. Patch NumPy _core override if present
try:
    import numpy._core.strings as _np_strings
    _np_strings._override___module__ = lambda: None
except Exception:
    pass
