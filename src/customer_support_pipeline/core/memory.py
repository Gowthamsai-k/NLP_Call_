"""
Memory management and cleanup utilities to prevent GPU and host RAM runtime exhaustion.
Features proactive memory purging, vLLM distributed environment teardown,
and dynamic GPU VRAM backoff on OOM errors.
"""

import os
import gc
import sys
import time
import logging
from typing import Any, Callable
import psutil

from .config import GPU_VRAM_BUDGET_GB, VLLM_MAX_GPU_FRACTION

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)

# Configure PyTorch memory allocator to avoid fragmentation in Colab
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def require_cuda_device():
    """Fail early when the inference pipeline cannot use a CUDA GPU."""
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError(
            "GPU inference was requested, but CUDA is unavailable. "
            "Install a CUDA-enabled PyTorch/vLLM build and use a supported NVIDIA GPU."
        )


def bounded_gpu_memory_utilization(requested: float) -> float:
    """Scale vLLM allocation to device/free VRAM, honoring an optional hard cap."""
    require_cuda_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    gib = 1024**3
    limits = [
        VLLM_MAX_GPU_FRACTION * total_bytes,
        max(0, free_bytes - (0.5 * gib)),
    ]
    if GPU_VRAM_BUDGET_GB > 0:
        limits.append(GPU_VRAM_BUDGET_GB * gib)
    budget_bytes = min(limits)
    effective = min(float(requested), budget_bytes / total_bytes)
    if effective < 0.20:
        raise RuntimeError(
            f"Only {free_bytes / gib:.2f} GiB of GPU memory is free; "
            "free VRAM or close other GPU workloads before running inference."
        )
    if effective < requested:
        logger.info(
            "Capping vLLM GPU memory utilization from %.2f to %.2f "
            "(configured ceiling: %s; %.2f GiB currently free).",
            requested,
            effective,
            f"{GPU_VRAM_BUDGET_GB:.1f} GiB" if GPU_VRAM_BUDGET_GB > 0 else "dynamic",
            free_bytes / gib,
        )
    return effective


def wait_for_free_vram(
    stage_name: str,
    timeout_seconds: float | None = None,
    poll_seconds: float = 2.0,
) -> tuple[float, float]:
    """Wait until a model stage's minimum vLLM VRAM budget is available.

    The minimum mirrors bounded_gpu_memory_utilization(): at least 20% of total
    VRAM must remain after reserving 0.5 GiB for CUDA/runtime allocations.
    This is a readiness barrier, not a way to reclaim memory owned by a live
    model or another process.
    """
    require_cuda_device()
    if timeout_seconds is None:
        timeout_seconds = float(os.environ.get("GPU_VRAM_WAIT_TIMEOUT_SECONDS", "60"))

    gib = 1024**3
    started_at = time.monotonic()
    last_report_at = 0.0

    while True:
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        required_bytes = (0.20 * total_bytes) + (0.5 * gib)
        if free_bytes >= required_bytes:
            logger.info(
                "[%s] VRAM ready: %.2f GiB free of %.2f GiB.",
                stage_name,
                free_bytes / gib,
                total_bytes / gib,
            )
            return free_bytes / gib, total_bytes / gib

        elapsed = time.monotonic() - started_at
        if elapsed >= timeout_seconds:
            raise RuntimeError(
                f"[{stage_name}] VRAM did not become available after "
                f"{timeout_seconds:.0f}s: {free_bytes / gib:.2f} GiB free of "
                f"{total_bytes / gib:.2f} GiB; at least "
                f"{required_bytes / gib:.2f} GiB is required to start vLLM. "
                "The previous stage may still hold GPU memory. Check the "
                "'After Diarization Cleanup' VRAM log and retry after cleanup."
            )

        if elapsed - last_report_at >= 10.0:
            logger.warning(
                "[%s] Waiting for VRAM release: %.2f GiB free; need at least "
                "%.2f GiB. Waiting up to %.0fs total.",
                stage_name,
                free_bytes / gib,
                required_bytes / gib,
                timeout_seconds,
            )
            last_report_at = elapsed
        time.sleep(min(poll_seconds, max(0.0, timeout_seconds - elapsed)))


def get_vram_usage() -> dict:
    """Returns current GPU VRAM and system memory usage statistics in GiB."""
    stats = {
        "system_ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "system_ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
    }
    if torch is not None and torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        stats.update({
            "gpu_allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 2),
            "gpu_reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 2),
            "gpu_max_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
            "gpu_device_used_gb": round((total_bytes - free_bytes) / (1024**3), 2),
            "gpu_device_free_gb": round(free_bytes / (1024**3), 2),
            "gpu_device_total_gb": round(total_bytes / (1024**3), 2),
        })
    return stats


def log_vram(prefix: str = "Memory Status"):
    """Logs current memory utilization."""
    stats = get_vram_usage()
    if "gpu_allocated_gb" in stats:
        logger.info(
            f"[{prefix}] GPU Allocated: {stats['gpu_allocated_gb']} GB | "
            f"Reserved: {stats['gpu_reserved_gb']} GB | "
            f"Device Used: {stats['gpu_device_used_gb']}/{stats['gpu_device_total_gb']} GB | "
            f"Device Free: {stats['gpu_device_free_gb']} GB | "
            f"RAM Avail: {stats['system_ram_available_gb']} GB"
        )
    else:
        logger.info(f"[{prefix}] RAM Avail: {stats['system_ram_available_gb']} GB")


def cleanup_vllm_distributed():
    """
    Destroys vLLM's internal distributed environment and process groups.
    Without this, vLLM holds onto GPU memory buffers even after deleting the LLM object.
    """
    # 1. Destroy vLLM model parallel & distributed environment
    for mod_name in (
        "vllm.distributed",
        "vllm.distributed.parallel_state",
        "vllm.model_executor.parallel_state",
    ):
        try:
            mod = __import__(mod_name, fromlist=["destroy_model_parallel", "destroy_distributed_environment"])
            if hasattr(mod, "destroy_model_parallel"):
                mod.destroy_model_parallel()
            if hasattr(mod, "destroy_distributed_environment"):
                mod.destroy_distributed_environment()
        except Exception:
            pass

    # 2. Destroy PyTorch distributed process group if active
    if torch is not None and hasattr(torch, "distributed") and torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            pass


def release_memory(*objects: Any, delay: float = 0.5):
    """
    Comprehensively frees memory occupied by model instances, tensors, and cache.
    1. Invokes .shutdown() on objects (e.g., vLLM engines) if available.
    2. Destroys vLLM distributed process groups and communicators.
    3. Runs Python garbage collection.
    4. Clears CUDA cache, synchronizes GPU streams, and runs IPC collect.
    """
    for obj in objects:
        if obj is None:
            continue
        for shutdown_method in ("shutdown", "close", "stop"):
            if hasattr(obj, shutdown_method) and callable(getattr(obj, shutdown_method)):
                try:
                    getattr(obj, shutdown_method)()
                except Exception as e:
                    logger.debug(f"Error during {shutdown_method} call: {e}")

    # Destroy vLLM communicators/process groups
    cleanup_vllm_distributed()

    del objects
    gc.collect()

    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"Error while clearing CUDA cache: {e}")

    if delay > 0:
        time.sleep(delay)

    gc.collect()


def retry_with_memory_cleanup(
    fn: Callable,
    *args,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    stage_name: str = "operation",
    **kwargs,
):
    """
    Executes a function with retry logic and dynamic memory mitigation.
    If an OutOfMemoryError occurs, it automatically dials down `gpu_memory_utilization`
    (if the function supports it), completely purges GPU cache, and retries.
    """
    last_exception = None
    delay = 1.0

    current_kwargs = dict(kwargs)

    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **current_kwargs)
        except Exception as exc:
            last_exception = exc
            exc_str = str(exc).lower()
            is_oom = (
                "out of memory" in exc_str
                or "cuda out of memory" in exc_str
                or "oom" in exc_str
                or "less than desired gpu memory utilization" in exc_str
                or isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", tuple()))
                if torch is not None
                else False
            )

            logger.error(
                f"[{stage_name}] Attempt {attempt}/{max_retries} failed "
                f"(Is OOM: {is_oom}): {type(exc).__name__}: {exc}"
            )

            # Deep GPU cache & distributed group purge
            release_memory(delay=1.0)

            # If out of memory, automatically back off gpu_memory_utilization
            if is_oom and "gpu_memory_utilization" in current_kwargs:
                old_util = current_kwargs["gpu_memory_utilization"]
                new_util = max(0.20, round(old_util - 0.12, 2))
                logger.warning(
                    f"[{stage_name}] Memory pressure detected! Automatically reducing GPU memory utilization from {old_util} to {new_util}"
                )
                current_kwargs["gpu_memory_utilization"] = new_util

            if attempt < max_retries:
                logger.info(f"[{stage_name}] Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
                delay *= backoff_factor

    raise RuntimeError(
        f"[{stage_name}] Failed after {max_retries} attempts: {last_exception}"
    ) from last_exception
