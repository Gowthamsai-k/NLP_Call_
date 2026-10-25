"""
CLI entry point for Customer Support Pipeline with GPU memory controls.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Compatibility shims for modern torch/torchaudio/numpy
from ..core import shims  # noqa: F401

from ..core.config import (
    DEFAULT_STAGE_RETRIES,
    RETRY_BACKOFF_FACTOR,
    WHISPER_GPU_MEM,
    INTENT_GPU_MEM,
    EVAL_GPU_MEM,
)
from ..orchestration.pipeline import process_single_call
from ..orchestration.batch_runner import process_audio_folder, unpack_archive


def setup_logging(level: str = "INFO"):
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        prog="csp-run",
        description="Customer Support Call Pipeline (Diarization + Whisper ASR + Punctuation + Intent + Evaluation)"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to an audio file, folder of audio files, or .zip/.tar archive.",
    )
    parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory where deliverables and logs will be saved.",
    )
    parser.add_argument(
        "--hf-token",
        help="Hugging Face User Access Token (or set HF_TOKEN environment variable).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_STAGE_RETRIES,
        help=f"Number of retries per pipeline stage (default: {DEFAULT_STAGE_RETRIES}).",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=RETRY_BACKOFF_FACTOR,
        help=f"Exponential backoff factor for retries (default: {RETRY_BACKOFF_FACTOR}).",
    )
    parser.add_argument(
        "--whisper-gpu-mem",
        type=float,
        default=WHISPER_GPU_MEM,
        help=f"VRAM fraction for Whisper vLLM (default: {WHISPER_GPU_MEM}). Use 0.25-0.35 on 15GB T4.",
    )
    parser.add_argument(
        "--llm-gpu-mem",
        type=float,
        default=INTENT_GPU_MEM,
        help=f"VRAM fraction for Qwen LLM vLLM (default: {INTENT_GPU_MEM}). Use 0.50-0.60 on 15GB T4.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run all stages even if deliverables or checkpoints already exist.",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Do not delete intermediate audio and transcript files after completion.",
    )
    parser.add_argument(
        "--no-reuse-llm",
        action="store_true",
        help="Release Qwen LLM engine between intent extraction and evaluation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("main")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    resume = not args.no_resume
    clean_intermediate = not args.keep_intermediates
    reuse_llm = not args.no_reuse_llm

    if input_path.is_file() and input_path.suffix.lower() in (".zip", ".tar", ".gz", ".tgz"):
        extracted_dir = unpack_archive(str(input_path))
        logger.info(f"Processing unpacked archive from {extracted_dir}...")
        process_audio_folder(
            input_folder=str(extracted_dir),
            output_base_dir=args.output,
            hf_token=hf_token,
            max_retries=args.retries,
            backoff_factor=args.backoff,
            resume=resume,
            clean_intermediate=clean_intermediate,
            reuse_llm=reuse_llm,
        )
    elif input_path.is_dir():
        logger.info(f"Processing folder of audio files: {input_path}...")
        process_audio_folder(
            input_folder=str(input_path),
            output_base_dir=args.output,
            hf_token=hf_token,
            max_retries=args.retries,
            backoff_factor=args.backoff,
            resume=resume,
            clean_intermediate=clean_intermediate,
            reuse_llm=reuse_llm,
        )
    elif input_path.is_file():
        logger.info(f"Processing single audio file: {input_path}...")
        process_single_call(
            input_audio_path=str(input_path),
            output_dir=args.output,
            hf_token=hf_token,
            max_retries=args.retries,
            backoff_factor=args.backoff,
            resume=resume,
            clean_intermediate=clean_intermediate,
            reuse_llm=reuse_llm,
            whisper_gpu_mem=args.whisper_gpu_mem,
            intent_gpu_mem=args.llm_gpu_mem,
            eval_gpu_mem=args.llm_gpu_mem,
        )


if __name__ == "__main__":
    main()
