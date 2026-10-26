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
    ASR_LANG,
    ASR_DECODER,
    ASR_MODEL_ID,
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
        required=False,
        default=None,
        help="Path to an audio file, folder of audio files, or .zip/.tar archive.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Start the interactive static dashboard web server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the dashboard server (default: 8000).",
    )
    parser.add_argument(
        "--asr-lang",
        default=ASR_LANG,
        help=f"Language code for Indic-Conformer ASR (default: {ASR_LANG}). E.g. 'ta', 'hi', 'en', 'te', 'kn'.",
    )
    parser.add_argument(
        "--asr-decoder",
        default=ASR_DECODER,
        choices=["ctc", "rnnt"],
        help=f"Decoding strategy for Indic-Conformer ASR (default: {ASR_DECODER}).",
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
        help=f"VRAM fraction for Whisper/ASR (default: {WHISPER_GPU_MEM}, optimized for 8GB VRAM).",
    )
    parser.add_argument(
        "--llm-gpu-mem",
        type=float,
        default=INTENT_GPU_MEM,
        help=f"VRAM fraction for Qwen LLM vLLM (default: {INTENT_GPU_MEM}, optimized for 8GB VRAM).",
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

    if args.dashboard:
        try:
            import dashboard_server
            dashboard_server.run_server(port=args.port)
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            import dashboard_server
            dashboard_server.run_server(port=args.port)
        return

    if not args.input:
        parser.error("--input / -i is required unless --dashboard is specified.")

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
            asr_lang=args.asr_lang,
            asr_decoder=args.asr_decoder,
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
            asr_lang=args.asr_lang,
            asr_decoder=args.asr_decoder,
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
            asr_lang=args.asr_lang,
            asr_decoder=args.asr_decoder,
        )


if __name__ == "__main__":
    main()
