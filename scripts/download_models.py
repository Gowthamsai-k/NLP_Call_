"""Pre-download every Hugging Face model used by the pipeline."""

import os
import sys

from huggingface_hub import snapshot_download

from customer_support_pipeline.core.config import (
    DIARIZATION_MODEL_ID,
    LLM_MODEL_ID,
    PUNCTUATION_MODEL_ID,
    WHISPER_MODEL_ID,
)


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "HF_TOKEN is required to download the gated Pyannote model. "
            "Accept its Hugging Face terms and set HF_TOKEN, then rerun.",
            file=sys.stderr,
        )
        return 2

    model_ids = (
        DIARIZATION_MODEL_ID,
        WHISPER_MODEL_ID,
        PUNCTUATION_MODEL_ID,
        LLM_MODEL_ID,
    )
    for model_id in model_ids:
        print(f"\nDownloading/checking {model_id} ...", flush=True)
        try:
            path = snapshot_download(repo_id=model_id, token=token)
        except Exception as exc:
            print(
                f"Failed to download {model_id}: {exc}\n"
                "Check network access, HF_TOKEN, and whether you have accepted "
                "the model's access terms on Hugging Face.",
                file=sys.stderr,
            )
            return 1
        print(f"Ready in Hugging Face cache: {path}", flush=True)

    print("\nAll pipeline models are cached and ready for runtime.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
