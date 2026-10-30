#!/usr/bin/env bash
set -e

echo "================================================================="
echo "  Google Colab Standalone Production Packager"
echo "  Customer Support Call Evaluation Pipeline"
echo "================================================================="

# Ensure working directory is project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH=".:${PYTHONPATH:-}"
export UV_HTTP_TIMEOUT=21600

echo ""
echo "[1/6] Installing OS audio and build libraries (ffmpeg, patchelf)..."
sudo -n apt-get update -qq 2>/dev/null || true
sudo -n apt-get install -y -qq ffmpeg patchelf libsndfile1 libsecret-1-0 gir1.2-secret-1 2>/dev/null || true

echo ""
echo "[2/6] Detecting and preparing high-speed installer (uv)..."
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv package manager for 10x faster installation..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

# Determine whether we can write to system python or should activate local .build_venv
if [ "$EUID" -ne 0 ] && [ ! -w "/usr/local/lib/python3.13" ]; then
  if [ ! -d ".build_venv" ]; then
    echo "Creating user build environment (.build_venv)..."
    uv venv .build_venv
  fi
  source .build_venv/bin/activate
  INSTALL_CMD="uv pip install --index-strategy unsafe-best-match"
  VENDOR_CMD="uv pip install --index-strategy unsafe-best-match --target"
else
  INSTALL_CMD="uv pip install --system --index-strategy unsafe-best-match"
  VENDOR_CMD="uv pip install --index-strategy unsafe-best-match --target"
fi

$INSTALL_CMD nuitka patchelf

echo ""
echo "[3/6] Installing full ML pipeline dependencies in environment..."
$INSTALL_CMD -r requirements.txt sentencepiece soundfile pyyaml psutil pysqlite3-binary

echo ""
echo "[4/6] Vendoring all runtime packages into offline bundle..."
rm -rf colab_dist
mkdir -p colab_dist/vendor_packages
$VENDOR_CMD colab_dist/vendor_packages -r requirements.txt sentencepiece soundfile pyyaml psutil pysqlite3-binary --no-compile

echo ""
echo "[5/6] Compiling dashboard_server with Nuitka standalone engine..."
python3 -m nuitka \
  --standalone \
  --disable-plugin=gi \
  --nofollow-import-to=gi \
  --nofollow-import-to=keyring \
  --nofollow-import-to=numpy \
  --nofollow-import-to=scipy \
  --nofollow-import-to=numba \
  --nofollow-import-to=IPython \
  --nofollow-import-to=jedi \
  --nofollow-import-to=traitlets \
  --nofollow-import-to=pygments \
  --nofollow-import-to=matplotlib \
  --nofollow-import-to=jupyter_server \
  --nofollow-import-to=jupyter_client \
  --nofollow-import-to=jupyter_core \
  --nofollow-import-to=notebook \
  --nofollow-import-to=tornado \
  --nofollow-import-to=google \
  --nofollow-import-to=torch \
  --nofollow-import-to=torchvision \
  --nofollow-import-to=torchaudio \
  --nofollow-import-to=torch_audiomentations \
  --nofollow-import-to=torchcodec \
  --nofollow-import-to=torchmetrics \
  --nofollow-import-to=lightning \
  --nofollow-import-to=pytorch_lightning \
  --nofollow-import-to=librosa \
  --nofollow-import-to=soundfile \
  --nofollow-import-to=soxr \
  --nofollow-import-to=sentencepiece \
  --nofollow-import-to=transformers \
  --nofollow-import-to=pyannote \
  --include-package=src \
  --no-deployment-flag=excluded-module-usage \
  --no-deployment-flag=self-execution \
  --include-data-dir=dashboard=dashboard \
  --output-dir=colab_dist \
  --lto=no \
  --jobs=2 \
  dashboard_server.py

# Remove any duplicate C10/PyTorch libraries or packages that Nuitka might have auto-copied into the root dist
echo "Purging any stray duplicate shared libraries from dashboard_server.dist..."
rm -f colab_dist/dashboard_server.dist/libc10*.so
rm -f colab_dist/dashboard_server.dist/libtorch*.so
rm -f colab_dist/dashboard_server.dist/libshm*.so
rm -rf colab_dist/dashboard_server.dist/torch*
rm -rf colab_dist/dashboard_server.dist/soxr
rm -rf colab_dist/dashboard_server.dist/librosa

# Move vendor_packages directly into the compiled distribution directory
echo "Integrating vendor_packages into dashboard_server.dist..."
mv colab_dist/vendor_packages colab_dist/dashboard_server.dist/
cp HOW_TO_USE.txt colab_dist/dashboard_server.dist/ 2>/dev/null || true

if [ "${BUNDLE_MODELS:-0}" = "1" ] || [ "${1}" = "--with-models" ]; then
  echo ""
  echo "[Extra] Downloading all Hugging Face AI models into offline bundle..."
  mkdir -p colab_dist/hf_models
  python3 -c "
import os
from huggingface_hub import snapshot_download

hf_token = os.environ.get('HF_TOKEN')
models_dir = 'colab_dist/hf_models'

models = [
    'bodhan-ai/indic-transcribe-core',
    'pyannote/speaker-diarization-community-1',
    'Horizon-Labs/punctuation-restoration-small',
    'Qwen/Qwen2.5-3B-Instruct',
]
for model_id in models:
    print(f'Downloading {model_id}...')
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=os.path.join(models_dir, 'hub', 'models--' + model_id.replace('/', '--')),
            token=hf_token,
        )
    except Exception as exc:
        print(f'Warning: Could not download {model_id}: {exc}')
"
  mv colab_dist/hf_models colab_dist/dashboard_server.dist/
fi

echo ""
echo "[6/6] Packaging complete self-contained production archive..."
tar -czf customer_support_pipeline_production_full.tar.gz -C colab_dist dashboard_server.dist

echo ""
echo "================================================================="
echo "  BUILD SUCCEEDED!"
echo "  Archive: customer_support_pipeline_production_full.tar.gz"
echo "================================================================="
