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

echo ""
echo "[1/6] Installing OS audio and build libraries (ffmpeg, patchelf)..."
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg patchelf libsndfile1 libsecret-1-0 gir1.2-secret-1 || true

echo ""
echo "[2/6] Upgrading pip and installing Nuitka compiler..."
python3 -m pip install --upgrade pip
python3 -m pip install nuitka patchelf

echo ""
echo "[3/6] Installing full ML pipeline dependencies in environment..."
python3 -m pip install -r requirements.txt

echo ""
echo "[4/6] Vendoring all runtime packages into offline bundle..."
rm -rf colab_dist
mkdir -p colab_dist/vendor_packages
python3 -m pip install -r requirements.txt --target colab_dist/vendor_packages --no-compile

echo ""
echo "[5/6] Compiling dashboard_server with Nuitka standalone engine..."
python3 -m nuitka \
  --standalone \
  --disable-plugin=gi \
  --nofollow-import-to=gi \
  --nofollow-import-to=keyring \
  --include-package=src \
  --include-package=soundfile \
  --include-package=numpy \
  --include-package=scipy \
  --include-package=yaml \
  --include-package=psutil \
  --no-deployment-flag=excluded-module-usage \
  --no-deployment-flag=self-execution \
  --nofollow-import-to=IPython \
  --nofollow-import-to=jedi \
  --nofollow-import-to=traitlets \
  --nofollow-import-to=pygments \
  --nofollow-import-to=matplotlib \
  --nofollow-import-to=torch \
  --include-data-dir=dashboard=dashboard \
  --output-dir=colab_dist \
  --jobs=4 \
  dashboard_server.py

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
    'pyannote/speaker-diarization-community-1',
    'ai4bharat/indic-conformer-600m-multilingual',
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
