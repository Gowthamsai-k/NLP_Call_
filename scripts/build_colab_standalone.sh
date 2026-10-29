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
sudo apt-get install -y -qq ffmpeg patchelf libsndfile1

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
  --include-package=src \
  --include-package=pysqlite3 \
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

echo ""
echo "[6/6] Packaging complete self-contained production archive..."
tar -czf customer_support_pipeline_production_full.tar.gz -C colab_dist dashboard_server.dist

echo ""
echo "================================================================="
echo "  BUILD SUCCEEDED!"
echo "  Archive: customer_support_pipeline_production_full.tar.gz"
echo "================================================================="
