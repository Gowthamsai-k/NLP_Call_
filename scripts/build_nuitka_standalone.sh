#!/usr/bin/env bash
set -e

echo "=== Starting Production Standalone Build with Nuitka ==="

# Ensure working directory is project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH=".:${PYTHONPATH:-}"

# Clean existing build distribution
rm -rf build_dist
mkdir -p build_dist/vendor_packages

echo "=== Vendoring runtime packages into offline bundle ==="
python3 -m pip install -r requirements.txt sentencepiece soundfile pyyaml psutil --target build_dist/vendor_packages --no-compile 2>/dev/null || true

echo "=== Compiling dashboard_server.py with Nuitka ==="
python3 -m nuitka \
  --standalone \
  --disable-plugin=gi \
  --nofollow-import-to=gi \
  --nofollow-import-to=keyring \
  --nofollow-import-to=numpy \
  --nofollow-import-to=scipy \
  --nofollow-import-to=torch \
  --nofollow-import-to=transformers \
  --nofollow-import-to=pyannote \
  --nofollow-import-to=IPython \
  --nofollow-import-to=jedi \
  --nofollow-import-to=traitlets \
  --nofollow-import-to=pygments \
  --nofollow-import-to=matplotlib \
  --include-package=src \
  --no-deployment-flag=excluded-module-usage \
  --no-deployment-flag=self-execution \
  --include-data-dir=dashboard=dashboard \
  --output-dir=build_dist \
  --jobs=4 \
  dashboard_server.py

# Move vendor_packages into dashboard_server.dist
if [ -d "build_dist/vendor_packages" ]; then
  mv build_dist/vendor_packages build_dist/dashboard_server.dist/
fi

# Ensure dashboard assets and documentation are present in the distribution
cp HOW_TO_USE.txt build_dist/dashboard_server.dist/ 2>/dev/null || true

echo "=== Creating production tar.gz archive ==="
tar -czf customer_support_dashboard_standalone.tar.gz -C build_dist dashboard_server.dist

echo "=== Standalone build completed successfully ==="
echo "Distribution folder: build_dist/dashboard_server.dist"
echo "Distribution archive: customer_support_dashboard_standalone.tar.gz"
