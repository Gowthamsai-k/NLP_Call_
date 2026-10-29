#!/usr/bin/env bash
set -e

echo "=== Starting Production Standalone Build with Nuitka ==="

# Ensure working directory is project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH=".:${PYTHONPATH:-}"

# Clean existing build distribution
rm -rf build_dist/dashboard_server.dist

echo "=== Compiling dashboard_server.py ==="
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
  --output-dir=build_dist \
  --jobs=4 \
  dashboard_server.py

# Ensure dashboard assets and documentation are present in the distribution
cp HOW_TO_USE.txt build_dist/dashboard_server.dist/

echo "=== Creating production tar.gz archive ==="
tar -czf customer_support_dashboard_standalone.tar.gz -C build_dist dashboard_server.dist

echo "=== Standalone build completed successfully ==="
echo "Distribution folder: build_dist/dashboard_server.dist"
echo "Distribution archive: customer_support_dashboard_standalone.tar.gz"
