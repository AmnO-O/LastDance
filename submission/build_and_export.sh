#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
IMAGE_NAME="zac2026-lifemate:v1"
TAR_NAME="zac2026_LifeMate.tar.gz"

echo "=== [1/3] Building ${IMAGE_NAME} ==="
docker build -f submission/Dockerfile -t "${IMAGE_NAME}" .

echo "=== [2/3] Exporting ${TAR_NAME} ==="
docker save "${IMAGE_NAME}" | gzip > "${TAR_NAME}"

echo "=== [3/3] MD5 ==="
md5sum "${TAR_NAME}" > "${TAR_NAME}.md5"
cat "${TAR_NAME}.md5"
