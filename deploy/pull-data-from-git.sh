#!/usr/bin/env bash
# Pull dataset from GitHub after push from PC.
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/jams}"
APP_USER="${APP_USER:-www-data}"

if command -v git-lfs >/dev/null 2>&1; then
  git lfs install
fi

cd "$APP_DIR"
sudo -u "$APP_USER" git pull
if command -v git-lfs >/dev/null 2>&1; then
  sudo -u "$APP_USER" git lfs pull
fi

echo "LHC PDFs: $(ls -1 "$APP_DIR/data/lhc/pdfs/"*.pdf 2>/dev/null | wc -l)"
chown -R "$APP_USER:$APP_USER" "$APP_DIR/data"
systemctl restart jams-backend
echo "Done."
