#!/usr/bin/env bash
# Rebuild frontend after git pull (production nginx serves frontend/dist)
set -euo pipefail
APP_DIR="${APP_DIR:-/opt/jams}"
APP_USER="${APP_USER:-www-data}"
NPM_CACHE="$APP_DIR/.npm-cache"
APP_HOME="$APP_DIR/.home"

cd "$APP_DIR/frontend"
sudo -u "$APP_USER" env \
  HOME="$APP_HOME" \
  NPM_CONFIG_CACHE="$NPM_CACHE" \
  npm ci 2>/dev/null || sudo -u "$APP_USER" env HOME="$APP_HOME" NPM_CONFIG_CACHE="$NPM_CACHE" npm install
sudo -u "$APP_USER" env HOME="$APP_HOME" VITE_API_URL= npm run build
echo "Frontend rebuilt → $APP_DIR/frontend/dist"
