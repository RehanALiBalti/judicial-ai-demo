#!/usr/bin/env bash
# JAMS — one-time Ubuntu server setup
# Run as root or with sudo: sudo bash deploy/ubuntu-setup.sh
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/jams}"
APP_USER="${APP_USER:-www-data}"
DOMAIN="${DOMAIN:-_}"

echo "==> JAMS setup — app dir: $APP_DIR"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
  python3 python3-venv python3-pip \
  nginx git curl ca-certificates unzip \
  build-essential

# Node.js 20 LTS (for building frontend)
if ! command -v node &>/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi

# Ollama (LLM)
if ! command -v ollama &>/dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

mkdir -p "$APP_DIR"
chown -R "$APP_USER:$APP_USER" "$APP_DIR" 2>/dev/null || true

if [[ -f "$APP_DIR/requirements.txt" ]]; then
  echo "==> Python venv + dependencies"
  sudo -u "$APP_USER" python3 -m venv "$APP_DIR/.venv"
  sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install --upgrade pip
  sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"
fi

if [[ -f "$APP_DIR/frontend/package.json" ]]; then
  echo "==> Build frontend"
  cd "$APP_DIR/frontend"
  sudo -u "$APP_USER" npm ci 2>/dev/null || sudo -u "$APP_USER" npm install
  sudo -u "$APP_USER" env VITE_API_URL= npm run build
fi

if [[ ! -f "$APP_DIR/.env" && -f "$APP_DIR/deploy/env.example" ]]; then
  cp "$APP_DIR/deploy/env.example" "$APP_DIR/.env"
  echo "==> Created $APP_DIR/.env — edit CORS_ORIGINS and domain before going live"
fi

mkdir -p "$APP_DIR/data/fccp/pdfs"
chown -R "$APP_USER:$APP_USER" "$APP_DIR/data" 2>/dev/null || true

echo "==> Install systemd service"
cp "$APP_DIR/deploy/jams-backend.service" /etc/systemd/system/jams-backend.service
systemctl daemon-reload
systemctl enable jams-backend

echo "==> Configure nginx"
sed "s/YOUR_DOMAIN/$DOMAIN/g" "$APP_DIR/deploy/nginx-jams.conf" > /etc/nginx/sites-available/jams
ln -sf /etc/nginx/sites-available/jams /etc/nginx/sites-enabled/jams
rm -f /etc/nginx/sites-enabled/default
nginx -t

echo "==> Pull Ollama model (may take a few minutes)"
ollama pull qwen2.5:1.5b || true

echo ""
echo "Done. Next steps:"
echo "  1. Edit $APP_DIR/.env"
echo "  2. sudo systemctl start jams-backend"
echo "  3. sudo systemctl reload nginx"
echo "  4. Open http://$DOMAIN in browser"
echo "  5. FCCP Import tab — sync judgments on this server (PDF paths from Windows won't work)"
