#!/usr/bin/env bash
# Finish JAMS setup if ubuntu-setup.sh stopped early (e.g. npm error).
# Run: sudo DOMAIN=YOUR_IP bash /opt/jams/deploy/finish-setup.sh
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/jams}"
APP_USER="${APP_USER:-www-data}"
DOMAIN="${DOMAIN:-_}"

echo "==> Finish JAMS setup"

if [[ ! -f "$APP_DIR/frontend/dist/index.html" ]]; then
  echo "==> Build frontend"
  NPM_CACHE="$APP_DIR/.npm-cache"
  APP_HOME="$APP_DIR/.home"
  mkdir -p "$NPM_CACHE" "$APP_HOME"
  chown -R "$APP_USER:$APP_USER" "$NPM_CACHE" "$APP_HOME" "$APP_DIR/frontend"
  rm -rf "$APP_DIR/frontend/node_modules"
  cd "$APP_DIR/frontend"
  sudo -u "$APP_USER" env \
    HOME="$APP_HOME" \
    NPM_CONFIG_CACHE="$NPM_CACHE" \
    npm install --no-audit --no-fund
  sudo -u "$APP_USER" env HOME="$APP_HOME" VITE_API_URL= npm run build
fi

if [[ ! -f "$APP_DIR/.env" ]]; then
  cp "$APP_DIR/deploy/env.example" "$APP_DIR/.env"
  echo "==> Created $APP_DIR/.env — edit CORS_ORIGINS"
fi

mkdir -p "$APP_DIR/data/fccp/pdfs" "$APP_DIR/.npm-cache" "$APP_DIR/.home" "$APP_DIR/.cache"
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

cp "$APP_DIR/deploy/jams-backend.service" /etc/systemd/system/jams-backend.service
systemctl daemon-reload
systemctl enable jams-backend

sed "s/YOUR_DOMAIN/$DOMAIN/g" "$APP_DIR/deploy/nginx-jams.conf" > /etc/nginx/sites-available/jams
ln -sf /etc/nginx/sites-available/jams /etc/nginx/sites-enabled/jams
rm -f /etc/nginx/sites-enabled/default
nginx -t

systemctl enable ollama 2>/dev/null || true
systemctl start ollama 2>/dev/null || true
ollama pull qwen2.5:7b 2>/dev/null || true

systemctl restart jams-backend
systemctl reload nginx

echo ""
echo "==> Status"
systemctl is-active jams-backend nginx || true
curl -sf http://127.0.0.1:8000/api/health && echo "" || echo "Backend not responding on :8000"
curl -sf "http://127.0.0.1/api/stats" && echo "" || echo "Nginx /api proxy not working"

echo ""
echo "Open: http://$DOMAIN"
