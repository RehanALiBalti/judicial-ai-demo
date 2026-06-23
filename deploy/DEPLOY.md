# JAMS — Ubuntu Deployment Guide

Deploy **backend** (FastAPI) + **frontend** (React/Vite) on a single Ubuntu server with **nginx**, **systemd**, and **Ollama**.

Recommended server: Ubuntu 22.04/24.04, 8+ GB RAM, GPU optional (faster Ollama).

---

## Fresh server + ZIP upload (start here)

Server bilkul khali ho — Python/Node kuch nahi — ye steps follow karein.

### Option 1: Git clone (recommended)

**Windows par GitHub par repo banayein** (neeche section), phir server par:

```bash
sudo apt update
sudo apt install -y git

sudo git clone https://github.com/RehanALiBalti/judicial-ai-demo.git /opt/jams
sudo chown -R www-data:www-data /opt/jams
cd /opt/jams
sudo DOMAIN=65.108.236.135 bash /opt/jams/deploy/ubuntu-setup.sh
```

Baad mein code update:

```bash
cd /opt/jams
sudo -u www-data git pull
cd frontend && sudo -u www-data npm ci && sudo -u www-data env VITE_API_URL= npm run build
sudo systemctl restart jams-backend
```

### Option 2: ZIP upload

### A) Windows par ZIP banayein

Project folder se zip banayein. **Ye folders ZIP mein na rakhein** (size barh jati hai, server par dubara install honge):

- `.venv`
- `frontend/node_modules`
- `frontend/dist`
- `__pycache__`

ZIP mein ye hona chahiye: `app.py`, `backend/`, `frontend/`, `deploy/`, `requirements.txt`, `data/` (optional)

Example PowerShell:

```powershell
cd E:\python\ji
Compress-Archive -Path judicial-ai-demo -DestinationPath jams.zip
```

### B) ZIP server par kahan upload karein?

Upload location: **`/home/ubuntu/`** (ya jo bhi aapka SSH user ho)

| Method | Command / Tool |
|--------|----------------|
| **SCP (recommended)** | `scp E:\python\ji\jams.zip ubuntu@65.108.236.135:/home/ubuntu/` |
| **WinSCP / FileZilla** | Connect karein → right panel mein `/home/ubuntu/` → zip drag & drop |
| **Cloud panel** | Provider ka file upload → phir SSH se move karein |

`65.108.236.135` = apne Ubuntu instance ka public IP (e.g. `203.0.113.10`)

### C) Server par SSH login

```bash
ssh ubuntu@65.108.236.135
```

(Pehli dafa password ya SSH key provider se milta hai.)

### D) ZIP unzip karke sahi jagah rakhein

**Final app path hamesha:** `/opt/jams`

```bash
# unzip tool (fresh server par)
sudo apt update
sudo apt install -y unzip

# zip home folder mein hai
cd ~
unzip jams.zip

# folder name check karein (judicial-ai-demo ya jams)
ls

# /opt/jams par move karein
sudo mkdir -p /opt
sudo mv ~/judicial-ai-demo /opt/jams
# agar zip andar ek aur folder banaye to path adjust karein:
# sudo mv ~/judicial-ai-demo/judicial-ai-demo /opt/jams

sudo chown -R www-data:www-data /opt/jams
```

### E) Auto setup (Python, Node, nginx, Ollama sab install)

```bash
cd /opt/jams
sudo DOMAIN=65.108.236.135 bash /opt/jams/deploy/ubuntu-setup.sh
```

`65.108.236.135` ki jagah apna real IP likhein.

Ye script 10–20 minute le sakti hai (Ollama model download).

### F) .env edit + services start

```bash
sudo nano /opt/jams/.env
```

`CORS_ORIGINS` mein apna IP:

```env
CORS_ORIGINS=http://203.0.113.10
```

Phir:

```bash
sudo systemctl start ollama
sudo systemctl start jams-backend
sudo systemctl reload nginx

# firewall (agar enabled ho)
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
```

Browser: **`http://65.108.236.135`**

### G) Pehli dafa data

Windows ka `data/jams_store.json` Ubuntu par theek kaam nahi karega. Server par:

1. App kholein → **FCCP Import** tab
2. **Sync** chalayein (judgments download + index)

---

## 1. Copy project to server (ZIP ke bajaye direct copy)

From your Windows machine (PowerShell):

```powershell
scp -r E:\python\ji\judicial-ai-demo ubuntu@YOUR_65.108.236.135:/tmp/jams
```

On Ubuntu:

```bash
sudo mv /tmp/jams /opt/jams
sudo chown -R www-data:www-data /opt/jams
```

> **Note:** `data/jams_store.json` from Windows has Windows PDF paths. On Ubuntu, use **FCCP Import → Sync** to re-index judgments, or re-upload cases.

---

## 2. One-time setup script

```bash
cd /opt/jams
sudo DOMAIN=65.108.236.135 bash /opt/jams/deploy/ubuntu-setup.sh
```

Replace `YOUR_65.108.236.135` with your public IP or domain (e.g. `203.0.113.10` or `jams.example.com`).

---

## 3. Configure environment

```bash
sudo nano /opt/jams/.env
```

Example:

```env
HOST=127.0.0.1
PORT=8000
CORS_ORIGINS=http://203.0.113.10,https://jams.example.com
OLLAMA_URL=http://127.0.0.1:11434/api/generate
OLLAMA_MODEL=qwen2.5:1.5b
```

---

## 4. Start services

```bash
# Ollama (usually auto-starts after install)
sudo systemctl enable ollama
sudo systemctl start ollama
ollama pull qwen2.5:1.5b

# Backend
sudo systemctl start jams-backend
sudo systemctl status jams-backend

# Nginx
sudo systemctl reload nginx
```

Open: `http://YOUR_65.108.236.135`

---

## 5. Firewall (if ufw enabled)

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

---

## 6. HTTPS (recommended for production)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d jams.example.com
```

Update `CORS_ORIGINS` in `.env` to include `https://jams.example.com`, then:

```bash
sudo systemctl restart jams-backend
```

---

## 7. Manual dev mode (testing only)

**Backend:**

```bash
cd /opt/jams
source .venv/bin/activate
python app.py
```

**Frontend (separate terminal):**

```bash
cd /opt/jams/frontend
npm install
npm run dev -- --host 0.0.0.0
```

Set in `.env`: `CORS_ORIGINS=http://65.108.236.135:5173`

For production, always use **nginx + built frontend** (`npm run build`), not Vite dev server.

---

## 8. Update after code changes

```bash
cd /opt/jams
git pull   # if using git

source .venv/bin/activate
pip install -r requirements.txt

cd frontend
npm ci
VITE_API_URL= npm run build

sudo systemctl restart jams-backend
sudo systemctl reload nginx
```

---

## 9. Troubleshooting

| Problem | Fix |
|--------|-----|
| `ModuleNotFoundError: uvicorn` | Use `/opt/jams/.venv/bin/python`, not system Python |
| Chat returns "AI generation failed" | `ollama list`, ensure `qwen2.5:1.5b` is pulled; `curl http://127.0.0.1:11434/api/tags` |
| 502 Bad Gateway | `sudo journalctl -u jams-backend -f` — backend still loading embeddings (~1–2 min first start) |
| Empty cases | Run **FCCP Import → Sync** on the server |
| `npm error EACCES /var/www/.npm` | Frontend build manually (neeche commands) |

**Frontend build fix (server par):**

```bash
sudo rm -rf /opt/jams/frontend/node_modules
sudo mkdir -p /opt/jams/.npm-cache /opt/jams/.home
sudo chown -R www-data:www-data /opt/jams

cd /opt/jams/frontend
sudo -u www-data env HOME=/opt/jams/.home NPM_CONFIG_CACHE=/opt/jams/.npm-cache npm install --no-audit --no-fund
sudo -u www-data env HOME=/opt/jams/.home VITE_API_URL= npm run build
```

**Logs:**

```bash
sudo journalctl -u jams-backend -f
sudo tail -f /var/log/nginx/error.log
```

---

## Architecture

```
Browser → nginx:80/443
            ├── /          → frontend/dist (static React)
            └── /api/*     → uvicorn 127.0.0.1:8000 (FastAPI)
                                    └── Ollama 127.0.0.1:11434
```

Ollama should stay on **localhost only** — do not expose port 11434 to the internet.

---

## Push project to GitHub (from Windows)

### 1. GitHub par naya public repo

1. https://github.com/new kholein
2. Repository name: `jams` (ya `judicial-ai-demo`)
3. **Public** select karein
4. README / .gitignore **mat** add karein (project mein pehle se hain)
5. **Create repository**

### 2. Windows par push (pehli dafa)

```powershell
cd E:\python\ji\judicial-ai-demo

git init
git add .
git status
git commit -m "Initial commit: JAMS judicial AI system"

git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/jams.git
git push -u origin main
```

`YOUR_USERNAME` apna GitHub username likhein.

GitHub login pooche to **Personal Access Token** use karein (password nahi chalta):
https://github.com/settings/tokens → Generate new token (classic) → `repo` scope

### 3. Server par clone

```bash
sudo git clone https://github.com/RehanALiBalti/judicial-ai-demo.git /opt/jams
    ```
