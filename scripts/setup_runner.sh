#!/usr/bin/env bash
# Installe le runner GitHub Actions dans la VM de production (à faire une seule fois).
#
#   scripts/setup_runner.sh ubuntu@<IP-de-la-VM> <JETON>
#
# Le jeton s'obtient sur GitHub : dépôt > Settings > Actions > Runners > New self-hosted runner
# (ligne « ./config.sh --url ... --token XXXX »). Il expire au bout d'une heure.
# Prérequis : scripts/deploy.sh déjà lancé une fois (Docker installé sur la VM).
set -euo pipefail

HOST="${1:?Usage : scripts/setup_runner.sh ubuntu@<IP-de-la-VM> <JETON>}"
TOKEN="${2:?Jeton manquant (GitHub > Settings > Actions > Runners > New self-hosted runner)}"
KEY="${SSH_KEY:-$HOME/.ssh/realorai_deploy}"
REPO_URL="https://github.com/$(git remote get-url origin | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')"

ssh -i "$KEY" "$HOST" bash -s -- "$REPO_URL" "$TOKEN" <<'REMOTE'
set -euo pipefail
REPO_URL="$1"; TOKEN="$2"
VERSION=$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest | grep -oP '"tag_name": "v\K[^"]+')
mkdir -p ~/actions-runner && cd ~/actions-runner
if [ ! -f config.sh ]; then
  echo "Téléchargement du runner $VERSION..."
  curl -fsSL -o runner.tgz "https://github.com/actions/runner/releases/download/v${VERSION}/actions-runner-linux-x64-${VERSION}.tar.gz"
  tar xzf runner.tgz && rm runner.tgz
fi
sudo usermod -aG docker "$USER"
./config.sh --unattended --replace --url "$REPO_URL" --token "$TOKEN" --name "$(hostname)" --labels prod
sudo ./svc.sh install "$USER" >/dev/null
sudo ./svc.sh start >/dev/null
sudo ./svc.sh status | grep -E "Active:" || true
echo "✅ Runner « $(hostname) » (label prod) installé et démarré."
REMOTE
