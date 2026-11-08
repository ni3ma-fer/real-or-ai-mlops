#!/usr/bin/env bash
# Déploie le système sur une VM Ubuntu en une commande (depuis Git Bash, Linux ou macOS).
#
#   scripts/deploy.sh ubuntu@<IP-de-la-VM> [--ship-images]
#
# --ship-images : construit l'image de l'API sur ce PC et copie toutes les images Docker
#                 vers la VM par SSH, au lieu que la VM les télécharge depuis internet
#                 (utile pour une VM locale avec une connexion internet lente).
#
# Idempotent : relançable autant de fois que voulu. Étapes :
#   1. vérifie la connexion SSH
#   2. installe Docker sur la VM s'il manque
#   3. envoie le code du dernier commit (git archive : uniquement ce qui est commité)
#   4. génère les mots de passe de production à la première fois (~/real-or-ai.env sur la VM)
#   5. (option) copie les images Docker depuis ce PC
#   6. construit et lance les conteneurs, puis attend que l'API réponde
#
# Variable optionnelle : SSH_KEY (défaut ~/.ssh/realorai_deploy)
set -euo pipefail

HOST="${1:?Usage : scripts/deploy.sh ubuntu@<IP-de-la-VM> [--ship-images]}"
SHIP=false; [ "${2:-}" = "--ship-images" ] && SHIP=true
KEY="${SSH_KEY:-$HOME/.ssh/realorai_deploy}"
DIR="real-or-ai"
ssh_vm() { ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$HOST" "$@"; }

cd "$(git rev-parse --show-toplevel)"
COMMIT="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  echo "⚠  Modifications non commitées : elles ne seront PAS déployées (seul le commit $COMMIT l'est)."
fi

echo "1/6 Connexion à $HOST..."
ssh_vm 'echo "    OK : $(hostname), $(lsb_release -ds 2>/dev/null || uname -sr), $(nproc) CPU, $(free -h | awk "/Mem/ {print \$2}") RAM"'

echo "2/6 Docker..."
ssh_vm 'if command -v docker >/dev/null; then echo "    déjà installé : $(docker --version)";
        else curl -fsSL https://get.docker.com | sudo sh >/dev/null && sudo usermod -aG docker "$USER" && echo "    installé : $(docker --version)"; fi'

echo "3/6 Envoi du code (commit $COMMIT)..."
git archive --format=tar HEAD | ssh_vm "rm -rf $DIR && mkdir -p $DIR && tar -x -C $DIR && echo $COMMIT > $DIR/DEPLOYED_COMMIT"

echo "4/6 Secrets de production..."
ssh_vm 'if [ -f ~/real-or-ai.env ]; then echo "    déjà présents (~/real-or-ai.env)";
        else umask 077; printf "POSTGRES_PASSWORD=%s\nGRAFANA_ADMIN_PASSWORD=%s\n" "$(openssl rand -hex 16)" "$(openssl rand -hex 16)" > ~/real-or-ai.env;
             echo "    générés dans ~/real-or-ai.env (lisible seulement par $USER)"; fi'

BUILD="--build"
if $SHIP; then
  echo "5/6 Copie des images Docker depuis ce PC..."
  # Images construites à partir du commit (pas des fichiers modifiés non commités)
  for svc in api:api/Dockerfile ui:ui/Dockerfile drift:monitoring/drift/Dockerfile; do
    git archive --format=tar HEAD | docker build -q -f "${svc#*:}" -t "real-or-ai-${svc%%:*}:latest" - >/dev/null
  done
  for img in real-or-ai-api:latest real-or-ai-ui:latest real-or-ai-drift:latest              $(docker compose -f docker-compose.yml config --images | grep -v '^real-or-ai-'); do
    local_id=$(docker image inspect -f '{{.Id}}' "$img" 2>/dev/null) || {
      docker pull -q "$img" >/dev/null; local_id=$(docker image inspect -f '{{.Id}}' "$img"); }
    if [ "$(ssh_vm "sudo docker image inspect -f '{{.Id}}' $img 2>/dev/null" || true)" = "$local_id" ]; then
      echo "    $img : déjà à jour"
    else
      start=$SECONDS; docker save "$img" | ssh_vm 'sudo docker load -q' >/dev/null
      echo "    $img : copiée ($((SECONDS - start)) s)"
    fi
  done
  BUILD=""
else
  echo "5/6 Images Docker : téléchargées par la VM"
fi

echo "6/6 Lancement des conteneurs..."
ssh_vm "cd $DIR && sudo docker compose --env-file ~/real-or-ai.env -f docker-compose.yml -f docker-compose.prod.yml up -d $BUILD --remove-orphans --quiet-pull 2>&1 | grep -vE '^ *#|Pulling|Waiting|Downloading|Extracting|Verifying' | tail -12"

echo "Attente de l'API..."
for _ in $(seq 1 30); do
  code=$(ssh_vm 'curl -s -o /dev/null -w "%{http_code}" localhost/health' || true)
  if [ "$code" = "200" ]; then
    echo "✅ Déployé : $(ssh_vm 'curl -s localhost/health') — http://${HOST#*@}/docs"
    exit 0
  fi
  sleep 5
done
if [ "$code" = "503" ]; then
  echo "✅ Conteneurs lancés, mais aucun modèle « champion » en production pour l'instant."
  echo "   Promouvoir le modèle de dev : scripts/promote_model.sh $HOST"
  exit 0
fi
echo "❌ L'API ne répond pas (code $code). Logs : ssh -i $KEY $HOST 'cd $DIR && sudo docker compose logs api --tail 50'"
exit 1
