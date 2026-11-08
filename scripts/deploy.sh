#!/usr/bin/env bash
# Déploie le système sur une VM Ubuntu en une commande (depuis Git Bash, Linux ou macOS).
#
#   scripts/deploy.sh ubuntu@<IP-de-la-VM>
#
# Idempotent : relançable autant de fois que voulu. Étapes :
#   1. vérifie la connexion SSH
#   2. installe Docker sur la VM s'il manque
#   3. envoie le code du dernier commit (git archive : uniquement ce qui est commité)
#   4. génère les mots de passe de production à la première fois (~/real-or-ai.env sur la VM)
#   5. construit et lance les conteneurs, puis attend que l'API réponde
#
# Variable optionnelle : SSH_KEY (défaut ~/.ssh/realorai_deploy)
set -euo pipefail

HOST="${1:?Usage : scripts/deploy.sh ubuntu@<IP-de-la-VM>}"
KEY="${SSH_KEY:-$HOME/.ssh/realorai_deploy}"
DIR="real-or-ai"
ssh_vm() { ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$HOST" "$@"; }

cd "$(git rev-parse --show-toplevel)"
COMMIT="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  echo "⚠  Modifications non commitées : elles ne seront PAS déployées (seul le commit $COMMIT l'est)."
fi

echo "1/5 Connexion à $HOST..."
ssh_vm 'echo "    OK : $(hostname), $(lsb_release -ds 2>/dev/null || uname -sr), $(nproc) CPU, $(free -h | awk "/Mem/ {print \$2}") RAM"'

echo "2/5 Docker..."
ssh_vm 'if command -v docker >/dev/null; then echo "    déjà installé : $(docker --version)";
        else curl -fsSL https://get.docker.com | sudo sh >/dev/null && sudo usermod -aG docker "$USER" && echo "    installé : $(docker --version)"; fi'

echo "3/5 Envoi du code (commit $COMMIT)..."
git archive --format=tar HEAD | ssh_vm "rm -rf $DIR && mkdir -p $DIR && tar -x -C $DIR && echo $COMMIT > $DIR/DEPLOYED_COMMIT"

echo "4/5 Secrets de production..."
ssh_vm 'if [ -f ~/real-or-ai.env ]; then echo "    déjà présents (~/real-or-ai.env)";
        else umask 077; printf "POSTGRES_PASSWORD=%s\nGRAFANA_ADMIN_PASSWORD=%s\n" "$(openssl rand -hex 16)" "$(openssl rand -hex 16)" > ~/real-or-ai.env;
             echo "    générés dans ~/real-or-ai.env (lisible seulement par $USER)"; fi'

echo "5/5 Construction et lancement des conteneurs..."
ssh_vm "cd $DIR && sudo docker compose --env-file ~/real-or-ai.env -f docker-compose.yml -f docker-compose.prod.yml up -d --build --remove-orphans --quiet-pull 2>&1 | grep -vE '^ *#|Pulling|Waiting|Downloading|Extracting|Verifying' | tail -12"

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
