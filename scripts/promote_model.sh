#!/usr/bin/env bash
# Promeut le modèle champion de dev (MLflow local) vers la production, par un tunnel SSH.
#   scripts/promote_model.sh ubuntu@<IP-de-la-VM> [options de promote_model.py]
set -euo pipefail

HOST="${1:?Usage : scripts/promote_model.sh ubuntu@<IP-de-la-VM> [--version N] [--alias A]}"
shift
KEY="${SSH_KEY:-$HOME/.ssh/realorai_deploy}"
PORT=5001
PY="$(git rev-parse --show-toplevel)/.venv/Scripts/python.exe"
[ -x "$PY" ] || PY="$(git rev-parse --show-toplevel)/.venv/bin/python"

# Tunnel : localhost:5001 sur ce PC -> 127.0.0.1:5000 sur la VM (MLflow de production)
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ExitOnForwardFailure=yes -N -L "$PORT:127.0.0.1:5000" "$HOST" &
TUNNEL=$!
trap 'kill $TUNNEL 2>/dev/null' EXIT
for _ in $(seq 1 20); do curl -sf "http://localhost:$PORT/health" >/dev/null && break; sleep 1; done

MLFLOW_DISABLE_AGENT_HINT=1 PYTHONIOENCODING=utf-8 "$PY" "$(dirname "$0")/promote_model.py" --target "http://localhost:$PORT" "$@"

echo "Attente du chargement par l'API de production..."
for _ in $(seq 1 30); do
  if out=$(ssh -i "$KEY" "$HOST" 'curl -sf localhost/health'); then echo "✅ $out"; exit 0; fi
  sleep 5
done
echo "❌ L'API n'a pas chargé le modèle. Logs : ssh -i $KEY $HOST 'cd real-or-ai && sudo docker compose logs api --tail 50'"
exit 1
