#!/usr/bin/env bash
# Change les alias du registre MLflow de PRODUCTION (par tunnel SSH). L'API applique le changement
# d'elle-même en moins de 30 s, sans redémarrage.
#
#   scripts/set_alias.sh ubuntu@<IP> promote            # challenger -> champion, fin du canary
#   scripts/set_alias.sh ubuntu@<IP> rollback           # supprime le challenger (le canary s'arrête)
#   scripts/set_alias.sh ubuntu@<IP> set champion 1     # met l'alias champion sur la version 1
#   scripts/set_alias.sh ubuntu@<IP> tag 2 quantized int8
set -euo pipefail
HOST="${1:?Usage : scripts/set_alias.sh ubuntu@<IP> promote|rollback|set <alias> <version>|tag <version> <clé> <valeur>}"
shift
KEY="${SSH_KEY:-$HOME/.ssh/realorai_deploy}"
PORT=5001
PY="$(git rev-parse --show-toplevel)/.venv/Scripts/python.exe"; [ -x "$PY" ] || PY="$(git rev-parse --show-toplevel)/.venv/bin/python"
ssh -i "$KEY" -o ExitOnForwardFailure=yes -N -L "$PORT:127.0.0.1:5000" "$HOST" &
TUNNEL=$!; trap 'kill $TUNNEL 2>/dev/null' EXIT
for _ in $(seq 1 20); do curl -sf "http://localhost:$PORT/health" >/dev/null && break; sleep 1; done
MLFLOW_DISABLE_AGENT_HINT=1 PYTHONIOENCODING=utf-8 "$PY" - "$PORT" "$@" <<'PY'
import sys
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
port, action, *rest = sys.argv[1:]
c, name = MlflowClient(f"http://localhost:{port}"), "real-or-ai"
def alias(a):
    try: return c.get_model_version_by_alias(name, a).version
    except MlflowException: return None
if action == "promote":
    v = alias("challenger") or sys.exit("Pas de challenger à promouvoir")
    c.set_registered_model_alias(name, "champion", v); c.delete_registered_model_alias(name, "challenger")
elif action == "rollback":
    alias("challenger") and c.delete_registered_model_alias(name, "challenger")
elif action == "set":
    c.set_registered_model_alias(name, rest[0], rest[1])
elif action == "tag":
    c.set_model_version_tag(name, rest[0], rest[1], rest[2])
print(f"Production : champion = v{alias('champion')}, challenger = v{alias('challenger')}")
PY
