#!/usr/bin/env bash
# One-command deploy (ASCII). Usage:  ./scripts/deploy.sh
# git pull -> build -> up -d -> wait health. Run on the orchestrator host (Linux).
set -euo pipefail
cd "$(dirname "$0")/.."
COMPOSE="docker compose -f deploy/docker-compose.prod.yml"

git pull --ff-only
$COMPOSE up -d --build
$COMPOSE ps

echo "Waiting for health (via caddy http://localhost/health)..."
for i in $(seq 1 30); do
  if curl -fsS http://localhost/health >/dev/null 2>&1; then
    echo "OK: http://localhost/health"
    exit 0
  fi
  sleep 2
done
echo "WARN: health not ready. Check: $COMPOSE logs api"
