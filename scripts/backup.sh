#!/usr/bin/env bash
# Backup: PostgreSQL dump + uploads volume tar. Run on orchestrator host (Linux). ASCII only.
# Usage:  ./scripts/backup.sh   (writes to backups/; schedule via cron)
set -euo pipefail
cd "$(dirname "$0")/.."
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p backups

docker compose -f deploy/docker-compose.prod.yml exec -T postgres pg_dump -U rag -d rag -F c \
  > "backups/rag_${STAMP}.dump"
echo "DB   -> backups/rag_${STAMP}.dump"

# uploads named volume (project name is fixed to `rag` by compose `name: rag`)
docker run --rm -v rag_upload:/data -v "$(pwd)/backups:/backup" alpine \
  sh -c "tar czf /backup/uploads_${STAMP}.tgz -C /data ."
echo "UP   -> backups/uploads_${STAMP}.tgz"
echo "done"
