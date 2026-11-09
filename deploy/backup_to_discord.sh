#!/bin/bash
# Nightly: dumps the FinSight database and posts it to the Discord alerts channel.
# Posts older than KEEP_DAYS are deleted. Any failure is reported to the same channel.
# Cron: 0 4 * * * /home/finsight/FinSight/deploy/backup_to_discord.sh
set -uo pipefail

KEEP_DAYS=30
MAX_BYTES=9500000                  # Discord's limit for a file is about 10 MB
SENT=~/.finsight_backups_sent      # one line per post: <epoch> <message id>
cd "$(dirname "$0")/.." || exit 1

WEBHOOK=$(grep -m1 '^DISCORD_WEBHOOK_URL=' .env | cut -d= -f2- | tr -d '\r')
[ -n "$WEBHOOK" ] || { echo "DISCORD_WEBHOOK_URL is not set"; exit 1; }
say() { curl -s -m 15 -H 'Content-Type: application/json' -d "{\"username\":\"FinSight alerts\",\"content\":\"$1\"}" "$WEBHOOK" >/dev/null; }

FILE=/tmp/finsight-$(date -u +%Y-%m-%d).sql.gz
trap 'rm -f "$FILE"' EXIT

if ! docker exec finsight-postgres pg_dump -U finsight -d finsight --clean --if-exists --no-owner | gzip > "$FILE"; then
  say "**FinSight backup failed**: could not dump the database."; exit 1
fi
SIZE=$(stat -c %s "$FILE")
if [ "$SIZE" -gt "$MAX_BYTES" ]; then
  say "**FinSight backup failed**: the file is $SIZE bytes, over Discord's limit."; exit 1
fi

RESPONSE=$(curl -s -m 60 -F "username=FinSight alerts" -F "content=FinSight database backup ($SIZE bytes)" -F "file=@$FILE;type=application/gzip" "$WEBHOOK?wait=true")
ID=$(printf '%s' "$RESPONSE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])' 2>/dev/null)
if [ -z "$ID" ]; then
  say "**FinSight backup failed**: Discord did not accept the file."; exit 1
fi
echo "$(date +%s) $ID" >> "$SENT"
printf '%s\n' "$RESPONSE"          # the attachment URL, for testing a restore by hand

# Delete posts past the retention window, keeping the record of any that could not be deleted.
CUTOFF=$(( $(date +%s) - KEEP_DAYS * 86400 ))
KEPT=$(mktemp)
while read -r WHEN OLD; do
  if [ "$WHEN" -lt "$CUTOFF" ]; then
    CODE=$(curl -s -m 15 -o /dev/null -w '%{http_code}' -X DELETE "$WEBHOOK/messages/$OLD")
    case "$CODE" in 204|404) continue ;; esac
  fi
  echo "$WHEN $OLD" >> "$KEPT"
done < "$SENT"
mv "$KEPT" "$SENT"
