#!/bin/sh
set -eu

: "${BRAVE_API_KEY:?BRAVE_API_KEY is not set}"

python3 - <<'PY'
import os
from pathlib import Path

src = Path("/etc/searxng/settings.template.yml")
dst = Path("/tmp/searxng-settings.yml")

text = src.read_text()

placeholder = "__BRAVE_API_KEY__"
if placeholder not in text:
    raise SystemExit(f"{placeholder} missing from settings template")

text = text.replace(placeholder, os.environ["BRAVE_API_KEY"])

dst.write_text(text)
PY

export SEARXNG_SETTINGS_PATH=/tmp/searxng-settings.yml

exec /usr/local/searxng/entrypoint.sh
