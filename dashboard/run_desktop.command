#!/bin/zsh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
  PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "python3 bulunamadi."
  read -r "?Kapatmak icin Enter'a basin..."
  exit 1
fi

cd "$SCRIPT_DIR"
"$PYTHON_BIN" desktop_app.py

read -r "?Pencere kapandi. Terminali kapatmak icin Enter'a basin..."
