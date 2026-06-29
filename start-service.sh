#!/bin/bash
set -euo pipefail

GATEWAY_DIR="/Users/mytselrunner/kiro-gateway"
PYTHON="/opt/homebrew/bin/python3"

cd "$GATEWAY_DIR"

# Load .env file
set -a
source "$GATEWAY_DIR/.env"
set +a

exec "$PYTHON" main.py
