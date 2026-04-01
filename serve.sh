#!/usr/bin/env bash
# Local preview for cluster_diagram.html. Restart: Ctrl+C in this terminal, then run again.
cd "$(dirname "$0")"
exec python3 -m http.server 8765
