#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"
# correção do Python do Homebrew no macOS 26 (expat): o instalador grava o caminho aqui quando precisa
[ -f "$DIR/.expat" ] && export DYLD_LIBRARY_PATH="$(cat "$DIR/.expat")${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
exec "$DIR/.venv/bin/python" -m cloaker "$@"
