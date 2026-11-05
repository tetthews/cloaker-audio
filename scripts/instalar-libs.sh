#!/usr/bin/env bash
# Instalação no Mac (Apple Silicon usa a GPU via MPS) ou Linux.
set -euo pipefail
cd "$(dirname "$0")/.."
# o macOS apaga DYLD_* ao abrir o bash: relê a correção do expat (ver o instalador do Mac)
if [ -f .expat ]; then export DYLD_LIBRARY_PATH="$(cat .expat)${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"; fi
echo "== Cloaker de áudio: instalação =="
PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null; then
  echo "Python 3 não encontrado. No Mac: brew install python@3.12"; exit 1
fi
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
if [ "$(uname)" = "Darwin" ]; then
  .venv/bin/python -m pip install torch torchaudio
elif command -v nvidia-smi >/dev/null; then
  .venv/bin/python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
else
  .venv/bin/python -m pip install torch torchaudio
fi
.venv/bin/python -m pip install -r requirements.txt
if ! command -v ffmpeg >/dev/null; then
  echo "Aviso: ffmpeg não encontrado; será usado o embutido. Recomendado: brew install ffmpeg"
fi
chmod +x cloaker.sh *.command scripts/*.sh 2>/dev/null || true
.venv/bin/python -m cloaker diagnostico
