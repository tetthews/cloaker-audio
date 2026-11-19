#!/usr/bin/env bash
# Mac: botão direito > Abrir. Abre o painel do Cloaker no navegador (arrastar vídeo, fila, relatório).
DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$DIR/.expat" ] && export DYLD_LIBRARY_PATH="$(cat "$DIR/.expat")${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
PY="$DIR/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "O Cloaker ainda não está instalado. Rode primeiro o \"2 - Instalar no Mac.command\"."
  read -n 1 -s -r -p "Aperte qualquer tecla para fechar."; exit 1
fi
echo "Fechar esta janela desliga o painel. Uma proteção em andamento continua e volta a aparecer quando abrir de novo."
"$PY" "$DIR/painel/painel.py" --pasta "$DIR"
