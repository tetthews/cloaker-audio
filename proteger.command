#!/usr/bin/env bash
# Mac: dê dois cliques. Ele pede o vídeo numa janela e protege.
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ $# -eq 0 ]; then
  ARQ=$(osascript -e 'POSIX path of (choose file with prompt "Escolha o vídeo para proteger")' 2>/dev/null) || exit 0
  set -- "$ARQ"
fi
for f in "$@"; do
  echo "=== Protegendo $(basename "$f") ==="
  "$DIR/cloaker.sh" proteger "$f"
done
echo
echo "Pronto. Cada vídeo ganhou uma pasta NOME_protegido ao lado do original, com o relatorio.html."
read -n 1 -s -r -p "Aperte qualquer tecla para fechar."
