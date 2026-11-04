#!/usr/bin/env bash
# Dê dois cliques (na 1ª vez: botão direito > Abrir). Só confere; não instala nada.
DIR="$(cd "$(dirname "$0")" && pwd)"
xattr -dr com.apple.quarantine "$DIR" 2>/dev/null || true
bash "$DIR/scripts/verificar-mac.sh"
read -n 1 -s -r -p "Aperte qualquer tecla para fechar."
