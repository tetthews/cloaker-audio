#!/usr/bin/env bash
# Verifica se este Mac roda o Cloaker de áudio. Saída: 0 pronto, 1 com ressalvas, 2 não dá.
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
nivel=0
ok()    { printf "  \033[32m[OK]\033[0m      %s\n" "$1"; }
aviso() { printf "  \033[33m[ATENCAO]\033[0m %s\n" "$1"; [ $nivel -lt 1 ] && nivel=1; }
falha() { printf "  \033[31m[FALTA]\033[0m   %s\n" "$1"; nivel=2; }

echo
echo "============================================="
echo "   Cloaker de áudio · verificando o computador"
echo "============================================="
echo

# Sistema
ver="$(sw_vers -productVersion 2>/dev/null || echo 0)"
maior="${ver%%.*}"
if [ "$(uname)" != "Darwin" ]; then falha "isto não é um Mac"
elif [ "$maior" -ge 13 ] 2>/dev/null; then ok "macOS $ver"
else aviso "macOS $ver (recomendado 13 Ventura ou mais novo)"; fi

# Chip
chip="$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
if [ "$(uname -m)" = "arm64" ]; then ok "Chip: $chip (usa a GPU da Apple)"
else aviso "Chip Intel ($chip): funciona, mas bem lento (horas por vídeo)"; fi

# RAM
ram=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
if [ $ram -ge 16 ]; then ok "Memória RAM: ${ram} GB"
elif [ $ram -ge 8 ]; then aviso "Memória RAM: ${ram} GB (funciona; feche outros programas e use --perfil rapido)"
else falha "Memória RAM: ${ram} GB (mínimo 8 GB)"; fi

# Disco
livre=$(df -g "$RAIZ" 2>/dev/null | awk 'NR==2{print $4}')
livre=${livre:-0}
if [ "$livre" -ge 20 ]; then ok "Espaço livre: ${livre} GB"
elif [ "$livre" -ge 12 ]; then aviso "Espaço livre: ${livre} GB (justo; recomendado 20 GB)"
else falha "Espaço livre: ${livre} GB (precisa de pelo menos 12 GB)"; fi

# Ferramentas
if command -v brew >/dev/null 2>&1 || [ -x /opt/homebrew/bin/brew ] || [ -x /usr/local/bin/brew ]; then ok "Homebrew: instalado"
else ok "Homebrew: ainda não tem; o instalador instala (vai pedir a senha do Mac)"; fi
if xcode-select -p >/dev/null 2>&1; then ok "Ferramentas de linha de comando da Apple: instaladas"
else ok "Ferramentas da Apple: o instalador do Homebrew instala junto"; fi

# Internet
if curl -fsI --max-time 10 https://pypi.org/simple/pip/ >/dev/null 2>&1; then ok "Internet: conectado (precisa baixar ~8 GB na instalação)"
else falha "Sem acesso à internet (pypi.org)"; fi

[ -x "$RAIZ/.venv/bin/python" ] && ok "Cloaker: já instalado nesta pasta (pode reinstalar por cima)"

echo
case $nivel in
  0) printf "  \033[32mRESULTADO: computador pronto. Pode instalar.\033[0m\n" ;;
  1) printf "  \033[33mRESULTADO: dá pra instalar, com as ATENÇÕES acima.\033[0m\n" ;;
  2) printf "  \033[31mRESULTADO: falta algo (itens FALTA acima). Resolva antes de instalar.\033[0m\n" ;;
esac
echo
exit $nivel
