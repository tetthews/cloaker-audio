#!/usr/bin/env bash
# Instalador do Cloaker de áudio para Mac. Dê dois cliques (na 1ª vez: botão direito > Abrir).
# Instala o que faltar (Homebrew, Python 3.12, ffmpeg), as bibliotecas, baixa os modelos
# e cria o app "Proteger Vídeo" na Mesa (arraste o vídeo em cima dele).
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

passo() { printf "\n\033[1;34m==> %s\033[0m\n" "$1"; }
erro() { printf "\n\033[1;31mERRO: %s\033[0m\n" "$1"; read -n 1 -s -r -p "Aperte qualquer tecla para fechar."; exit 1; }

clear
echo "=============================================="
echo "   Cloaker de áudio · instalação no Mac"
echo "=============================================="
echo "São 6 passos e leva de 15 a 40 minutos (depende da internet). Pode pedir a senha do Mac."
echo "Não feche esta janela até aparecer INSTALADO."

# tira a "quarentena" do macOS dos arquivos baixados (senão ele bloqueia os scripts)
xattr -dr com.apple.quarantine "$DIR" 2>/dev/null || true
chmod +x "$DIR"/*.command "$DIR"/*.sh "$DIR"/scripts/*.sh 2>/dev/null || true

passo "1/6 Verificando o computador"
bash "$DIR/scripts/verificar-mac.sh"
nivel=$?
[ $nivel -eq 2 ] && erro "o computador não atende ao mínimo (veja os itens FALTA acima)"
if [ $nivel -eq 1 ]; then
  read -r -p "Há ATENÇÕES acima. Quer continuar mesmo assim? (s/n) " r
  case "$r" in s|S|sim|SIM) ;; *) exit 0 ;; esac
fi

passo "2/6 Homebrew (gerenciador de programas do Mac)"
if ! command -v brew >/dev/null 2>&1; then
  for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do [ -x "$b" ] && eval "$("$b" shellenv)"; done
fi
if ! command -v brew >/dev/null 2>&1; then
  echo "Instalando o Homebrew (gerenciador de programas do Mac)..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || erro "não deu pra instalar o Homebrew"
  for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do [ -x "$b" ] && eval "$("$b" shellenv)"; done
fi
command -v brew >/dev/null 2>&1 || erro "Homebrew não encontrado depois da instalação"
echo "ok: $(brew --version | head -1)"

passo "3/6 Python 3.12 e ffmpeg"
brew list python@3.12 >/dev/null 2>&1 || brew install python@3.12 || erro "falhou ao instalar o Python"
brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg || echo "aviso: ffmpeg do brew falhou; vai usar o embutido"
PY="$(brew --prefix python@3.12)/bin/python3.12"
[ -x "$PY" ] || erro "Python 3.12 não encontrado em $PY"
rm -f "$DIR/.expat"
if ! "$PY" -c "import pyexpat" >/dev/null 2>&1; then
  # defeito conhecido do Python do Homebrew em macOS novo: a biblioteca de XML do sistema é velha
  echo "Corrigindo a biblioteca de XML do Python (expat)..."
  brew list expat >/dev/null 2>&1 || brew install expat || erro "falhou ao instalar o expat"
  echo "$(brew --prefix expat)/lib" > "$DIR/.expat"
  export DYLD_LIBRARY_PATH="$(cat "$DIR/.expat")${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
  "$PY" -c "import pyexpat" >/dev/null 2>&1 || erro "o Python continua sem a biblioteca de XML (pyexpat)"
fi
echo "ok: $("$PY" --version)"

passo "4/6 Bibliotecas (PyTorch, Whisper...) · o maior download, pode parecer parado"
PYTHON="$PY" bash "$DIR/scripts/instalar-libs.sh" || erro "falhou ao instalar as bibliotecas"

passo "5/6 Baixando os modelos de IA (~5 GB, só desta vez)"
"$DIR/cloaker.sh" baixar || echo "aviso: alguns modelos serão baixados no 1º uso"

passo "6/6 Criando o app \"Proteger Vídeo\" na Mesa"
APP="$HOME/Desktop/Proteger Vídeo.app"
rm -rf "$APP"
cat > /tmp/cloaker_app.applescript <<EOF
on run
	set f to choose file with prompt "Escolha o vídeo para proteger" with multiple selections allowed
	proteger(f)
end run
on open f
	proteger(f)
end open
on proteger(lista)
	set cmd to "clear; "
	repeat with x in lista
		set cmd to cmd & quoted form of "$DIR/cloaker.sh" & " proteger " & quoted form of (POSIX path of x) & "; "
	end repeat
	set cmd to cmd & "echo; echo 'Pronto. Cada vídeo ganhou uma pasta NOME_protegido ao lado do original.'"
	tell application "Terminal"
		activate
		do script cmd
	end tell
end proteger
EOF
if osacompile -o "$APP" /tmp/cloaker_app.applescript 2>/dev/null; then
  echo "ok: $APP (arraste vídeos em cima dele)"
else
  echo "aviso: não deu pra criar o app; use o proteger.command desta pasta"
fi

passo "Conferindo"
"$DIR/cloaker.sh" diagnostico

echo
echo "=============================================="
echo "  INSTALADO. Para proteger um vídeo:"
echo "  - arraste o vídeo no app \"Proteger Vídeo\" da Mesa, ou"
echo "  - no Terminal: $DIR/cloaker.sh proteger video.mp4"
echo "=============================================="
read -n 1 -s -r -p "Aperte qualquer tecla para fechar."
