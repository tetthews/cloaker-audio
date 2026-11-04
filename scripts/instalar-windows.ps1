# Instalador passo a passo do Cloaker de áudio no Windows.
$ErrorActionPreference = "Continue"
$raiz = Split-Path -Parent $PSScriptRoot
Set-Location $raiz
$total = 6

function Passo($n, $t, $explica) {
    Write-Host ""
    Write-Host "---------------------------------------------" -ForegroundColor Cyan
    Write-Host " PASSO $n de ${total}: $t" -ForegroundColor Cyan
    Write-Host "---------------------------------------------" -ForegroundColor Cyan
    if ($explica) { Write-Host " $explica" -ForegroundColor Gray }
}
function Parar($t) {
    Write-Host ""
    Write-Host " ERRO: $t" -ForegroundColor Red
    Write-Host " Tire um print desta janela e mande pra quem te passou o programa." -ForegroundColor Red
    Read-Host " Aperte ENTER para fechar"
    exit 1
}
function Rodar($exe, [string[]]$argumentos, $msg) {
    & $exe @argumentos
    if ($LASTEXITCODE -ne 0) { Parar $msg }
}

Clear-Host
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "   Cloaker de áudio · instalação no Windows" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " São 6 passos e leva de 15 a 40 minutos (depende da internet)."
Write-Host " Não feche esta janela até aparecer 'INSTALADO'."

# 1. Verificação
Passo 1 "Verificando o computador" "Memória, disco, placa de vídeo, Python e internet."
& powershell -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\verificar-windows.ps1"
$nivel = $LASTEXITCODE
if ($nivel -eq 2) { Parar "o computador não atende ao mínimo (veja os itens FALTA acima)" }
if ($nivel -eq 1) {
    $r = Read-Host " Há ATENÇÕES acima. Quer continuar mesmo assim? (S/N)"
    if ($r -notmatch "^[sS]") { exit 0 }
}

# 2. Python
Passo 2 "Python 3.12" "É a linguagem em que o programa foi feito."
$py = $null
foreach ($c in @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe", "$env:ProgramFiles\Python312\python.exe")) {
    if (Test-Path $c) { $py = $c; break }
}
if (-not $py) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notlike "*WindowsApps*") {
        $v = & $cmd.Source -c "import sys;print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($v -in @("3.10", "3.11", "3.12")) { $py = $cmd.Source }
    }
}
if (-not $py) {
    Write-Host " Instalando o Python 3.12 (pode aparecer uma janela pedindo permissão: clique em Sim)..."
    winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
    foreach ($c in @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe", "$env:ProgramFiles\Python312\python.exe")) {
        if (Test-Path $c) { $py = $c; break }
    }
    if (-not $py) { Parar "não consegui instalar o Python. Instale manualmente em https://www.python.org/downloads/ e rode o instalador de novo." }
}
Write-Host " Python: $py" -ForegroundColor Green

# 3. Ambiente + PyTorch
Passo 3 "Motor de IA (PyTorch)" "O maior download (~2,5 GB). A barra pode ficar parada um tempo: é normal."
if (-not (Test-Path "$raiz\.venv\Scripts\python.exe")) { Rodar $py @("-m", "venv", "$raiz\.venv") "não consegui criar o ambiente (.venv)" }
$vpy = "$raiz\.venv\Scripts\python.exe"
Rodar $vpy @("-m", "pip", "install", "--upgrade", "pip", "--quiet") "falha ao atualizar o pip"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host " Placa NVIDIA encontrada: versão com aceleração (CUDA)."
    Rodar $vpy @("-m", "pip", "install", "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu126") "falha ao instalar o PyTorch"
} else {
    Write-Host " Sem NVIDIA: versão para processador (vai ser lento)."
    Rodar $vpy @("-m", "pip", "install", "torch", "torchaudio") "falha ao instalar o PyTorch"
}

# 4. Bibliotecas
Passo 4 "Bibliotecas do programa" "Whisper, ffmpeg e o resto."
Rodar $vpy @("-m", "pip", "install", "-r", "$raiz\requirements.txt") "falha ao instalar as bibliotecas"

# 5. Modelos
Passo 5 "Modelos de IA (~5 GB, só desta vez)" "São as IAs de transcrição que o escudo aprende a enganar."
& $vpy -m cloaker baixar
if ($LASTEXITCODE -ne 0) { Write-Host " Alguns modelos não baixaram agora; eles serão baixados no primeiro uso." -ForegroundColor Yellow }

# 6. Atalho + teste
Passo 6 "Atalho na Área de Trabalho e teste final" ""
try {
    $mesa = [Environment]::GetFolderPath("Desktop")
    $s = (New-Object -ComObject WScript.Shell).CreateShortcut("$mesa\Proteger Video.lnk")
    $s.TargetPath = "$raiz\Proteger video (arraste aqui).bat"
    $s.WorkingDirectory = $raiz
    $s.IconLocation = "$env:SystemRoot\System32\imageres.dll,77"
    $s.Description = "Arraste um vídeo aqui para proteger a fala contra transcrição por IA"
    $s.Save()
    Write-Host " Atalho criado: 'Proteger Video' na Área de Trabalho." -ForegroundColor Green
} catch { Write-Host " Não consegui criar o atalho; use o arquivo 'Proteger video (arraste aqui).bat' desta pasta." -ForegroundColor Yellow }
& $vpy -m cloaker diagnostico

Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host "   INSTALADO" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host " Como usar:"
Write-Host "  1. Arraste o vídeo em cima do atalho 'Proteger Video' da Área de Trabalho."
Write-Host "  2. Espere terminar (uns 20 min por minuto de vídeo com placa NVIDIA)."
Write-Host "  3. Ao lado do vídeo aparece a pasta NOME_protegido: poste o arquivo NOME_protegido."
Write-Host "     O relatorio.html mostra o que a IA consegue ler antes e depois."
Write-Host ""
Read-Host " Aperte ENTER para fechar"
