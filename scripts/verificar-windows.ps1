# Verifica se este computador Windows roda o Cloaker de áudio.
# Saída: 0 = pronto, 1 = funciona com ressalvas, 2 = não dá pra instalar.
param([switch]$Silencioso)

$ErrorActionPreference = "SilentlyContinue"
$raiz = Split-Path -Parent $PSScriptRoot
$nivel = 0
function Ok($t)    { Write-Host "  [OK]      $t" -ForegroundColor Green }
function Aviso($t) { Write-Host "  [ATENCAO] $t" -ForegroundColor Yellow; if ($script:nivel -lt 1) { $script:nivel = 1 } }
function Falha($t) { Write-Host "  [FALTA]   $t" -ForegroundColor Red; $script:nivel = 2 }

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "   Cloaker de áudio · verificando o computador" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Sistema
$os = Get-CimInstance Win32_OperatingSystem
$arq = $env:PROCESSOR_ARCHITECTURE
if ([Environment]::Is64BitOperatingSystem) { Ok "Sistema: $($os.Caption) 64 bits ($arq)" }
else { Falha "Sistema de 32 bits: precisa de Windows 64 bits" }
if ($arq -eq "ARM64") { Aviso "Processador ARM: funciona só no modo lento (sem placa de vídeo NVIDIA)" }

# Memória RAM
$ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
if ($ram -ge 16) { Ok "Memória RAM: $ram GB" }
elseif ($ram -ge 8) { Aviso "Memória RAM: $ram GB (funciona; feche o navegador e outros programas enquanto protege)" }
else { Falha "Memória RAM: $ram GB (mínimo 8 GB)" }

# Espaço em disco (na unidade da pasta)
$letra = (Get-Item $raiz).PSDrive.Name
$livre = [math]::Round((Get-PSDrive $letra).Free / 1GB, 1)
if ($livre -ge 20) { Ok "Espaço livre no disco ${letra}: $livre GB" }
elseif ($livre -ge 12) { Aviso "Espaço livre no disco ${letra}: $livre GB (justo; recomendado 20 GB)" }
else { Falha "Espaço livre no disco ${letra}: $livre GB (precisa de pelo menos 12 GB: programas + modelos de IA)" }

# Placa de vídeo
$gpus = Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }
$nvidia = $false
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $info = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if ($info) {
        $p = $info.Split(",") | ForEach-Object { $_.Trim() }
        $vram = [math]::Round([double]$p[1] / 1024, 1)
        $nvidia = $true
        if ($vram -ge 6) { Ok "Placa de vídeo: $($p[0]) com $vram GB (driver $($p[2])) · rápido (~20 min por vídeo de 1 min)" }
        else { Aviso "Placa de vídeo: $($p[0]) com só $vram GB · use o perfil rápido (--perfil rapido)" }
    }
}
if (-not $nvidia) {
    Aviso "Sem placa NVIDIA ($($gpus -join ', ')). Funciona pelo processador, mas é MUITO lento (horas por vídeo)."
}

# Python
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
if ($py) { Ok "Python: $py" }
elseif (Get-Command winget -ErrorAction SilentlyContinue) { Ok "Python: ainda não tem; o instalador vai instalar sozinho (via winget)" }
else { Falha "Python 3.12 não encontrado e sem winget: instale em https://www.python.org/downloads/ (marque 'Add to PATH')" }

# ffmpeg (opcional)
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) { Ok "ffmpeg: instalado" }
else { Ok "ffmpeg: vai usar o que vem junto com as bibliotecas" }

# Internet
try {
    $r = Invoke-WebRequest -Uri "https://pypi.org/simple/pip/" -Method Head -UseBasicParsing -TimeoutSec 10
    Ok "Internet: conectado (precisa baixar ~8 GB na instalação)"
} catch { Falha "Sem acesso à internet (pypi.org). Precisa de internet pra instalar." }

# Já instalado?
if (Test-Path "$raiz\.venv\Scripts\python.exe") { Ok "Cloaker: já instalado nesta pasta (pode reinstalar por cima)" }

Write-Host ""
switch ($nivel) {
    0 { Write-Host "  RESULTADO: computador pronto. Pode instalar." -ForegroundColor Green }
    1 { Write-Host "  RESULTADO: dá pra instalar, com as ATENÇÕES acima." -ForegroundColor Yellow }
    2 { Write-Host "  RESULTADO: falta algo (itens FALTA acima). Resolva antes de instalar." -ForegroundColor Red }
}
Write-Host ""
exit $nivel
