@echo off
chcp 65001 >nul
REM Abre o painel do Cloaker no navegador (arrastar video, fila, relatorio).
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo O Cloaker ainda nao esta instalado. Rode primeiro o "2 - Instalar no Windows.bat".
  pause
  exit /b 1
)
echo Fechar esta janela desliga o painel. Uma protecao em andamento continua e volta a aparecer quando abrir de novo.
"%~dp0.venv\Scripts\python.exe" "%~dp0painel\painel.py" --pasta "%~dp0."
pause
