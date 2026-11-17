@echo off
chcp 65001 >nul
REM Abre o painel do Cloaker no navegador (arrastar video, fila, relatorio).
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo O Cloaker ainda nao esta instalado. Rode primeiro o "2 - Instalar no Windows.bat".
  pause
  exit /b 1
)
echo Deixe esta janela aberta enquanto usa o painel. Fechar a janela desliga o painel.
"%~dp0.venv\Scripts\python.exe" "%~dp0painel\painel.py" --pasta "%~dp0."
pause
