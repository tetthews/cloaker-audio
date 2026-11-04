@echo off
chcp 65001 >nul
rem Arraste um ou mais vídeos em cima deste arquivo.
if "%~1"=="" (
  echo Arraste um ou mais videos em cima deste arquivo para proteger.
  pause
  exit /b 0
)
:loop
if "%~1"=="" goto fim
echo.
echo === Protegendo %~nx1 ===
"%~dp0.venv\Scripts\python.exe" -m cloaker proteger "%~1"
shift
goto loop
:fim
echo.
echo Pronto. Cada video ganhou uma pasta NOME_protegido ao lado do original, com o relatorio.html.
pause
