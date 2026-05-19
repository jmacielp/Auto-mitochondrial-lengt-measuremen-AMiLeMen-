@echo off
echo Iniciando Analizador de Mitocondrias...
python "%~dp0BACKUP.py"

if errorlevel 1 (
    echo.
    echo Ocurrio un error. Presiona cualquier tecla para ver el detalle.
    pause
)
