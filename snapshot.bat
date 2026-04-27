@echo off
echo ================================
echo SNAPSHOT DEL PROYECTO
echo ================================

echo Fecha: %date% %time%
echo.

IF NOT EXIST ".venv\Scripts\activate" (
    echo [ERROR] Entorno virtual no encontrado.
    pause
    exit /b 1
)

call .venv\Scripts\activate

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] No se pudo activar el entorno virtual.
    pause
    exit /b 1
)

python scripts\context_snapshot.py

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo al generar snapshot.
    pause
    exit /b 1
)

echo.
echo [OK] Snapshot generado correctamente.
pause