@echo off
echo ==========================================
echo    HFT BACKTESTER - BOTON DE PANICO
echo ==========================================
echo Matando todos los procesos de Python...
taskkill /f /t /im python.exe
taskkill /f /t /im python3.11.exe
taskkill /f /t /im python3.exe
echo.
echo Limpiando memoria compartida (SharedMemory)...
echo Si ves errores arriba es normal, significa que ya estaba limpio.
echo.
echo SISTEMA LIMPIO. Ya puedes volver a trabajar.
pause
