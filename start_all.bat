@echo off
REM Seatwise + BiletBul -- her iki sunucuyu paralel baslat
setlocal
cd /d "%~dp0"

echo ==============================
echo   SeatWise + BiletBul Launcher
echo ==============================
echo.
echo [1/2] Ana Dashboard baslatiliyor (port 5005)...
start "Seatwise Main 5005" cmd /k "cd /d %~dp0dashboard && python app.py"

echo [*] Ana dashboard yukleniyor, 8 saniye bekleniyor...
timeout /t 8 /nobreak >nul

echo [2/2] Musteri Arayuzu baslatiliyor (port 5006)...
start "BiletBul Customer 5006" cmd /k "cd /d %~dp0dashboard && python customer\run.py"

echo.
echo ==============================
echo   Her iki sunucu calisiyor!
echo   Ana Dashboard  : http://localhost:5005
echo   Musteri Arayuzu: http://localhost:5006
echo   Pencereleri kapatarak durdurabilirsin
echo ==============================
echo.
pause
