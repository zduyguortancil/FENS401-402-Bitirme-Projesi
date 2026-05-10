#!/bin/bash
# Her iki sunucuyu ayni anda baslat
cd "$(dirname "$0")"

echo "=============================="
echo "  SeatWise + BiletBul Launcher"
echo "=============================="
echo ""
echo "[1/2] Ana Dashboard baslatiliyor (port 5005)..."
python3 dashboard/app.py &
APP_PID=$!

echo "[*] Ana dashboard yukleniyor, 12 saniye bekleniyor..."
sleep 12

echo "[2/2] Musteri Arayuzu baslatiliyor (port 5006)..."
python3 dashboard/customer/run.py &
CUST_PID=$!

echo ""
echo "=============================="
echo "  Her iki sunucu calisiyor!"
echo "  Ana Dashboard : http://localhost:5005"
echo "  Musteri Arayuzu: http://localhost:5006"
echo "  Durdurmak icin: Ctrl+C"
echo "=============================="

# Her ikisi de kapaninca cik
wait $APP_PID $CUST_PID
