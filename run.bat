@echo off
title Automated Vehicle Trafficability Prediction System
cd /d "%~dp0"

echo =======================================================================
echo  AUTOMATED VEHICLE TRAFFICABILITY PREDICTION ^& MAPPING SYSTEM
echo  Sapta Shakti Command - Geospatial Intelligence ^& Defense-AI
echo =======================================================================
echo.

:: Detect free port
for /f "delims=" %%p in ('python -c "import socket; print(next(p for p in range(8501, 8520) if socket.socket().connect_ex(('127.0.0.1', p)) != 0))"') do set FREE_PORT=%%p

if "%FREE_PORT%"=="" set FREE_PORT=8505

echo [INFO] Active Port: %FREE_PORT%
echo [INFO] Starting tactical mobility dashboard in your default browser...
echo.

start "" "http://localhost:%FREE_PORT%"
python -m streamlit run app.py --server.port %FREE_PORT% --server.headless true

pause
