# Automated Vehicle Trafficability System PowerShell Launcher
Set-Location -Path $PSScriptRoot

Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host " AUTOMATED VEHICLE TRAFFICABILITY PREDICTION & MAPPING SYSTEM" -ForegroundColor Green
Write-Host " Sapta Shakti Command - Geospatial Intelligence & Defense-AI" -ForegroundColor Cyan
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host ""

# Find available port
$freePort = 8501
for ($p = 8501; $p -le 8520; $p++) {
    $tcp = New-Object System.Net.Sockets.TcpClient
    try {
        $tcp.Connect("127.0.0.1", $p)
        $tcp.Close()
    } catch {
        $freePort = $p
        break
    }
}

Write-Host "[INFO] Free Port Detected: $freePort" -ForegroundColor Yellow
Write-Host "[INFO] Launching dashboard in browser: http://localhost:$freePort" -ForegroundColor Green
Write-Host ""

Start-Process "http://localhost:$freePort"
python -m streamlit run app.py --server.port $freePort --server.headless true
