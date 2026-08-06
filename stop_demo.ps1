<#
Shuts down the photo booth demo: ComfyUI and the web server.

    powershell -ExecutionPolicy Bypass -File stop_demo.ps1
#>

$stopped = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match "main.py.*--port.*8188" -or $_.CommandLine -match "web_server\.py" }

if (-not $stopped) {
    Write-Host "nothing running."
    exit 0
}

$stopped | ForEach-Object {
    Write-Host "stopping PID $($_.ProcessId): $($_.CommandLine)"
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}

Write-Host "done."
