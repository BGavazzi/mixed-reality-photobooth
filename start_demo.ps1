<#
One-click startup for the mixed-reality photo booth demo.
Starts ComfyUI, waits for it to be ready, starts the web server, waits for
it to be ready, then opens the browser. Run stop_demo.ps1 to shut down.

    powershell -ExecutionPolicy Bypass -File start_demo.ps1

Note: while building this, running it through a sandboxed/automated shell
occasionally lost the spawned processes when the wrapping shell exited
(a Windows job-object quirk). It ran cleanly every time from a normal
interactive PowerShell window. If double-clicking / "Run with PowerShell"
ever doesn't leave the processes running, open a PowerShell window and run
this script directly instead -- that's the one usage path that was fully
verified.
#>

$ErrorActionPreference = "Stop"

$ComfyDir = "D:\vibes\ComfyUI"
$ComfyPython = "$ComfyDir\venv\Scripts\python.exe"
$BridgeDir = "D:\vibes\resolume-genai-bridge"
$BridgePython = "$BridgeDir\.venv\Scripts\python.exe"

function Wait-ForHttp($url, $label, $timeoutSec = 90) {
    Write-Host "waiting for $label ($url)..." -NoNewline
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
            if ($resp.StatusCode -eq 200) { Write-Host " up." ; return $true }
        } catch {}
        Start-Sleep -Seconds 1
        Write-Host "." -NoNewline
    }
    Write-Host " TIMED OUT."
    return $false
}

Write-Host "=== stopping any existing instances first ==="
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match "main.py.*--port.*8188" -or $_.CommandLine -match "web_server\.py" }
$existing | ForEach-Object {
    Write-Host "  stopping PID $($_.ProcessId)"
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}
if ($existing) {
    Start-Sleep -Seconds 2
}

Write-Host "`n=== starting ComfyUI ==="
Start-Process -FilePath $ComfyPython `
    -ArgumentList "main.py","--listen","127.0.0.1","--port","8188","--preview-method","auto" `
    -WorkingDirectory $ComfyDir -WindowStyle Hidden `
    -RedirectStandardOutput "$ComfyDir\comfy_stdout.log" `
    -RedirectStandardError "$ComfyDir\comfy_stderr.log"

if (-not (Wait-ForHttp "http://127.0.0.1:8188/system_stats" "ComfyUI")) {
    Write-Host "ComfyUI did not come up - check $ComfyDir\comfy_stderr.log"
    exit 1
}

Write-Host "`n=== starting web server ==="
Start-Process -FilePath $BridgePython `
    -ArgumentList "web_server.py" `
    -WorkingDirectory $BridgeDir -WindowStyle Hidden `
    -RedirectStandardOutput "$BridgeDir\web_stdout.log" `
    -RedirectStandardError "$BridgeDir\web_stderr.log"

if (-not (Wait-ForHttp "http://127.0.0.1:8000/" "web server")) {
    Write-Host "web server did not come up - check $BridgeDir\web_stderr.log"
    exit 1
}

Write-Host "`n=== opening browser ==="
Start-Process "http://127.0.0.1:8000"

Write-Host "`nReady. Run stop_demo.ps1 when done."
