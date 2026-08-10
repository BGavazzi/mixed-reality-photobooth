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

# Derived, not hardcoded. These were absolute D:\vibes\... paths, so the
# script only ever worked in one checkout on one machine -- anyone cloning
# the repo (or moving it) got a silent failure to launch. $PSScriptRoot is
# wherever this file lives; ComfyUI is assumed to be a sibling directory,
# overridable with the COMFYUI_DIR environment variable.
$BridgeDir = $PSScriptRoot
$ComfyDir = if ($env:COMFYUI_DIR) { $env:COMFYUI_DIR } else { Join-Path (Split-Path $PSScriptRoot -Parent) "ComfyUI" }
$ComfyPython = Join-Path $ComfyDir "venv\Scripts\python.exe"
$BridgePython = Join-Path $BridgeDir ".venv\Scripts\python.exe"

foreach ($check in @(
    @{ Path = $ComfyDir;     What = "ComfyUI directory";     Hint = "set COMFYUI_DIR to where ComfyUI is checked out" }
    @{ Path = $ComfyPython;  What = "ComfyUI's venv python"; Hint = "create it with: python -m venv $ComfyDir\venv" }
    @{ Path = $BridgePython; What = "this repo's venv python"; Hint = "create it with: python -m venv $BridgeDir\.venv" }
)) {
    if (-not (Test-Path $check.Path)) {
        Write-Host "Cannot find $($check.What): $($check.Path)" -ForegroundColor Red
        Write-Host "  -> $($check.Hint)"
        exit 1
    }
}

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

# Pre-flight: confirm the models the workflow names are actually installed.
# Without this, a missing checkpoint doesn't surface until ~40 seconds into
# the first generation, as a red line in the browser's event log -- which is
# a bad thing to discover while someone is watching. ComfyUI's
# /object_info/<node> reports the exact enum of files it can see on disk, so
# this is the same list the node itself would validate against.
Write-Host "`n=== checking models named in the workflow ==="
$workflow = Get-Content (Join-Path $BridgeDir "workflows\photoshoot_bg_api.json") -Raw | ConvertFrom-Json
$missing = @()
foreach ($req in @(
    @{ Node = "CheckpointLoaderSimple"; Input = "ckpt_name";       Class = "CheckpointLoaderSimple"; Folder = "models\checkpoints" }
    @{ Node = "ControlNetLoader";       Input = "control_net_name"; Class = "ControlNetLoader";       Folder = "models\controlnet" }
)) {
    $wanted = $workflow.PSObject.Properties.Value |
        Where-Object { $_.class_type -eq $req.Class } |
        ForEach-Object { $_.inputs.($req.Input) } | Select-Object -First 1
    if (-not $wanted) { continue }
    try {
        $info = Invoke-RestMethod -Uri "http://127.0.0.1:8188/object_info/$($req.Node)" -TimeoutSec 10
        $available = $info.($req.Node).input.required.($req.Input)[0]
        if ($available -contains $wanted) {
            Write-Host "  ok: $wanted"
        } else {
            $missing += "$wanted  (expected in $($ComfyDir)\$($req.Folder))"
        }
    } catch {
        Write-Host "  (could not query /object_info/$($req.Node): $_)" -ForegroundColor DarkYellow
    }
}
if ($missing) {
    Write-Host "`nMISSING MODELS -- generation will fail partway through:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    Write-Host "See the README's 'Models' section for download links."
    Write-Host "Continuing anyway; the analyze/rotoscope/pose/depth stages still work.`n" -ForegroundColor DarkYellow
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
