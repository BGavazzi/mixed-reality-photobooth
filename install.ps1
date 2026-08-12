<#
One-command setup for the mixed-reality photo booth.

    powershell -ExecutionPolicy Bypass -File install.ps1
    powershell -ExecutionPolicy Bypass -File install.ps1 -Resolume   # + OSC/Spout bridge
    powershell -ExecutionPolicy Bypass -File install.ps1 -All        # + hosted backends

Creates .venv if it isn't there, installs the right requirement files, and
finishes by running doctor.py so you're told what's still missing rather than
finding out during a generation.

Downloads roughly 2GB (torch and friends come in via controlnet_aux) and takes
several minutes on a cold pip cache. It is safe to re-run: pip skips whatever
is already satisfied.
#>

param(
    [switch]$Resolume,
    [switch]$Backends,
    [switch]$All,
    [switch]$Dev
)

$ErrorActionPreference = "Stop"
$RepoDir = $PSScriptRoot
$VenvDir = Join-Path $RepoDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

function Write-Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }

# --- find an interpreter new enough -----------------------------------------
Write-Step "checking Python"
$python = $null
foreach ($candidate in @("py -3.12", "py -3.11", "py -3.10", "python")) {
    $parts = $candidate.Split(" ")
    $exe = $parts[0]
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
    try {
        $version = & $exe @($parts[1..($parts.Length - 1)]) -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    } catch { continue }
    if ($version -and [version]$version -ge [version]"3.10") {
        $python = $candidate
        Write-Host "  using $candidate (Python $version)"
        break
    }
}
if (-not $python) {
    Write-Host "No Python 3.10+ found on PATH." -ForegroundColor Red
    Write-Host "  Install it from https://www.python.org/downloads/ and re-run this script."
    exit 1
}

# --- venv --------------------------------------------------------------------
if (Test-Path $VenvPython) {
    Write-Step "reusing existing .venv"
} else {
    Write-Step "creating .venv"
    $parts = $python.Split(" ")
    & $parts[0] @($parts[1..($parts.Length - 1)]) -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Write-Host "venv creation failed" -ForegroundColor Red; exit 1 }
}

Write-Step "upgrading pip"
& $VenvPython -m pip install --upgrade pip --quiet

# --- requirements ------------------------------------------------------------
$files = @("requirements.txt")
if ($Resolume -or $All) { $files += "requirements-resolume.txt" }
if ($Backends -or $All) { $files += "requirements-backends.txt" }
if ($Dev -or $All)      { $files += "requirements-test.txt" }

foreach ($file in $files) {
    Write-Step "installing $file"
    & $VenvPython -m pip install -r (Join-Path $RepoDir $file)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`npip failed on $file." -ForegroundColor Red
        Write-Host "  Re-run this script to retry -- partial installs resume cleanly."
        exit 1
    }
}

# The GUI opencv build has to be the last writer to win on disk, because
# controlnet_aux depends on the headless build and both install the same `cv2`.
# Only spout_viewer.py needs the window, so this runs only when the bridge was
# actually requested -- the photo booth path never touches it.
if ($Resolume -or $All) {
    Write-Step "making the GUI opencv build the one on disk (for spout_viewer.py)"
    & $VenvPython -m pip install --force-reinstall --quiet opencv-python
}

# --- verify ------------------------------------------------------------------
Write-Step "checking the result"
$doctorArgs = @("doctor.py")
if ($Resolume -or $All) { $doctorArgs += "--resolume" }
if ($Backends -or $All) { $doctorArgs += "--backends" }
& $VenvPython @doctorArgs
$doctorExit = $LASTEXITCODE

Write-Host "`nActivate the environment with:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "`nThe photo booth also needs ComfyUI plus two model files -- doctor.py above"
Write-Host "says whether it can see them. See the README's Models section for links."

exit $doctorExit
