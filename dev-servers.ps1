#Requires -Version 5.1
# Dev servers launcher (backend FastAPI + frontend Vite). See also dev-servers.cmd.
# PIDs in .dev/*.pid; stop uses taskkill /T for child processes (uvicorn --reload, npm).

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('start', 'stop', 'status', 'help')]
    [string] $Action,

    [Parameter(Position = 1)]
    [ValidateSet('backend', 'frontend', 'all')]
    [string] $Target
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$StateDir = Join-Path $RepoRoot '.dev'
$BackendPidFile = Join-Path $StateDir 'backend.pid'
$FrontendPidFile = Join-Path $StateDir 'frontend.pid'

$BackendPort = 8000
$FrontendPort = 5173

function Ensure-StateDir {
    if (-not (Test-Path $StateDir)) {
        New-Item -ItemType Directory -Path $StateDir | Out-Null
    }
}

function Get-StoredPid {
    param([string] $Path)
    if (-not (Test-Path $Path)) { return $null }
    $raw = (Get-Content -Path $Path -Raw -ErrorAction SilentlyContinue).Trim()
    if ([string]::IsNullOrWhiteSpace($raw)) { return $null }
    $n = 0
    if (-not [int]::TryParse($raw, [ref]$n)) { return $null }
    return $n
}

function Test-PidRunning {
    param([int] $ProcessId)
    try {
        Get-Process -Id $ProcessId -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Remove-StalePidFile {
    param([string] $Path)
    $pidVal = Get-StoredPid -Path $Path
    if ($null -eq $pidVal) {
        if (Test-Path $Path) { Remove-Item $Path -Force -ErrorAction SilentlyContinue }
        return
    }
    if (-not (Test-PidRunning -ProcessId $pidVal)) {
        Remove-Item $Path -Force -ErrorAction SilentlyContinue
    }
}

function Stop-TrackedProcess {
    param(
        [string] $Name,
        [string] $PidFile
    )
    Remove-StalePidFile -Path $PidFile
    $pidVal = Get-StoredPid -Path $PidFile
    if ($null -eq $pidVal) {
        Write-Host "[$Name] No tracked process (not started by this script or already stopped)."
        return
    }
    if (-not (Test-PidRunning -ProcessId $pidVal)) {
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        Write-Host "[$Name] Stale PID file removed."
        return
    }
    & taskkill.exe /PID $pidVal /T /F 2>$null | Out-Null
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    Write-Host "[$Name] Stopped (PID $pidVal)."
}

function Start-Backend {
    Ensure-StateDir
    Remove-StalePidFile -Path $BackendPidFile
    $existing = Get-StoredPid -Path $BackendPidFile
    if ($null -ne $existing -and (Test-PidRunning -ProcessId $existing)) {
        Write-Host "[backend] Already running (PID $existing). Run: stop backend"
        return
    }

    $python = Join-Path $RepoRoot 'backend\venv\Scripts\python.exe'
    if (-not (Test-Path $python)) {
        Write-Error "Missing $python. Create venv: cd backend; python -m venv venv; pip install -r requirements.txt"
        return
    }

    $backendDir = Join-Path $RepoRoot 'backend'
    $proc = Start-Process -FilePath $python `
        -ArgumentList @('-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', "$BackendPort", '--reload') `
        -WorkingDirectory $backendDir `
        -WindowStyle Normal `
        -PassThru

    Set-Content -Path $BackendPidFile -Value $proc.Id -NoNewline
    Write-Host "[backend] Started PID=$($proc.Id)  http://localhost:$BackendPort/docs"
}

function Start-Frontend {
    Ensure-StateDir
    Remove-StalePidFile -Path $FrontendPidFile
    $existing = Get-StoredPid -Path $FrontendPidFile
    if ($null -ne $existing -and (Test-PidRunning -ProcessId $existing)) {
        Write-Host "[frontend] Already running (PID $existing). Run: stop frontend"
        return
    }

    $frontendDir = Join-Path $RepoRoot 'frontend'
    if (-not (Test-Path (Join-Path $frontendDir 'node_modules'))) {
        Write-Warning "frontend/node_modules not found. Run: cd frontend; npm install"
    }

    $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npm) { $npm = Get-Command npm -ErrorAction SilentlyContinue }
    if (-not $npm) {
        Write-Error "npm not found. Install Node.js and add it to PATH."
        return
    }
    $npmExe = $npm.Path
    if (-not $npmExe) { $npmExe = $npm.Definition }

    $proc = Start-Process -FilePath $npmExe `
        -ArgumentList @('run', 'dev') `
        -WorkingDirectory $frontendDir `
        -WindowStyle Normal `
        -PassThru

    Set-Content -Path $FrontendPidFile -Value $proc.Id -NoNewline
    Write-Host "[frontend] Started PID=$($proc.Id)  http://localhost:$FrontendPort"
}

function Show-Status {
    Remove-StalePidFile -Path $BackendPidFile
    Remove-StalePidFile -Path $FrontendPidFile

    $bp = Get-StoredPid -Path $BackendPidFile
    $fp = Get-StoredPid -Path $FrontendPidFile

    if ($null -ne $bp -and (Test-PidRunning -ProcessId $bp)) {
        Write-Host "[backend]  RUNNING  PID=$bp  http://localhost:$BackendPort"
    } else {
        Write-Host "[backend]  not running (or not started by this script)"
    }

    if ($null -ne $fp -and (Test-PidRunning -ProcessId $fp)) {
        Write-Host "[frontend] RUNNING  PID=$fp  http://localhost:$FrontendPort"
    } else {
        Write-Host "[frontend] not running (or not started by this script)"
    }
}

function Show-Help {
    Write-Host @'
Usage:
  .\dev-servers.ps1                    Interactive menu
  .\dev-servers.ps1 start backend|frontend|all
  .\dev-servers.ps1 stop  backend|frontend|all
  .\dev-servers.ps1 status

PIDs are stored under .dev\; stop uses taskkill /T (process tree).
'@
}

function Show-Menu {
    while ($true) {
        Write-Host ""
        Write-Host "========== DocAna dev servers =========="
        Write-Host "  1  Start backend"
        Write-Host "  2  Start frontend"
        Write-Host "  3  Start both"
        Write-Host "  4  Stop backend"
        Write-Host "  5  Stop frontend"
        Write-Host "  6  Stop both"
        Write-Host "  7  Status"
        Write-Host "  H  Help (CLI)"
        Write-Host "  0  Exit"
        Write-Host "=========================================="
        $c = Read-Host "Choice"
        switch ($c.Trim().ToLowerInvariant()) {
            '1' { Start-Backend }
            '2' { Start-Frontend }
            '3' { Start-Backend; Start-Frontend }
            '4' { Stop-TrackedProcess -Name 'backend' -PidFile $BackendPidFile }
            '5' { Stop-TrackedProcess -Name 'frontend' -PidFile $FrontendPidFile }
            '6' {
                Stop-TrackedProcess -Name 'backend' -PidFile $BackendPidFile
                Stop-TrackedProcess -Name 'frontend' -PidFile $FrontendPidFile
            }
            '7' { Show-Status }
            'h' { Show-Help }
            '0' { return }
            default { Write-Host "Invalid choice." }
        }
    }
}

if ($Action -eq 'help') {
    Show-Help
    exit 0
}

if (-not $Action) {
    Show-Menu
    exit 0
}

switch ($Action) {
    'status' { Show-Status }
    'start' {
        if (-not $Target) {
            Write-Error "Specify target: start backend | start frontend | start all"
            exit 1
        }
        switch ($Target) {
            'backend' { Start-Backend }
            'frontend' { Start-Frontend }
            'all' { Start-Backend; Start-Frontend }
        }
    }
    'stop' {
        if (-not $Target) {
            Write-Error "Specify target: stop backend | stop frontend | stop all"
            exit 1
        }
        switch ($Target) {
            'backend' { Stop-TrackedProcess -Name 'backend' -PidFile $BackendPidFile }
            'frontend' { Stop-TrackedProcess -Name 'frontend' -PidFile $FrontendPidFile }
            'all' {
                Stop-TrackedProcess -Name 'backend' -PidFile $BackendPidFile
                Stop-TrackedProcess -Name 'frontend' -PidFile $FrontendPidFile
            }
        }
    }
}
