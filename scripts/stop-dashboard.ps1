<#
.SYNOPSIS
    Stop the MRTO dashboard and backend (companion to start-dashboard.ps1).

.DESCRIPTION
    Stops background Python jobs and frees ports 8000 / 8080.

.PARAMETER BackendPort
    Port to free for the backend (default 8000).

.PARAMETER DashboardPort
    Port to free for the dashboard (default 8080).

.EXAMPLE
    .\scripts\stop-dashboard.ps1
    .\scripts\stop-dashboard.ps1 -BackendPort 9000 -DashboardPort 9001
#>
[CmdletBinding()]
param(
    [int] $BackendPort   = 8000,
    [int] $DashboardPort = 8080
)

$ErrorActionPreference = 'SilentlyContinue'

function Write-OK   { param($m) Write-Host "  [ok]  $m" -ForegroundColor Green }
function Write-Warn { param($m) Write-Host "  [warn] $m" -ForegroundColor Yellow }

Write-Host "Stopping MRTO dashboard processes..." -ForegroundColor Cyan

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PidFile = Join-Path $RepoRoot 'logs\mrto.pids.json'
if (Test-Path $PidFile) {
    try {
        $pids = Get-Content $PidFile -Raw | ConvertFrom-Json
        foreach ($procId in @($pids.backend, $pids.dashboard)) {
            if ($procId) {
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                Write-OK "Stopped PID $procId"
            }
        }
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    } catch {
        Write-Warn "Could not read $PidFile"
    }
}

# Stop jobs started by older launcher versions in this session
$jobs = Get-Job -Name 'mrto-*' -ErrorAction SilentlyContinue
if ($jobs) {
    $jobs | Stop-Job -ErrorAction SilentlyContinue
    $jobs | Remove-Job -ErrorAction SilentlyContinue
    Write-OK "Stopped $($jobs.Count) job(s)"
}

# Free only listeners we started (python/uvicorn/http.server). Never kill
# unrelated services that happen to use the same port numbers.
foreach ($port in @($BackendPort, $DashboardPort)) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        $proc = Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue
        if (-not $proc) { continue }
        $owned = $proc.ProcessName -match 'python|uvicorn'
        if ($owned) {
            Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
            Write-OK "Killed $($proc.ProcessName) PID $($c.OwningProcess) on port $port"
        }
    }
}

Write-Host "`n✓ Done. Ports $BackendPort and $DashboardPort are free." -ForegroundColor Green
