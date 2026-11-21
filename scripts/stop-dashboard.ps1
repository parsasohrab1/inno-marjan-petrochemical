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

Write-Host "Stopping MRTO background jobs..." -ForegroundColor Cyan

# Stop jobs started by start-dashboard.ps1
$jobs = Get-Job -Name 'mrto-*' -ErrorAction SilentlyContinue
if ($jobs) {
    $jobs | Stop-Job -ErrorAction SilentlyContinue
    $jobs | Remove-Job -ErrorAction SilentlyContinue
    Write-OK "Stopped $($jobs.Count) job(s)"
} else {
    Write-Warn "No mrto-* jobs found in this session"
}

# Free the ports (in case they're held by orphaned python processes)
foreach ($port in @($BackendPort, $DashboardPort)) {
    $conns = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        try {
            Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
            Write-OK "Killed PID $($c.OwningProcess) on port $port"
        } catch {
            Write-Warn "Could not kill PID $($c.OwningProcess) on port $port"
        }
    }
}

Write-Host "`n✓ Done. Ports $BackendPort and $DashboardPort are free." -ForegroundColor Green
