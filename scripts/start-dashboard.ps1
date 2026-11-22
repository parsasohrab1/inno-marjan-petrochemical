<#
.SYNOPSIS
    MRTO — Methanol Reformer Tube & Operation Optimizer
    One-shot launcher that brings up the full stack on Windows / PowerShell.

.DESCRIPTION
    This script performs the COMPLETE bring-up of the MRTO dashboard:
      1. Verifies prerequisites (Python 3.10+)
      2. Creates a Python virtual environment (.venv) if missing
      3. Installs backend dependencies + pyjwt + bcrypt
      4. Regenerates synthetic data (4 modules) if data files are missing
      5. Trains all 4 ML modules if model artifacts are missing
      6. Starts FastAPI backend on http://localhost:8000  (background job)
      7. Starts static dashboard server on http://localhost:8080  (background job)
      8. Waits for /api/v1/health to return 200
      9. Seeds the live SQLite DB with realistic demo data:
           - 720 temperature readings (12 tubes × 60 minutes)
           - 240 burner status readings (8 burners × 30 minutes)
         Runs one /api/v1/optimization call so Module 3 + 4 light up.
     10. Opens the dashboard in the default browser
     11. Prints a summary with all URLs and the admin credentials

.PARAMETER NoBrowser
    Skip opening the browser (useful for headless / CI runs).

.PARAMETER NoSeed
    Skip seeding demo data (assumes DB already populated).

.PARAMETER BackendPort
    Port for the FastAPI backend (default 8000).

.PARAMETER DashboardPort
    Port for the static dashboard server (default 8080).

.PARAMETER Reset
    Delete .venv and re-install dependencies from scratch.

.EXAMPLE
    .\scripts\start-dashboard.ps1
    # Brings up the full stack and opens the browser.

.EXAMPLE
    .\scripts\start-dashboard.ps1 -NoBrowser -BackendPort 9000
    # Headless launch on custom ports.

.EXAMPLE
    .\scripts\start-dashboard.ps1 -Reset
    # Rebuild venv and reinstall deps.

.NOTES
    Repository: https://github.com/parsasohrab1/inno-marjan-petrochemical-
    Admin login: admin / admin123  (change in production via env MRTO_ADMIN_PASS)
#>

[CmdletBinding()]
param(
    [switch] $NoBrowser,
    [switch] $NoSeed,
    [switch] $Reset,
    [int]    $BackendPort    = 8000,
    [int]    $DashboardPort  = 8080
)

# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap: strict error handling, colours, repo root
# ─────────────────────────────────────────────────────────────────────────────
$ErrorActionPreference = 'Stop'

function Write-Step  { param($m) Write-Host "`n[step] $m" -ForegroundColor Cyan }
function Write-OK    { param($m) Write-Host "  [ok]  $m" -ForegroundColor Green }
function Write-Warn  { param($m) Write-Host "  [warn] $m" -ForegroundColor Yellow }
function Write-Err   { param($m) Write-Host "  [err]  $m" -ForegroundColor Red }

# Resolve repo root (parent of this script's directory)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
$BackendDir  = Join-Path $RepoRoot 'backend'
$FrontendDir = Join-Path $RepoRoot 'frontend'
$ModelsDir   = Join-Path $RepoRoot 'models'
$DataRawDir  = Join-Path (Join-Path $RepoRoot 'data') 'raw'
$VenvDir     = Join-Path $RepoRoot '.venv'

Write-Host "=====================================================" -ForegroundColor White
Write-Host " MRTO — Dashboard launcher (TRL 5)" -ForegroundColor White
Write-Host " Marjan Petrochemical — Methanol Reformer Optimizer" -ForegroundColor White
Write-Host "=====================================================" -ForegroundColor White
Write-Host " Repo root: $RepoRoot"
Write-Host " Backend:   http://localhost:$BackendPort   (FastAPI)"
Write-Host " Dashboard: http://localhost:$DashboardPort   (static UI)"
if (-not $NoBrowser) { Write-Host " Browser:   will open automatically" }
Write-Host "-----------------------------------------------------"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Prerequisites
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Checking prerequisites"

# python / python3 / py launcher
$pythonCmd = $null
foreach ($c in @('python', 'python3', 'py')) {
    try {
        $ver = & $c --version 2>$null
        if ($LASTEXITCODE -eq 0 -and $ver -match '3\.(\d+)') {
            $minor = [int]$Matches[1]
            if ($minor -ge 10) {
                $pythonCmd = $c
                Write-OK "Found $ver via '$c'"
                break
            }
        }
    } catch { }
}
if (-not $pythonCmd) {
    Write-Err "Python 3.10+ not found. Install from https://python.org and re-run."
    exit 1
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Virtual environment
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Preparing Python virtual environment"

if ($Reset -and (Test-Path $VenvDir)) {
    Write-Warn "Removing existing .venv (-Reset)"
    Remove-Item -Recurse -Force $VenvDir
}

if (-not (Test-Path $VenvDir)) {
    & $pythonCmd -m venv $VenvDir
    Write-OK "Created .venv"
} else {
    Write-OK ".venv already exists"
}

# Activate for this session
$VenvPython = Join-Path (Join-Path $VenvDir 'Scripts') 'python.exe'
if (-not (Test-Path $VenvPython)) {
    Write-Err "Virtual env python not found at $VenvPython"
    exit 1
}
& $VenvPython -m pip install --upgrade pip --quiet
Write-OK "pip upgraded"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Install dependencies
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Installing backend dependencies"

$ReqFile = Join-Path $BackendDir 'requirements.txt'
& $VenvPython -m pip install -r $ReqFile --quiet
& $VenvPython -m pip install pyjwt bcrypt --quiet
Write-OK "fastapi, uvicorn, sklearn, pandas, numpy, joblib, pyjwt, bcrypt installed"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Regenerate synthetic data if missing
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Checking synthetic data"

$expectedData = @(
    (Join-Path $DataRawDir 'synthetic_tubes.csv'),
    (Join-Path $DataRawDir 'synthetic_timeseries.csv'),
    (Join-Path $DataRawDir 'synthetic_temp_map.csv'),
    (Join-Path $DataRawDir 'synthetic_rul.csv'),
    (Join-Path $DataRawDir 'synthetic_sc_opt.csv'),
    (Join-Path $DataRawDir 'synthetic_burner_profiles.csv'),
    (Join-Path $DataRawDir 'synthetic_burner_status.csv'),
    (Join-Path $DataRawDir 'synthetic_catalyst.csv')
)
$missingData = $expectedData | Where-Object { -not (Test-Path $_) }

if ($missingData.Count -gt 0) {
    Write-Warn "Missing $($missingData.Count) data file(s) — regenerating"
    Push-Location $BackendDir
    try {
        & $VenvPython -m mrto.synthetic_data $DataRawDir
        Write-OK "Synthetic data regenerated"
    } finally {
        Pop-Location
    }
} else {
    Write-OK "All synthetic data files present"
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. Train models if missing
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Checking trained models"

$expectedModels = @(
    (Join-Path $ModelsDir 'module1_temp_map.joblib'),
    (Join-Path $ModelsDir 'module2_rul.joblib'),
    (Join-Path $ModelsDir 'module3_optimization.joblib'),
    (Join-Path $ModelsDir 'module4_burner_catalyst.joblib')
)
$missingModels = $expectedModels | Where-Object { -not (Test-Path $_) }

if ($missingModels.Count -gt 0) {
    Write-Warn "Missing $($missingModels.Count) model(s) — training (may take 1-2 min)"
    Push-Location $BackendDir
    try {
        & $VenvPython -m mrto.train_models `
            --out-dir  (Join-Path $RepoRoot 'data') `
            --models-dir $ModelsDir `
            --report   (Join-Path $RepoRoot 'training_report.json')
        Write-OK "Models trained"
    } finally {
        Pop-Location
    }
} else {
    Write-OK "All 4 models present"
}

# ─────────────────────────────────────────────────────────────────────────────
# 6. Environment for backend
# ─────────────────────────────────────────────────────────────────────────────
$DbPath = Join-Path (Join-Path $BackendDir 'data') 'mrto.sqlite3'
$DbDir = Split-Path -Parent $DbPath
if (-not (Test-Path $DbDir)) { New-Item -ItemType Directory -Path $DbDir -Force | Out-Null }

$backendEnv = @{
    MRTO_DB_PATH     = $DbPath
    MRTO_MODELS_DIR  = $ModelsDir
    MRTO_JWT_SECRET  = 'live-demo-secret-32chars-1234567890'
    MRTO_ADMIN_USER  = 'admin'
    MRTO_ADMIN_PASS  = 'admin123'
    MRTO_LIVE_SIM    = '1'
    PYTHONUNBUFFERED = '1'
}
$LogDir = Join-Path $RepoRoot 'logs'
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
$PidFile = Join-Path $LogDir 'mrto.pids.json'

# ─────────────────────────────────────────────────────────────────────────────
# 7. Stop a previous MRTO launch, then pick ports that are actually free.
#    Do not kill unrelated listeners (local Apache/IIS may already own 8000/8080).
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Selecting free ports"

if (Test-Path $PidFile) {
    try {
        $old = Get-Content $PidFile -Raw | ConvertFrom-Json
        foreach ($procId in @($old.backend, $old.dashboard)) {
            if ($procId) {
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                Write-Warn "Stopped previous MRTO PID $procId"
            }
        }
    } catch { }
    Start-Sleep -Seconds 1
}

function Test-PortFree([int] $Port) {
    $listener = $null
    try {
        $listener = New-Object System.Net.Sockets.TcpListener ([System.Net.IPAddress]::Loopback, $Port)
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($listener) { $listener.Stop() }
    }
}
function Find-FreePort([int] $Preferred) {
    for ($p = $Preferred; $p -lt ($Preferred + 30); $p++) {
        if (Test-PortFree $p) { return $p }
    }
    throw "No free TCP port found near $Preferred"
}

$BackendPort = Find-FreePort $BackendPort
$DashboardPort = Find-FreePort $DashboardPort
if ($BackendPort -eq $DashboardPort) { $DashboardPort = Find-FreePort ($DashboardPort + 1) }
Write-OK "Backend port $BackendPort, dashboard port $DashboardPort"

$apiConfig = "window.MRTO_API = 'http://127.0.0.1:$BackendPort';"
Set-Content -Path (Join-Path $FrontendDir 'api-config.js') -Value $apiConfig -Encoding ascii

# ─────────────────────────────────────────────────────────────────────────────
# 8. Start FastAPI backend (background job)
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Starting FastAPI backend on port $BackendPort"

# Start-Process inherits this process environment. Do not use Start-Job:
# jobs do not keep the backend working directory, so `mrto` fails to import.
$env:MRTO_DB_PATH     = $backendEnv.MRTO_DB_PATH
$env:MRTO_MODELS_DIR  = $backendEnv.MRTO_MODELS_DIR
$env:MRTO_JWT_SECRET  = $backendEnv.MRTO_JWT_SECRET
$env:MRTO_ADMIN_USER  = $backendEnv.MRTO_ADMIN_USER
$env:MRTO_ADMIN_PASS  = $backendEnv.MRTO_ADMIN_PASS
$env:MRTO_LIVE_SIM    = $backendEnv.MRTO_LIVE_SIM
$env:PYTHONUNBUFFERED = '1'

$backendOut = Join-Path $LogDir 'backend.out.log'
$backendErr = Join-Path $LogDir 'backend.err.log'
Remove-Item $backendOut, $backendErr -ErrorAction SilentlyContinue
$backendProc = Start-Process -FilePath $VenvPython `
    -ArgumentList @('-m', 'uvicorn', 'mrto.app:app', '--host', '0.0.0.0', '--port', "$BackendPort") `
    -WorkingDirectory $BackendDir `
    -WindowStyle Hidden `
    -RedirectStandardOutput $backendOut `
    -RedirectStandardError $backendErr `
    -PassThru

$ready = $false
for ($i = 1; $i -le 20; $i++) {
    Start-Sleep -Seconds 1
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$BackendPort/api/v1/health" `
                                  -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            $ready = $true
            $healthJson = $resp.Content | ConvertFrom-Json
            Write-OK "Backend up — status: $($healthJson.status), n_tubes: $($healthJson.n_tubes), pid: $($backendProc.Id)"
            break
        }
    } catch {
        Write-Host "." -NoNewline
    }
}
if (-not $ready) {
    Write-Err "Backend failed to start. See $backendErr"
    if (Test-Path $backendErr) { Get-Content $backendErr -Tail 40 | Write-Host }
    exit 1
}

# ─────────────────────────────────────────────────────────────────────────────
# 9. Start dashboard static server (background job)
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Starting dashboard on port $DashboardPort"

$dashOut = Join-Path $LogDir 'dashboard.out.log'
$dashErr = Join-Path $LogDir 'dashboard.err.log'
Remove-Item $dashOut, $dashErr -ErrorAction SilentlyContinue
# Windows PowerShell joins -ArgumentList on spaces, so quote paths that contain spaces.
$dashArgs = "-m http.server $DashboardPort --bind 127.0.0.1 --directory `"$FrontendDir`""
$dashboardProc = Start-Process -FilePath $VenvPython `
    -ArgumentList $dashArgs `
    -WorkingDirectory $FrontendDir `
    -WindowStyle Hidden `
    -RedirectStandardOutput $dashOut `
    -RedirectStandardError $dashErr `
    -PassThru

$dashReady = $false
for ($i = 1; $i -le 8; $i++) {
    Start-Sleep -Seconds 1
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$DashboardPort/index.html" `
                                  -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            $dashReady = $true
            Write-OK "Dashboard serving index.html ($($resp.Content.Length) bytes), pid: $($dashboardProc.Id)"
            break
        }
    } catch { }
}
if (-not $dashReady) {
    Write-Warn "Dashboard probe failed. See $dashErr"
    if (Test-Path $dashErr) { Get-Content $dashErr -Tail 20 | Write-Host }
}

@{
    backend   = $backendProc.Id
    dashboard = $dashboardProc.Id
} | ConvertTo-Json | Set-Content -Path $PidFile -Encoding ascii
Write-OK "PID file: $PidFile"

# ─────────────────────────────────────────────────────────────────────────────
# 10. Seed demo data so the dashboard shows realistic content
# ─────────────────────────────────────────────────────────────────────────────
if (-not $NoSeed) {
    Write-Step "Seeding live demo data into backend"

    # ── Temperature readings via POST /api/v1/ingest/temperature ──
    Write-Host "  → Temperature: 12 tubes × 60 minutes = 720 readings (one batch)"
    $tubes = @()
    for ($r = 1; $r -le 4; $r++) {
        for ($c = 1; $c -le 3; $c++) {
            $tubes += "T{0:D2}-{1:D3}" -f $r, $c
        }
    }
    $baseTemp = @{ 'T01' = 890.0; 'T02' = 885.0; 'T03' = 870.0; 'T04' = 860.0 }
    $rowOff   = @{ '001' = 0.0; '002' = -5.0; '003' = 6.0 }
    $baseTs   = 1000
    $rng      = [System.Random]::new(42)
    $readings = @()

    $loginBody = '{"username":"admin","password":"admin123"}'
    $token = $null
    try {
        $loginResp = Invoke-WebRequest -Uri "http://127.0.0.1:$BackendPort/api/v1/auth/login" `
            -Method Post -Body $loginBody -ContentType 'application/json' `
            -UseBasicParsing -TimeoutSec 5
        $token = ($loginResp.Content | ConvertFrom-Json).access_token
    } catch {
        Write-Warn "Login failed (ingest may proceed anonymously): $($_.Exception.Message)"
    }

    for ($minute = 0; $minute -lt 60; $minute++) {
        foreach ($tid in $tubes) {
            $parts = $tid -split '-'
            $rowKey = $parts[0]; $colKey = $parts[1]
            $base = $baseTemp[$rowKey]
            $off  = $rowOff[$colKey]
            $wave = 4 * [Math]::Sin(($minute + [int]$colKey) * 0.15)
            $noise = $rng.NextDouble() * 2 - 1
            $temp = $base + $off + $wave + $noise
            # Stay more than 30 °C above the furnace mean so hotspots are real.
            if ($tid -eq 'T01-003' -or $tid -eq 'T02-003') {
                $temp = 945 + ($rng.NextDouble() * 6 - 3)
            }
            $readings += @{
                tube_id     = $tid
                ts_minute   = $baseTs + $minute
                wall_temp_c = [Math]::Round($temp, 1)
                source      = 'seed'
            }
        }
    }
    $body = @{ readings = $readings } | ConvertTo-Json -Compress -Depth 4
    $headers = @{ 'Content-Type' = 'application/json' }
    if ($token) { $headers['Authorization'] = "Bearer $token" }
    try {
        $ing = Invoke-WebRequest -Uri "http://127.0.0.1:$BackendPort/api/v1/ingest/temperature/batch" `
            -Method Post -Body $body -Headers $headers `
            -UseBasicParsing -TimeoutSec 60
        $n = ($ing.Content | ConvertFrom-Json).n
        Write-OK "Ingested $n temperature readings"
    } catch {
        Write-Warn "Temperature batch failed: $($_.Exception.Message)"
    }

    # ── Burner status directly into SQLite (no public ingest endpoint) ──
    Write-Host "  → Burners: 8 burners × 30 minutes = 240 readings (direct DB)"
    $seedScript = Join-Path $ScriptDir 'seed_burners_db.py'
    if (Test-Path $seedScript) {
        & $VenvPython $seedScript --db $DbPath --minutes 30 --burners 8 --base-ts 1000
        Write-OK "Burner readings inserted"
    } else {
        Write-Warn "seed_burners_db.py not found at $seedScript — skipping burners"
    }

    # ── Trigger one optimization run so Module 3 panel lights up ──
    Write-Host "  → Optimization run (Module 3 + 4)"
    $optBody = @{
        sc_ratio       = 3.0
        t_out_c        = 880.0
        feed_flow_nm3h = 28000.0
        fuel_flow_nm3h = 12000.0
        cat_age        = 0.8
    } | ConvertTo-Json -Compress
    $headers = @{ 'Content-Type' = 'application/json' }
    if ($token) { $headers['Authorization'] = "Bearer $token" }
    try {
        $optResp = Invoke-WebRequest -Uri "http://127.0.0.1:$BackendPort/api/v1/optimization" `
            -Method Post -Body $optBody -Headers $headers `
            -UseBasicParsing -TimeoutSec 10
        $opt = $optResp.Content | ConvertFrom-Json
        Write-OK "Optimization: eff=$([Math]::Round($opt.efficiency_pred,3)), coking_risk=$([Math]::Round($opt.coking_risk_pred,3))"
    } catch {
        Write-Warn "Optimization call failed: $($_.Exception.Message)"
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# 11. Final verification
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Final verification"
try {
    $h = (Invoke-WebRequest "http://127.0.0.1:$BackendPort/api/v1/health" -UseBasicParsing).Content | ConvertFrom-Json
    $t = (Invoke-WebRequest "http://127.0.0.1:$BackendPort/api/v1/temperature-map" -UseBasicParsing).Content | ConvertFrom-Json
    Write-OK "Health: n_tubes=$($h.n_tubes), open_alerts=$($h.open_alerts)"
    Write-OK "Temp map: mean=$($t.mean_wall_temp_c)°C, hotspots=$($t.hotspots.Count)"
} catch {
    Write-Warn "Final probe failed: $($_.Exception.Message)"
}

# ─────────────────────────────────────────────────────────────────────────────
# 12. Open browser
# ─────────────────────────────────────────────────────────────────────────────
$dashboardUrl = "http://localhost:$DashboardPort/index.html"
$docsUrl      = "http://localhost:$BackendPort/docs"

if (-not $NoBrowser) {
    Write-Step "Opening dashboard in browser"
    Start-Process $dashboardUrl
    Write-OK "Browser launched"
}

# ─────────────────────────────────────────────────────────────────────────────
# 13. Summary
# ─────────────────────────────────────────────────────────────────────────────
Write-Host "`n=====================================================" -ForegroundColor White
Write-Host " ✅ MRTO Dashboard is LIVE" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor White
Write-Host " 🖥  Dashboard:  $dashboardUrl" -ForegroundColor Cyan
Write-Host " 🔌 API:        http://localhost:$BackendPort" -ForegroundColor Cyan
Write-Host " 📖 Swagger:   $docsUrl" -ForegroundColor Cyan
Write-Host " 💚 Health:    http://localhost:$BackendPort/api/v1/health" -ForegroundColor Cyan
Write-Host " ---------------------------------------------------"
Write-Host " 🔑 Admin login:  admin / admin123" -ForegroundColor Yellow
Write-Host " 🛑 Stop:          .\scripts\stop-dashboard.ps1"
Write-Host " 📄 Logs:          $LogDir"
Write-Host "=====================================================" -ForegroundColor White
Write-Host ""
Write-Host "Servers keep running after this script exits."
Write-Host "A live simulator appends a new furnace sample every 5 seconds."
