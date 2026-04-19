$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
$pidFile = Join-Path $scriptDir ".game-server.pid"
$logFile = Join-Path $scriptDir ".game-server.log"
$errLogFile = Join-Path $scriptDir ".game-server.err.log"
$battleUrl = if ($env:BATTLE_URL) {
  $env:BATTLE_URL
} else {
  "http://127.0.0.1:4000/?mode=battle&ml=1&ml_conf=0.26&ml_move_conf=0.34&ml_margin=0.03&ml_force_move_eta=460&ml_wait_block_eta=760&ml_move_threat_ms=300&ml_model=/output/ml/models/dodge_bc_v1.onnx"
}

function Wait-ProcessExit {
  param(
    [Parameter(Mandatory = $true)]
    [int]$ProcessId,
    [int]$TimeoutMs = 2000
  )

  $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
  if (-not $proc) {
    return $true
  }

  try {
    $null = $proc.WaitForExit($TimeoutMs)
  } catch {
  }

  return -not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)
}

function Stop-ServerProcess {
  param(
    [Parameter(Mandatory = $true)]
    [int]$ProcessId
  )

  $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
  if (-not $proc) {
    return
  }

  Write-Host "Stopping process (PID: $ProcessId)..."
  try {
    Stop-Process -Id $ProcessId -ErrorAction SilentlyContinue
  } catch {
  }

  if (Wait-ProcessExit -ProcessId $ProcessId -TimeoutMs 2000) {
    return
  }

  try {
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
  } catch {
  }
  $null = Wait-ProcessExit -ProcessId $ProcessId -TimeoutMs 1000
}

if (Test-Path $pidFile) {
  $pidText = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($null -ne $pidText) {
    $pidText = $pidText.ToString().Trim()
    if ($pidText -match "^\d+$") {
      Stop-ServerProcess -ProcessId ([int]$pidText)
    }
  }
  Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

$portOwners = @()
try {
  $portOwners = @(Get-NetTCPConnection -LocalPort 4000 -State Listen -ErrorAction Stop | Select-Object -ExpandProperty OwningProcess -Unique)
} catch {
  $portOwners = @()
}

if ($portOwners.Count -gt 0) {
  Write-Host "Stopping process(es) listening on :4000 -> $($portOwners -join ', ')"
  foreach ($ownerId in $portOwners) {
    if ($ownerId -is [int]) {
      Stop-ServerProcess -ProcessId $ownerId
    } elseif ($ownerId -match "^\d+$") {
      Stop-ServerProcess -ProcessId ([int]$ownerId)
    }
  }
}

Push-Location $rootDir
try {
  $newProc = Start-Process -FilePath "node" -ArgumentList "app.js" -WorkingDirectory $rootDir -RedirectStandardOutput $logFile -RedirectStandardError $errLogFile -PassThru
} finally {
  Pop-Location
}

Start-Sleep -Milliseconds 300
if (Get-Process -Id $newProc.Id -ErrorAction SilentlyContinue) {
  Set-Content -Path $pidFile -Value $newProc.Id -NoNewline -Encoding ascii
  Write-Host "Game server restarted (PID: $($newProc.Id))."
  Write-Host "URL: http://127.0.0.1:4000"
  Write-Host "Battle URL:"
  Write-Host $battleUrl
  Write-Host "Log: $logFile"
  Write-Host "Error Log: $errLogFile"
} else {
  Write-Error "Failed to start game server. Check log: $logFile / $errLogFile"
  exit 1
}
