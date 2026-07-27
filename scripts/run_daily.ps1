<#
Runs the daily Stock Scrapper pipeline unattended: collects/validates/analyzes/reports
(`main.py run`), then writes a plain-language buy/watch/sell digest (`main.py digest`).
Intended to be invoked by Windows Task Scheduler once per day, after the market
close plus the configured provider delay (see market_data.provider_delay_minutes
in config/settings.yaml).

Uses cmd.exe for output redirection rather than PowerShell's native `*>>`/`2>&1`,
which wraps every stderr line from a native exe (Python's logger writes INFO to
stderr) in a NativeCommandError and can emit UTF-16 log files.
#>

param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot)
)

$pythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Error "Virtual environment python not found at $pythonExe"
    exit 1
}

$logDir = Join-Path $RepoRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$logFile = Join-Path $logDir ("daily_run_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

Set-Location $RepoRoot

Add-Content -Path $logFile -Value "=== Stock Scrapper daily run started $(Get-Date -Format o) ==="

cmd.exe /c "`"$pythonExe`" main.py run >> `"$logFile`" 2>&1"
$runExit = $LASTEXITCODE

cmd.exe /c "`"$pythonExe`" main.py digest >> `"$logFile`" 2>&1"
$digestExit = $LASTEXITCODE

Add-Content -Path $logFile -Value "=== Stock Scrapper daily run finished $(Get-Date -Format o): run=$runExit digest=$digestExit ==="

exit ([Math]::Max($runExit, $digestExit))
