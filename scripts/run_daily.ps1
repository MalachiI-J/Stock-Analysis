<#
Runs the daily Stock Scrapper pipeline unattended: collects/validates/analyzes/reports
(`main.py run`), writes a plain-language buy/watch/sell digest (`main.py digest`),
computes sized advisory trade recommendations (`main.py recommend` — suggestions
only, nothing is ever bought or sold automatically), shows a combined summary as a
Windows toast notification, opens that day's canonical Phase 2 HTML report in the
default browser, then deletes log files and unreferenced report artifacts (old
digest/recommendations/data-health/screener files; never analysis/backtest reports
tied to a saved run) older than the configured retention window
(`main.py cleanup-logs --include-reports`). Intended to be invoked by Windows Task
Scheduler once per day, after the market close plus the configured provider delay
(see market_data.provider_delay_minutes in config/settings.yaml).

The toast notification and report auto-open both require an interactive logon
session, which is why this task is registered with LogonType=Interactive rather
than running detached. The report is located by its documented canonical filename
pattern (`stock_summary_<date>_candidates_<hash>.html`) rather than by parsing
`main.py run`'s output; if `run` failed to produce a report for today, opening is
skipped and noted in the log rather than treated as a fatal error. Auto-open is
gated by `open_reports_automatically` in config/settings.yaml, so it can be
turned off without editing this script.

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

cmd.exe /c "`"$pythonExe`" main.py recommend >> `"$logFile`" 2>&1"
$recommendExit = $LASTEXITCODE

$today = Get-Date -Format "yyyy-MM-dd"
$summaryPath = Join-Path $RepoRoot "reports\digest_$today.summary.json"
$recommendPath = Join-Path $RepoRoot "reports\recommendations_$today.summary.json"
if (Test-Path $summaryPath) {
    try {
        $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
        $sellSymbols = @($summary.holdings_sell_symbols)
        $changes = @($summary.changes)
        $title = "Stock Scrapper digest - $($summary.as_of_date)"
        $lines = @("Buy: $($summary.buy_count)  Sell: $($summary.sell_count)  Watch: $($summary.watch_count)")
        if ($sellSymbols.Count -gt 0) { $lines += "Consider selling: " + ($sellSymbols -join ", ") }
        if ($changes.Count -gt 0) { $lines += ($changes | Select-Object -First 3) -join "; " }
        if (Test-Path $recommendPath) {
            $recommend = Get-Content $recommendPath -Raw | ConvertFrom-Json
            $recs = @($recommend.recommendations)
            $buyCount = ($recs | Where-Object { $_.action -eq "BUY" }).Count
            $sellCount = ($recs | Where-Object { $_.action -eq "SELL" }).Count
            $lines += "Recommend (advisory, unproven model): $buyCount buy / $sellCount sell - review before acting"
        }
        $message = $lines -join "`n"
        & (Join-Path $PSScriptRoot "send_toast.ps1") -Title $title -Message $message *>> $logFile
    } catch {
        Add-Content -Path $logFile -Value "Toast notification skipped: $_"
    }
} else {
    Add-Content -Path $logFile -Value "Toast notification skipped: summary file not found at $summaryPath"
}

$openReportsSetting = (& $pythonExe -c "from stock_scrapper.config import load_config; print(bool(load_config().get('open_reports_automatically', False)))").Trim()
if ($openReportsSetting -eq "True") {
    $reportFile = Get-ChildItem -Path (Join-Path $RepoRoot "reports") -Filter "stock_summary_${today}_candidates_*.html" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($reportFile) {
        try {
            Start-Process $reportFile.FullName
            Add-Content -Path $logFile -Value "Opened report: $($reportFile.FullName)"
        } catch {
            Add-Content -Path $logFile -Value "Report auto-open skipped: $_"
        }
    } else {
        Add-Content -Path $logFile -Value "Report auto-open skipped: no report file found matching stock_summary_${today}_candidates_*.html"
    }
} else {
    Add-Content -Path $logFile -Value "Report auto-open disabled (open_reports_automatically is false in config/settings.yaml)"
}

cmd.exe /c "`"$pythonExe`" main.py cleanup-logs --include-reports >> `"$logFile`" 2>&1"
$cleanupExit = $LASTEXITCODE

Add-Content -Path $logFile -Value "=== Stock Scrapper daily run finished $(Get-Date -Format o): run=$runExit digest=$digestExit recommend=$recommendExit cleanup=$cleanupExit ==="

exit ([Math]::Max($runExit, [Math]::Max($digestExit, [Math]::Max($recommendExit, $cleanupExit))))
