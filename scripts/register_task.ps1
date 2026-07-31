<#
(Re)creates the \StockScrapper\DailyRun Windows Task Scheduler task from source,
instead of the task's configuration existing only as manually-clicked, undocumented
live state on one machine. Safe to re-run: it unregisters any existing
\StockScrapper\DailyRun task first, then registers it fresh from the settings below,
so this script is always the single source of truth for what the task should look
like. Run once to create the task, or again any time after changing the settings in
this file (time, retry policy, etc.) to apply them.

Requires an elevated PowerShell only if the current user account does not already
have "Log on as a batch job" rights for scheduled tasks; ordinarily this can be run
as the interactive user who will also be logged in when the task fires (LogonType
Interactive requires that same user to be logged in at trigger time, since the toast
notification and report auto-open both need an interactive desktop session).
#>

param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$StartTime = "06:00",
    # Retry policy: a single transient failure (a flaky network request during
    # collection, a momentarily locked file) previously meant waiting a full day for
    # the next scheduled run. Two retries five minutes apart gives a same-morning
    # second and third chance before falling back to the next scheduled day.
    [int]$RestartCount = 2,
    [int]$RestartIntervalMinutes = 5,
    [int]$ExecutionTimeLimitHours = 1
)

$taskPath = "\StockScrapper\"
$taskName = "DailyRun"
$scriptPath = Join-Path $RepoRoot "scripts\run_daily.ps1"

if (-not (Test-Path $scriptPath)) {
    Write-Error "run_daily.ps1 not found at $scriptPath"
    exit 1
}

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`"" `
    -WorkingDirectory $RepoRoot

$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $StartTime

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours $ExecutionTimeLimitHours) `
    -RestartCount $RestartCount `
    -RestartInterval (New-TimeSpan -Minutes $RestartIntervalMinutes)

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

$existing = Get-ScheduledTask -TaskPath $taskPath -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Output "Existing task found; unregistering before re-registering from source."
    Unregister-ScheduledTask -TaskPath $taskPath -TaskName $taskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskPath $taskPath `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Stock Scrapper: collect, analyze, report, digest, and advisory-recommend, Mon-Fri after market close." `
    | Out-Null

Write-Output "Registered \$taskPath$taskName -> $scriptPath, weekdays at $StartTime, retry $RestartCount x ${RestartIntervalMinutes}min."
Get-ScheduledTask -TaskPath $taskPath -TaskName $taskName | Select-Object TaskName, State
