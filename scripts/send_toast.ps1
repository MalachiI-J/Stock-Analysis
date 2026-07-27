<#
Shows a native Windows toast notification. Uses no external module (no
BurntToast install required) via the built-in WinRT notification API,
attributed to PowerShell's own registered AppUserModelID so Windows will
display it without a custom app registration.

Requires an interactive logon session — this is why the Stock Scrapper
daily Task Scheduler entry is registered with LogonType=Interactive rather
than running detached in the background.

Best-effort only: failures are caught and reported as a warning, never as a
fatal error, since a missed notification should not fail the daily pipeline.
#>

param(
    [Parameter(Mandatory = $true)][string]$Title,
    [Parameter(Mandatory = $true)][string]$Message
)

try {
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
    [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null

    $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(
        [Windows.UI.Notifications.ToastTemplateType]::ToastText02
    )
    $textNodes = $template.GetElementsByTagName("text")
    $textNodes.Item(0).AppendChild($template.CreateTextNode($Title)) | Out-Null
    $textNodes.Item(1).AppendChild($template.CreateTextNode($Message)) | Out-Null

    $appId = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
    $toast = [Windows.UI.Notifications.ToastNotification]::new($template)
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId).Show($toast)
    exit 0
} catch {
    Write-Warning "Toast notification failed: $_"
    exit 1
}
