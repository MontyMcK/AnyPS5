# Boots the title, goes title screen -> World Map -> Garage by keyboard, and screenshots the garage.
# Reports per shot the mean brightness (0 = black) and whether the game is still running, then the
# device-fault / FATAL lines of the log. Usage: start the game first (best-run.ps1 -Tag <tag> with
# APS5_DEVICE_FAULT=1, from its own shell), then nav-garage.ps1 -Tag <tag>. A game launched from a
# PowerShell job cannot be brought to the foreground, so this script does not launch it.
param([Parameter(Mandatory)][string]$Tag, [int]$Shots = 8, [int]$ShotGapSeconds = 8, [int]$ReadySeconds = 45, [switch]$Stop)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$out = "D:\ps5\work\nav\$Tag"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$samples = "D:\ps5\gt7\diagnostics-rapid\$Tag\samples.jsonl"
$deadline = (Get-Date).AddSeconds($ReadySeconds + 300)
while ((Get-Date) -lt $deadline) {
    if (Test-Path $samples) {
        $last = Get-Content $samples -Tail 1 -ErrorAction SilentlyContinue
        if ($last -and (($last | ConvertFrom-Json).elapsedSeconds -gt $ReadySeconds)) { break }
    }
    Start-Sleep -Seconds 3
}
function Alive { $p = [System.Text.Json.JsonDocument]::Parse([IO.File]::ReadAllText("D:\ps5\gt7\diagnostics-rapid\$Tag\process.json")).RootElement.GetProperty('pid').GetInt32(); return [bool](Get-Process -Id $p -ErrorAction SilentlyContinue) }
& "$here\shot.ps1" -Tag $Tag -OutFile "$out\title.png"
& "$here\press.ps1" -Tag $Tag -Keys ENTER
Start-Sleep -Seconds 12
& "$here\shot.ps1" -Tag $Tag -OutFile "$out\worldmap.png"
& "$here\press.ps1" -Tag $Tag -Keys RIGHT
Start-Sleep -Seconds 3
& "$here\shot.ps1" -Tag $Tag -OutFile "$out\garage-selected.png"
& "$here\press.ps1" -Tag $Tag -Keys ENTER
for ($i = 1; $i -le $Shots; $i++) {
    Start-Sleep -Seconds $ShotGapSeconds
    if (-not (Alive)) { Write-Output "game exited before garage shot $i"; break }
    & "$here\shot.ps1" -Tag $Tag -OutFile ("$out\garage{0:D2}.png" -f $i)
}
if ($Stop -and (Alive)) { & "$here\stop-run.ps1" -Tag $Tag | Out-Null }
$log = "D:\ps5\gt7\diagnostics-rapid\$Tag\run-err.log"
if (Test-Path $log) { Select-String -Path $log -Pattern 'device-fault\]|FATAL|left unprepared|\[skips\]' | Select-Object -First 30 | ForEach-Object { $_.Line.Substring(0, [Math]::Min(260, $_.Line.Length)) } }
