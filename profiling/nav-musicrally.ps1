# Boots into Music Rally: title screen -> LEFT, RIGHT, ENTER on Music Rally -> wait for the song list ->
# ENTER on the first song -> optional extra ENTERs for confirmation screens. Prints a timestamp per
# step so frame dumps (APS5_DUMP_FRAMES/APS5_DUMP_FRAMES_EVERY) can be matched to them. Start the
# game first with best-run.ps1 -Tag <tag> from its own shell; this script does not launch it.
param([Parameter(Mandatory)][string]$Tag, [int]$ReadySeconds = 120, [int]$MenuSeconds = 15, [int]$SongSeconds = 12, [int]$ExtraEnters = 0, [int]$ExtraGapSeconds = 8)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$samples = "D:\ps5\gt7\diagnostics-rapid\$Tag\samples.jsonl"
$deadline = (Get-Date).AddSeconds($ReadySeconds + 300)
while ((Get-Date) -lt $deadline) {
    if (Test-Path $samples) {
        $last = Get-Content $samples -Tail 1 -ErrorAction SilentlyContinue
        if ($last -and (($last | ConvertFrom-Json).elapsedSeconds -gt $ReadySeconds)) { break }
    }
    Start-Sleep -Seconds 3
}
function Step([string]$name, [string[]]$keys) {
    "{0:HH:mm:ss} {1}" -f (Get-Date), $name
    & "$here\press.ps1" -Tag $Tag -Keys $keys | Out-Null
}
# The title screen moves an on-screen cursor spatially between its items, and where it starts varies:
# World Map -RIGHT-> top-right link icons -DOWN-> Music Rally. titlefocus.py reads the newest frame
# dump and says which button is highlighted (WM, MR, NONE); keys are pressed until MR is lit.
$noneMoves = @('DOWN', 'UP', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'DOWN')
$noneIndex = 0
$reached = $false
for ($attempt = 0; $attempt -lt 16; $attempt++) {
    Start-Sleep -Seconds 3
    $state = (python -I "$here\titlefocus.py").Split(' ')[0]
    if ($state -eq 'MR') { $reached = $true; break }
    if ($state -eq 'WM') { Step 'title: WM highlighted, RIGHT' @('RIGHT'); continue }
    $move = $noneMoves[$noneIndex % $noneMoves.Count]; $noneIndex++
    Step "title: $state, $move" @($move)
}
if (-not $reached) { throw 'could not highlight Music Rally on the title screen' }
Step 'title: ENTER Music Rally' @('ENTER')
Start-Sleep -Seconds $MenuSeconds
Step 'song list: ENTER first song' @('ENTER')
Start-Sleep -Seconds $SongSeconds
for ($i = 1; $i -le $ExtraEnters; $i++) {
    Step "confirm: ENTER $i" @('ENTER')
    Start-Sleep -Seconds $ExtraGapSeconds
}
"{0:HH:mm:ss} done" -f (Get-Date)
