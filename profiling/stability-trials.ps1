# Runs -Count sequential Music Rally race trials with best-run.ps1 plus -ExtraEnv, each driven by
# nav-musicrally2.py and held -HoldSeconds after the race start, then classified from its logs:
# race (reached and survived), fatal (crash or device loss), assert (game BVH assertion), hang
# (no race and no fatal). Results append to D:\ps5\gt7\diagnostics-rapid\<Prefix>-results.txt.
param([Parameter(Mandatory)][string]$Prefix, [int]$Count = 6, [int]$HoldSeconds = 180, [string[]]$ExtraEnv = @())
$ErrorActionPreference = 'Stop'
$root = 'D:\ps5\gt7\diagnostics-rapid'
$results = Join-Path $root "$Prefix-results.txt"
for ($i = 1; $i -le $Count; $i++) {
    $tag = "$Prefix$i"
    Set-Content D:\ps5\gt7\debug-control.txt '# idle'
    $env = @('APS5_DUMP_FRAMES=3000', 'APS5_DUMP_FRAMES_EVERY=60', 'APS5_AUTOMATION=1', "APS5_AUTOMATION_EVENTS=$root\$tag\automation-events.jsonl") + $ExtraEnv
    $job = Start-Job -ScriptBlock { param($t, $e) & C:\repos\AnyPS5-pr639\profiling\best-run.ps1 -Tag $t -Seconds 1500 -ExtraEnv $e } -ArgumentList $tag, $env
    $deadline = (Get-Date).AddSeconds(240)
    while (-not (Test-Path "$root\$tag\process.json") -and (Get-Date) -lt $deadline) { Start-Sleep 3 }
    $pipeReady = $false
    while ((Get-Date) -lt $deadline) {
        if (Select-String -Path 'D:\ps5\gt7\run-err.log' -Pattern '\[automation\] commands' -Quiet -ErrorAction SilentlyContinue) { $pipeReady = $true; break }
        Start-Sleep 3
    }
    $navOut = if ($pipeReady) { & python -I C:\repos\AnyPS5-pr639\profiling\nav-musicrally2.py $tag --screen-timeout 150 --start-presses 2 2>&1 | Out-String } else { 'no pipe' }
    $raced = $navOut -match 'race started'
    if ($raced) {
        $holdEnd = (Get-Date).AddSeconds($HoldSeconds)
        while ((Get-Date) -lt $holdEnd -and -not (Test-Path "$root\$tag\exit.json")) { Start-Sleep 5 }
    }
    $log = Get-Content 'D:\ps5\gt7\run-err.log' -ErrorAction SilentlyContinue
    $fatal = @($log | Select-String -Pattern 'FATAL').Count -gt 0
    $assert = @($log | Select-String -Pattern 'refit_leaf_count').Count -gt 0
    $exited = Test-Path "$root\$tag\exit.json"
    $outcome = if ($fatal) { 'fatal' } elseif ($assert) { 'assert' } elseif ($raced -and -not $exited) { 'race' } else { 'hang' }
    $detail = (@($log | Select-String -Pattern 'FATAL') | Select-Object -First 1)
    "{0:HH:mm:ss} {1} {2} {3}" -f (Get-Date), $tag, $outcome, $detail | Tee-Object -FilePath $results -Append
    try { & C:\repos\AnyPS5-pr639\profiling\stop-run.ps1 -Tag $tag | Out-Null } catch {}
    Start-Sleep 5
    Stop-Job $job -ErrorAction SilentlyContinue; Remove-Job $job -Force -ErrorAction SilentlyContinue
}
