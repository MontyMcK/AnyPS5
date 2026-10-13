# Alternating A/B race trials: for each round, one stability-trials.ps1 trial per config (in order),
# so drift (heat, background load) hits every config alike. Configs are 'name=ENV1,ENV2,...' strings;
# an empty env list ('base=') runs the shared profiling env only. Results land in
# D:\ps5\gt7\diagnostics-rapid\<name>-results.txt per config (tags <name>1, <name>2, ...).
param([Parameter(Mandatory)][string[]]$Configs, [int]$Rounds = 2, [int]$HoldSeconds = 150, [string[]]$CommonEnv = @('APS5_PROFILE_COLLECT=1', 'APS5_PROFILE_DRAW=1'))
$ErrorActionPreference = 'Stop'
$root = 'D:\ps5\gt7\diagnostics-rapid'
for ($round = 1; $round -le $Rounds; $round++) {
    foreach ($config in $Configs) {
        $name, $list = $config -split '=', 2
        $extra = @($CommonEnv) + @($list -split ',' | Where-Object { $_ -ne '' })
        $n = 1
        while (Test-Path "$root\$name$n") { $n++ }
        # stability-trials numbers its tags from 1; give it a prefix that lands on a fresh tag.
        & "$PSScriptRoot\stability-trials.ps1" -Prefix "$name" -Count 1 -HoldSeconds $HoldSeconds -ExtraEnv $extra -StartIndex $n
    }
}
