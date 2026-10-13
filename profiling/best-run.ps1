# Boots the menu with the best-known configuration (2026-10-11) plus any extra NAME=VALUE pairs.
param([Parameter(Mandatory)][string]$Tag, [int]$Seconds = 80, [string[]]$ExtraEnv = @())
$best = @('APS5_ADDRESS_TEMPLATES=1','APS5_RESOURCE_CACHE_ENTRIES=8192','APS5_SPLIT_DRAWS=6','APS5_MOVABLE_IMAGES=1','APS5_SHARED_EPOCH_MEMO=1','APS5_PIPELINED_DISPATCHES=1','APS5_PRESENT_MODE=mailbox','APS5_SPLIT_DRAW_FRONT=1','APS5_FLIP_EARLY=1','APS5_LIVE_FLAGS=D:\ps5\gt7\live-flags.txt','APS5_MOUSE_SPACE=1920x1080','APS5_PERF_CONTROL=D:\ps5\gt7\perf-control.txt','APS5_TRACE_ALIAS_FLUSH=','APS5_BDA_HINT_FAST=1','APS5_BDA_INLINE_HIT=1','APS5_PROFILE_DRAW=','APS5_FAST_HEAP=1','APS5_NO_ARM_RESIDENT=1','APS5_HOST_IMPORT_MIB=12288','APS5_RECORDER_THREAD=1','APS5_STARVE_SUBMIT=4','APS5_VOLUME_ALIAS_DISCARD=1','APS5_RT_SLICE_LAYERS=1','SDL_AUDIODRIVER=dummy','APS5_NET_OFFLINE=1')
# NAME=! in ExtraEnv drops that default instead of setting it (offline networking always stays).
$drop = @($ExtraEnv | Where-Object { $_ -like '*=!' } | ForEach-Object { $_.Substring(0, $_.Length - 2) })
$best = @($best | Where-Object { $drop -notcontains ($_ -split '=', 2)[0] -or $_ -eq 'APS5_NET_OFFLINE=1' })
$ExtraEnv = @($ExtraEnv | Where-Object { $_ -notlike '*=!' })
& "$PSScriptRoot\boot-run.ps1" -Tag $Tag -Seconds $Seconds -ExtraEnv ($best + $ExtraEnv)
