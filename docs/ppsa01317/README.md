# PPSA01317 bleeding-edge branch: how to run it

This branch is an experimental performance branch for PPSA01317 and is not meant for upstream. Many of its changes are opt-in, so a plain launch is slower than our measurements. Set the variables below to match the setup that reached about 19 FPS on the World Map with a warm shader cache.

## Environment variables

| Variable | Value | What it does |
|---|---|---|
| `APS5_NET_OFFLINE` | `1` | Makes network calls fail at once. Without it the title waits minutes at boot for server responses. |
| `APS5_NO_PIPELINE_JOURNAL` | `1` | Turns off the pipeline journal replay at boot (it costs FPS while it runs). |
| `APS5_LIVE_FLAGS` | path to `live-flags.txt` | Live switches, re-read while the game runs. Copy `docs/ppsa01317/live-flags.txt` next to the game and point this at it. |
| `APS5_ADDRESS_TEMPLATES` | `1` | |
| `APS5_RESOURCE_CACHE_ENTRIES` | `8192` | |
| `APS5_SPLIT_DRAWS` | `6` | |
| `APS5_SPLIT_DRAW_FRONT` | `1` | |
| `APS5_MOVABLE_IMAGES` | `1` | |
| `APS5_SHARED_EPOCH_MEMO` | `1` | |
| `APS5_PIPELINED_DISPATCHES` | `1` | |
| `APS5_PRESENT_MODE` | `mailbox` | |
| `APS5_FLIP_EARLY` | `1` | |
| `APS5_BDA_HINT_FAST` | `1` | |
| `APS5_BDA_INLINE_HIT` | `1` | |
| `APS5_FAST_HEAP` | `1` | |
| `APS5_NO_ARM_RESIDENT` | `1` | |
| `APS5_HOST_IMPORT_MIB` | `12288` | Host-import budget in MiB. Lower it on machines with less than 32 GB of RAM. |
| `APS5_RECORDER_THREAD` | `1` | |
| `APS5_STARVE_SUBMIT` | `4` | |
| `APS5_VOLUME_ALIAS_DISCARD` | `1` | |
| `APS5_RT_SLICE_LAYERS` | `1` | |
| `APS5_MOUSE_SPACE` | `1920x1080` | Only needed for our automation scripts. |
| `SDL_AUDIODRIVER` | `dummy` | Mutes audio (used for measurements). Leave it unset to hear the game. |

The unlabelled rows are performance switches from earlier experiments. Each is described where it is read in the code.

## PowerShell launch example

```powershell
$env:APS5_NET_OFFLINE = '1'
$env:APS5_NO_PIPELINE_JOURNAL = '1'
$env:APS5_LIVE_FLAGS = "$PWD\live-flags.txt"
$env:APS5_ADDRESS_TEMPLATES = '1'
$env:APS5_RESOURCE_CACHE_ENTRIES = '8192'
$env:APS5_SPLIT_DRAWS = '6'
$env:APS5_SPLIT_DRAW_FRONT = '1'
$env:APS5_MOVABLE_IMAGES = '1'
$env:APS5_SHARED_EPOCH_MEMO = '1'
$env:APS5_PIPELINED_DISPATCHES = '1'
$env:APS5_PRESENT_MODE = 'mailbox'
$env:APS5_FLIP_EARLY = '1'
$env:APS5_BDA_HINT_FAST = '1'
$env:APS5_BDA_INLINE_HIT = '1'
$env:APS5_FAST_HEAP = '1'
$env:APS5_NO_ARM_RESIDENT = '1'
$env:APS5_HOST_IMPORT_MIB = '12288'
$env:APS5_RECORDER_THREAD = '1'
$env:APS5_STARVE_SUBMIT = '4'
$env:APS5_VOLUME_ALIAS_DISCARD = '1'
$env:APS5_RT_SLICE_LAYERS = '1'
& .\app.exe 1> run-out.log 2> run-err.log
```

The window title shows the current FPS.

## Notes

- The first boot compiles shaders and is slow. FPS settles after a couple of minutes on the World Map, and later boots start warm from the saved pipeline cache.
- Newer switches (lazy write-back, native pattern fill, and the committer cuts `room_spin`, `fast_heap_prefetch`, `lock_spin`, `pending_fill`, `index_scan_memo`, `window_early_out`) are on by default. Setting one to `0` in `live-flags.txt` turns it off while the game runs.
- Results were measured on an NVIDIA GPU at 4K. Other GPUs are untested.

AI-assisted: yes
