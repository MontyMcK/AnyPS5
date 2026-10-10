# Current shader investigation handoff

Work in `C:\repos\AnyPS5-pr639`, branch `try-639-on-runtree`; the chat's ef40 worktree is not the active checkout. Preserve all local changes and `D:\ps5\gt7\shader_cache`. Nothing pushed or committed during this work. Objective remains resolve all shader issues, not merely reduce the failure counter.

## Live run

`packed-float-menu`: PID7036, executable `D:\ps5\gt7\app.exe`, exact UTC start2026-10-09T16:08:57.7608311Z. Observer session22974,1800second cap. Confirm path/start before stopping. Evidence at `D:\ps5\gt7\diagnostics-rapid\packed-float-menu`. At~30seconds startup was presenting; no settled menu or packed-format success claim yet.

Includes PR2253 direct reads, PR2105 control-flow optimization, specialized-module disk cache, dense-table SpecializeSpirv optimization, restored reversed blending, export-mapping static cache-key fix, and new packed-float target support. Preserve caches; source changes automatically select new generations. Avoid build/GPU-test overlap with performance measurement.

## Verified changes

- PR2105: offline2035decoded graphs, no graph-dump/exception differences,31269.5ms ->2114.87ms. Boot first presentation36sec versus earlier131–161sec; uncontrolled before/after, not complete-menu loading time. Main menu remained~7FPS.
- Reversed blending: restored three merge-removed pieces, adapted to runtime component mapping, added dual-source device/attachment guards.300GPU word/pixel cases and existing color-target suite passed. In reversed-blend-menu, seven reported skip windows contain8319packed-format errors and no reversed-blend rejection. User confirmed6.55FPS, smoother/quicker loading, black foreground remained.
- Shared shader key no longer includes runtime targetExportMapping; specialized keys still include it. Pixel replay and full shader-disk-cache suite passed.
- Dense specialization: production baseline exactly matched prior offline402case comparison; zero byte mismatches,10182ms ->3702ms. Built and passed full disk-cache,300blend GPU cases, native unnormalized sampling GPU suite. Live benefit not isolated.
- Packed float: captured16actual rejected draw states, all format7/number7/swap2,3840x2160,no DCC/CMASK, fullRGB mask7or15.15disable blending; one uses ADD ONE/ZERO replacement. New Float11_11_10 packing renders into R32_UINT, component mapping0xc6, reuses existing format43 float store conversion. Unsupported actual blending, partialRGB and compression still reject. Adapted output specialization architecture from PR1485, not its unrelated UNORM feature.14exact GPU readback cases cover linear/tiled targets, FP32 and compressed FP16 exports, round-even ties, tiny values, negatives and scissor preservation. State negative/positive checks, pixel/legacy replay, full disk-cache and ordinarycolor/300blend regression tests passed. Actual registered draw/alias/game result still needs live verification.

## Current source caveats

Request serialization version14 appends8packing words after localversion13 reversedBlend. PR2302 also usesversion14 with a DIFFERENT layout: reconcile versions before importing it. No claim that arbitrary destination blending, other component swaps, compressed packed targets or partialRGB writes are fixed. Review new FragmentOutputSpecialization.hpp and runtime module/memo/draw packing keys when extending.

Sampler offset repair now built but NOT deployed to PID7036. Static NativeOffsetUnsupported metadata covers regular samples too. Unavailable native branches emit Undef; descriptor preparation strictly rejects their selection (including indirect child slots) unless a supported unnormalized/emulated/constant mode is selected. Focused --unnormalized and full shader-memory suites pass; disk-cache suite passes; GPU native-only suite passes with nonConstantImageOffsets forced false. libs built. Full unnormalized GPU suite still expects wrap-on-X rejection despite current acceptance; verify wrap results before updating that test. See BOOT-RESULTS for details.

## Remaining observed failures

-204registration SOP1opcode33 failures (201compute,3pixel). PR2302 remains unported; proven user-data compute calls only, no arbitrary loaded targets or pixel integration. Raw captures lack invocation pointers/callee memory. Startup registration failure does not prove runtime dispatch skip or FPS causality.
-7MIMG failures, instruction0xf1949f05/opcode0xe5, no verified semantics. Do not guess BVH aliases.
-2conditional-block6 missing structured merge, static bindless image capacity, GetImageResource runtime provenance, sampler heap capacity, ABIregister201/329 and unsupported fetch registration paths remain.
-Previously observed runtime image-interface mismatches need reassessment after packed target draw restoration exposes downstream work.

## Rapid tools / artifacts

`profiling/BOOT-RESULTS.md` has chronological evidence. `profiling/shader-failure-summary.py` separates registration counts and top8 runtime skip windows. `profiling/draw-state-summary.py` decodes captured target blend/mask/format settings.

`APS5_CAPTURE_DRAW_STATE=existingdirectory` captures at most16unique DecodeState-failure register snapshots,1MiB each. Actual captures live in `diagnostics-rapid\packed-target-captures`; none uploaded. `agc_driver_graphics_tests --decode-state file` reproduces the original early rejection offline; successful later decode requires guest-memory remapping (not full draw replay).

Build with `C:\tools\mingw64\bin` first in PATH. Tests need `build\core\libs\libs\unpatched` BEFORE patched libs, then MinGW. `cmake --build build --target libs -j8` patches deployment libraries. boot-run.ps1 copies patched libs and archives a bounded run.

## Previous run crash

`dense-specialization-menu` PID16440 stopped at361sec, unhandled std::logic_error `sceWebBrowserDialogTerminate: not initialized or still running`, SIGABRT/0xc0000409. Packed-float changes were not deployed in that run. It captured the failing draw states. Last41FPS title was scene unknown; not a menu benchmark. Observer47813 completed and archived.

## External communication

PR2302 comment https://github.com/boykopovar/AnyPS5/pull/2302#issuecomment-6084211175 was explicitly authorized, then shortened at user's request to performance gains only. Do not restore the detailed findings without being asked. PR1485 is reference only; no merge or GitHub comment. Existing PR monitor remains read-only and quiet without actionable new candidates.

