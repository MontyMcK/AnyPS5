"""GPU idle ("bubbles") from an APS5_GPU_TIMELINE capture.

Reads gpu-timeline.txt (the recorder's timed ranges, with APS5_PROFILE_GPU; optional) and
cpu-timeline.txt (submits with their
reason, batches with serials, waits, GPU mutex holds, commits, EOP deliveries, flips, queue
dequeues) from one window and prints:
  - batches per frame by submit reason (frames = flip events),
  - GPU idle per frame by the reason of the batch that ended each gap, split into gaps where the
    GPU had nothing queued at that submit ("starved") and boundary gaps with work queued,
  - every long gap (default >= 5 ms) with the CPU events that overlap it.

usage: python bubbles.py <dir with both files> [--long-ms 5] [--show 20]
       python bubbles.py <dir> --by-pass [--top 25] [--frames N]
Without calibration lines in cpu-timeline.txt the CPU times are steady-clock ns; the offset to the
GPU domain is then estimated from the earliest start of a starved batch after its submit.

--by-pass reads the identity fields the recorder appends to gpu-timeline.txt lines (libs bs41+;
see Recorder::readGpuTiming) and prints, per frame (flips in cpu-timeline.txt; --frames overrides,
and is needed without cpu-timeline.txt):
  - draw passes grouped by their targets (first color target + depth target: address, size and
    VkFormat): GPU ms, ranges and draws per frame, with the pixel/vertex programs that drew most,
  - pixel programs by the draw-range ms attributed to them (each range's ms split by its draws),
  - storage surfaces (storage-writeback, storage-upload; staging-in/out buffers) grouped by guest
    address, size and format: GPU ms and MiB moved per frame, with the paths that moved them.
Line format: <begin ns> <end ns> <key hex> <name> <bytes> [key=value ...], where a draw range has
n=<draws> rt=0x<addr>:<w>x<h>:<VkFormat>[+<more targets>]|none ds=...|none ps=0x<addr>:<draws>,...
[,other:<draws>] vs=..., and a surface range surf=0x<addr> size=<bytes> fmt=<VkFormat>
gfmt=<guest format> dim=<w>x<h> kind=<path> pieces=<regions>.
"""
import argparse
import collections
import os
import sys


def read_gpu(path):
    """Batches as [begin, end, ranges]; a range is (begin, end, key, kind, bytes, fields), with
    `fields` the key=value identity fields appended to the line (empty on older captures)."""
    batches = []
    for line in open(path):
        parts = line.split()
        if len(parts) < 4:
            continue
        begin, end, key, kind = int(parts[0]), int(parts[1]), int(parts[2], 16), parts[3]
        moved = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0
        fields = dict(part.split('=', 1) for part in parts[5:] if '=' in part)
        if kind == 'batch':
            batches.append([begin, end, []])
        elif batches:
            batches[-1][2].append((begin, end, key, kind, moved, fields))
    batches.sort(key=lambda b: b[0])
    return batches


def program_counts(text):
    """'0xa:3,0xb:1,other:2' -> {'0xa': 3, '0xb': 1, 'other': 2}."""
    counts = {}
    for item in text.split(',') if text else []:
        name, _, count = item.rpartition(':')
        if name and count.isdigit():
            counts[name] = counts.get(name, 0) + int(count)
    return counts


VK_FORMATS = {9: 'R8_UNORM', 16: 'RG8_UNORM', 37: 'RGBA8_UNORM', 43: 'RGBA8_SRGB', 44: 'BGRA8_UNORM', 50: 'BGRA8_SRGB',
              64: 'A2B10G10R10_UNORM', 70: 'R16_UNORM', 74: 'R16_UINT', 76: 'R16_SFLOAT', 77: 'RG16_UNORM',
              83: 'RG16_SFLOAT', 91: 'RGBA16_UNORM', 95: 'RGBA16_UINT', 97: 'RGBA16_SFLOAT', 98: 'R32_UINT',
              100: 'R32_SFLOAT', 101: 'RG32_UINT', 103: 'RG32_SFLOAT', 107: 'RGBA32_UINT', 109: 'RGBA32_SFLOAT',
              122: 'B10G11R11_UFLOAT', 123: 'E5B9G9R9_UFLOAT', 124: 'D16_UNORM', 126: 'D32_SFLOAT',
              129: 'D24_UNORM_S8', 130: 'D32_SFLOAT_S8'}


def format_name(text):
    if not text.isdigit() or text == '0':
        return '-'
    return VK_FORMATS.get(int(text), f'vkfmt{text}')


def target_name(text):
    """'0x1234:3840x2160:97+1' -> '0x1234 3840x2160 RGBA16_SFLOAT +1'."""
    if not text or text == 'none':
        return 'none'
    more = ''
    if '+' in text:
        text, _, extra = text.partition('+')
        more = f' +{extra}'
    address, _, rest = text.partition(':')
    dims, _, fmt = rest.partition(':')
    return f'{address} {dims} {format_name(fmt)}{more}'


def top_programs(counts, limit=3):
    total = sum(counts.values())
    if not total:
        return '-'
    ranked = sorted(counts.items(), key=lambda x: -x[1])[:limit]
    return ' '.join(f'{name}({100.0 * n / total:.0f}%)' for name, n in ranked)


def by_pass(batches, frames, top):
    """Top draw passes, pixel programs and storage surfaces by GPU ms per frame."""
    passes = collections.defaultdict(lambda: {'ms': 0.0, 'ranges': 0, 'draws': 0, 'ps': collections.Counter(), 'vs': collections.Counter()})
    pixel = collections.defaultdict(lambda: {'ms': 0.0, 'draws': 0, 'targets': collections.Counter()})
    surfaces = collections.defaultdict(lambda: {'ms': 0.0, 'bytes': 0, 'ranges': 0, 'kinds': collections.Counter()})
    draw_ms = unnamed_draw_ms = storage_ms = unnamed_storage_ms = 0.0
    storage_kinds = ('storage-writeback', 'storage-upload', 'staging-in', 'staging-out')
    for _, _, ranges in batches:
        for begin, end, key, kind, moved, fields in ranges:
            ms = (end - begin) / 1e6
            # A draw class range, or a pass-keyed one (APS5_PROFILE_GPU_PASSES) that noted draws.
            if kind == 'draw' or (kind == 'program' and 'n' in fields):
                draw_ms += ms
                if 'n' not in fields:
                    unnamed_draw_ms += ms
                    continue
                rt, ds = target_name(fields.get('rt')), target_name(fields.get('ds'))
                entry = passes[(rt, ds)]
                entry['ms'] += ms
                entry['ranges'] += 1
                entry['draws'] += int(fields['n'])
                ps, vs = program_counts(fields.get('ps', '')), program_counts(fields.get('vs', ''))
                entry['ps'].update(ps)
                entry['vs'].update(vs)
                total = sum(ps.values()) or 1
                for name, n in ps.items():
                    pixel[name]['ms'] += ms * n / total
                    pixel[name]['draws'] += n
                    pixel[name]['targets'][rt] += n
            elif kind in storage_kinds:
                storage_ms += ms
                if 'surf' not in fields:
                    unnamed_storage_ms += ms
                    continue
                group = 'staging' if kind.startswith('staging') else kind
                if group == 'staging':
                    name = f"{fields['surf']} buffer"
                else:
                    name = f"{fields['surf']} {int(fields.get('size', 0)) / 1048576.0:.2f}MiB {fields.get('dim', '?')} {format_name(fields.get('fmt', '0'))} gfmt{fields.get('gfmt', '?')}"
                entry = surfaces[(group, name)]
                entry['ms'] += ms
                entry['bytes'] += moved
                entry['ranges'] += 1
                entry['kinds'][f"{kind}/{fields.get('kind', '?')}"] += 1
    f = float(frames)
    print(f'\n== draw passes: {draw_ms / f:.2f} ms/frame in draw ranges ({unnamed_draw_ms / f:.2f} without identity fields: older libs, or ranges outside the identity window)')
    print(f'{"ms/fr":>7} {"%":>5} {"rng/fr":>7} {"drw/fr":>7} {"us/rng":>7}  color target | depth target | pixel programs | vertex programs')
    for (rt, ds), e in sorted(passes.items(), key=lambda x: -x[1]['ms'])[:top]:
        share = 100.0 * e['ms'] / draw_ms if draw_ms else 0.0
        print(f"{e['ms'] / f:7.2f} {share:5.1f} {e['ranges'] / f:7.1f} {e['draws'] / f:7.1f} {1000.0 * e['ms'] / e['ranges']:7.0f}  {rt} | {ds} | ps {top_programs(e['ps'])} | vs {top_programs(e['vs'])}")
    print("\n== pixel programs by attributed draw ms (each range's ms split by its draws)")
    print(f'{"ms/fr":>7} {"drw/fr":>7}  program  (main color target)')
    for name, e in sorted(pixel.items(), key=lambda x: -x[1]['ms'])[:top]:
        target = e['targets'].most_common(1)[0][0] if e['targets'] else '-'
        print(f"{e['ms'] / f:7.2f} {e['draws'] / f:7.1f}  {name}  ({target})")
    print(f'\n== storage surfaces: {storage_ms / f:.2f} ms/frame in storage/staging ranges ({unnamed_storage_ms / f:.2f} without identity fields)')
    for group in ('storage-writeback', 'storage-upload', 'staging'):
        rows = sorted(((name, e) for (g, name), e in surfaces.items() if g == group), key=lambda x: -x[1]['ms'])
        if not rows:
            continue
        group_ms = sum(e['ms'] for _, e in rows)
        group_mb = sum(e['bytes'] for _, e in rows) / 1048576.0
        print(f'-- {group}: {group_ms / f:.2f} ms/frame, {group_mb / f:.2f} MiB/frame moved, {len(rows)} surfaces')
        print(f'{"ms/fr":>7} {"MiB/fr":>7} {"rng/fr":>7}  surface (address size dims VkFormat guest-format)  paths')
        for name, e in rows[:top]:
            paths = ' '.join(f'{k}x{n}' for k, n in e['kinds'].most_common(3))
            print(f"{e['ms'] / f:7.2f} {e['bytes'] / 1048576.0 / f:7.2f} {e['ranges'] / f:7.1f}  {name}  {paths}")


def read_cpu(path):
    events, rows, calibrated = [], {}, False
    for line in open(path):
        if line.startswith('#'):
            calibrated = calibrated or line.startswith('# calibration gpu_ns')
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        begin, end, kind, thread = float(parts[0]), float(parts[1]), parts[2], parts[3]
        a, b = int(parts[5]), int(parts[6])
        text = ' '.join(parts[7:])
        if kind == 'batch':
            reason, _, starved = parts[7].partition('/')
            submit = float(parts[9]) if len(parts) > 9 else 0.0
            rows[(int(begin), int(end))] = {'serial': a, 'queue': b, 'reason': reason, 'starved': starved == '1', 'submit': submit}
        else:
            events.append((begin, end, kind, thread, a, b, text))
    return events, rows, calibrated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--long-ms', type=float, default=5.0)
    parser.add_argument('--show', type=int, default=20)
    parser.add_argument('--by-pass', action='store_true', help='top draw passes, pixel programs and storage surfaces by GPU ms per frame (gpu-timeline.txt with identity fields)')
    parser.add_argument('--top', type=int, default=25, help='rows per --by-pass table')
    parser.add_argument('--frames', type=int, default=0, help='frames in the window for --by-pass (default: flips in cpu-timeline.txt)')
    args = parser.parse_args()
    cpu_path = os.path.join(args.directory, 'cpu-timeline.txt')
    gpu_path = os.path.join(args.directory, 'gpu-timeline.txt')
    if args.by_pass and not os.path.exists(gpu_path):
        sys.exit('--by-pass needs gpu-timeline.txt (APS5_GPU_TIMELINE with APS5_PROFILE_GPU)')
    if args.by_pass and (args.frames or not os.path.exists(cpu_path)):
        batches = read_gpu(gpu_path)
        if not batches:
            sys.exit('no batches in gpu-timeline.txt')
        frames = args.frames or 1
        note = '' if args.frames else ' (no cpu-timeline.txt: totals for the whole window; pass --frames N for per-frame figures)'
        print(f'window {(batches[-1][1] - batches[0][0]) / 1e6:.0f} ms, {len(batches)} batches, {frames} frames{note}')
        by_pass(batches, frames, args.top)
        return
    events, rows, calibrated = read_cpu(cpu_path)
    # Without APS5_PROFILE_GPU there is no gpu-timeline.txt: the batch rows carry the stamps.
    batches = read_gpu(gpu_path) if os.path.exists(gpu_path) else sorted([begin, end, []] for begin, end in rows)
    if not batches:
        sys.exit('no batches in gpu-timeline.txt')
    offset = 0.0
    if not calibrated:
        deltas = [b[0] - rows[(b[0], b[1])]['submit'] for b in batches if (b[0], b[1]) in rows and rows[(b[0], b[1])]['starved']]
        offset = min(deltas) if deltas else 0.0
        print(f'no calibration: CPU times shifted by an estimated {offset:.0f} ns (earliest starved batch start after its submit)')
    events = [(e[0] + offset, e[1] + offset) + e[2:] for e in events]
    flips = sorted(e[0] for e in events if e[2] == 'flip')
    t0, t1 = batches[0][0], batches[-1][1]
    frames = max(1, len([f for f in flips if t0 <= f <= t1]) - 1) if len(flips) > 1 else 1
    busy = sum(b[1] - b[0] for b in batches)
    print(f'window {(t1 - t0) / 1e6:.0f} ms, {len(batches)} batches, {frames} frames, GPU busy {busy / 1e6:.0f} ms ({100.0 * busy / (t1 - t0):.0f}%)')
    if args.by_pass:
        by_pass(batches, frames, args.top)
        return
    by_reason = collections.Counter()
    idle = collections.defaultdict(float)
    starved_idle = queued_idle = 0.0
    long_gaps = []
    for i, batch in enumerate(batches):
        row = rows.get((batch[0], batch[1]))
        reason = row['reason'] if row else '?'
        by_reason[reason] += 1
        if i == 0:
            continue
        gap = (batch[0] - batches[i - 1][1]) / 1e6
        if gap <= 0:
            continue
        idle[reason] += gap
        if row and row['starved']:
            starved_idle += gap
        else:
            queued_idle += gap
        if gap >= args.long_ms:
            long_gaps.append((gap, i, reason))
    print('batches per frame by reason: ' + ', '.join(f'{r} {n / frames:.1f}' for r, n in by_reason.most_common()))
    total_idle = sum(idle.values())
    print(f'GPU idle per frame {total_idle / frames:.1f} ms (nothing queued at the next submit {starved_idle / frames:.1f}, boundary with work queued {queued_idle / frames:.1f}); by the next batch\'s reason: ' + ', '.join(f'{r} {ms / frames:.1f}' for r, ms in sorted(idle.items(), key=lambda x: -x[1])))
    print(f'{len(long_gaps)} gaps >= {args.long_ms} ms, {sum(g for g, _, _ in long_gaps):.0f} ms in total')
    for gap, i, reason in sorted(long_gaps, reverse=True)[:args.show]:
        begin, end = batches[i - 1][1], batches[i][0]
        print(f'-- gap {gap:.1f} ms at +{(begin - t0) / 1e6:.1f} ms, ended by a {reason} batch')
        overlapping = [e for e in events if e[1] >= begin and e[0] <= end and e[2] not in ('submit',)]
        summary = collections.defaultdict(lambda: [0, 0.0])
        for e in overlapping:
            covered = (min(e[1], end) - max(e[0], begin)) / 1e6
            label = f'{e[2]} {e[3]}' + (f' {e[6]}' if e[2] in ('hold', 'wait-hook') and e[6] else '')
            summary[label][0] += 1
            summary[label][1] += max(covered, 0.0)
        for label, (count, ms) in sorted(summary.items(), key=lambda x: (-x[1][1], -x[1][0]))[:12]:
            print(f'     {label}: x{count}, {ms:.1f} ms of the gap')


if __name__ == '__main__':
    main()
