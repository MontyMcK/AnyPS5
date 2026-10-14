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
Without calibration lines in cpu-timeline.txt the CPU times are steady-clock ns; the offset to the
GPU domain is then estimated from the earliest start of a starved batch after its submit.
"""
import argparse
import collections
import os
import sys


def read_gpu(path):
    batches = []
    for line in open(path):
        parts = line.split()
        if len(parts) < 4:
            continue
        begin, end, key, kind = int(parts[0]), int(parts[1]), int(parts[2], 16), parts[3]
        if kind == 'batch':
            batches.append([begin, end, []])
        elif batches:
            batches[-1][2].append((begin, end, key, kind))
    batches.sort(key=lambda b: b[0])
    return batches


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
    args = parser.parse_args()
    events, rows, calibrated = read_cpu(os.path.join(args.directory, 'cpu-timeline.txt'))
    gpu_path = os.path.join(args.directory, 'gpu-timeline.txt')
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
