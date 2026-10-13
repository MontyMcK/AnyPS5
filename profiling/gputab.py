"""Split APS5_PROFILE_GPU [gputime] lines by race and flag state (windows from raceab.py's `mark` flag).

usage: gputab.py run-err.log <flag>[,<flag>...] [--top 8]

Only [gputime] lines that fall wholly inside a mark=1 window count (the first line after each mark=1
is skipped: its 10 s straddle the window start). Per state of the flags it prints GPU ms per present:
everything timed, the guest compute programs (main-image shader addresses), and the top programs."""
import argparse
import re
from collections import defaultdict

LIVE = re.compile(r'^\[live\] (\w+)=(-?\d+)')
GPUTIME = re.compile(r'^\[gputime\] (\d+) ms of GPU time in (\d+) batches over 10 s .*?; (\d+) presents;.*?by program: (.*)$')
PROGRAM = re.compile(r'(0x[0-9a-f]+) x(\d+) (\d+)ms')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log')
    parser.add_argument('flags')
    parser.add_argument('--top', type=int, default=8)
    args = parser.parse_args()
    names = args.flags.split(',')
    live = {}
    marked = False
    skip = False
    totals = defaultdict(lambda: {'lines': 0, 'gpu': 0, 'presents': 0, 'batches': 0, 'compute': 0, 'programs': defaultdict(int), 'counts': defaultdict(int)})
    for line in open(args.log, encoding='utf-8', errors='replace'):
        m = LIVE.match(line)
        if m:
            live[m.group(1)] = int(m.group(2))
            if m.group(1) == 'mark':
                marked = live['mark'] == 1
                skip = marked
            continue
        if not marked:
            continue
        m = GPUTIME.match(line)
        if not m:
            continue
        if skip:
            skip = False
            continue
        key = ','.join('%s=%s' % (n, live.get(n, '?')) for n in names)
        t = totals[key]
        t['lines'] += 1
        t['gpu'] += int(m.group(1))
        t['batches'] += int(m.group(2))
        t['presents'] += int(m.group(3))
        for address, count, ms in PROGRAM.findall(m.group(4)):
            value = int(address, 16)
            t['programs'][address] += int(ms)
            t['counts'][address] += int(count)
            if 0x7ff000000000 <= value < 0x800000000000:
                t['compute'] += int(ms)
    for key in sorted(totals):
        t = totals[key]
        presents = max(1, t['presents'])
        print('%s: %d lines, %d presents (%.2f FPS), GPU %.1f ms/present, main-image programs %.1f ms/present, %.0f batches/present'
              % (key, t['lines'], t['presents'], t['presents'] / (10.0 * max(1, t['lines'])), t['gpu'] / presents, t['compute'] / presents, t['batches'] / presents))
        top = sorted(t['programs'].items(), key=lambda kv: -kv[1])[:args.top]
        print('    ' + '  '.join('%s %.1fms x%.0f' % (a, ms / presents, t['counts'][a] / presents) for a, ms in top))


if __name__ == '__main__':
    main()
