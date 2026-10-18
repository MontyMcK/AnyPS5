#!/usr/bin/env python3
"""Summarize a diagnostic session's run-err.log: storage-image flush pairs and speculation refusals.

  python diag-summary.py D:\\ps5\\gt7\\run-err.log [--skip-windows 3] [--top 25]

Reads:
  [flush] ... flushed {IMAGE} by {FORCER} site SITE      (APS5_TRACE_FLUSH=1, bs48 format)
  [alias-flush] top / bytes (10 s): ...                  (APS5_TRACE_ALIAS_FLUSH=1)
  [spec-refusals] ...                                    (APS5_PROFILE_DRAW=1, bs48)
  [commit] (10 s) ...                                    (APS5_PROFILE_DRAW=1)

Prints: flush pairs (flushed image -> image whose refresh forced it, by reason and site) with
counts; the [alias-flush] pair lists summed over windows; and the [spec-refusals] classes and
sites summed over the 10 s windows (counts and held ms), plus the mean [commit] line numbers.
The first --skip-windows 10 s report windows (boot, map load) can be skipped.
"""
import argparse
import collections
import re
import sys

FLUSH = re.compile(r'^\[flush\] image (0x[0-9a-f]+)\+(0x[0-9a-f]+) for (.+?) (0x[0-9a-f]+)\+(0x[0-9a-f]+) \(packet 0x[0-9a-f]+\)(?: flushed \{([^}]*)\} by \{([^}]*)\} site (\S+))?')
ALIAS = re.compile(r'^\[alias-flush\] (top|bytes) \(10 s\):(.*)$')
CLASS = re.compile(r'^\[spec-refusals\] (.+?): (\d+) commits at ([\d.]+) us \(([\d.]+) ms\), (\d+) sites')
SITE = re.compile(r'^\[spec-refusals\]   (\d+) at ([\d.]+) us: (.*)$')
COMMIT = re.compile(r'^\[commit\] \(10 s\) (\d+) draw commits.*?held ([\d.]+) \(.*?Graphics::Draw ([\d.]+)')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('log')
    parser.add_argument('--skip-windows', type=int, default=0, help='10 s report windows to skip at the start')
    parser.add_argument('--top', type=int, default=25)
    args = parser.parse_args()

    pairs = collections.Counter()
    pair_reasons = collections.defaultdict(collections.Counter)
    alias_top = collections.Counter()
    alias_bytes = collections.Counter()
    classes = collections.defaultdict(lambda: [0, 0.0])
    sites = collections.defaultdict(lambda: [0, 0.0])
    commits = []
    window = 0
    current = None
    with open(args.log, encoding='utf-8', errors='replace') as handle:
        for line in handle:
            line = line.rstrip('\n')
            if line.startswith('[commit] (10 s)'):
                window += 1
                m = COMMIT.match(line)
                if m and window > args.skip_windows:
                    commits.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
                continue
            if window < args.skip_windows:
                continue
            m = FLUSH.match(line)
            if m:
                flushed = m.group(6) or f'{m.group(1)}+{m.group(2)}'
                forcer = m.group(7) or '?'
                key = f'{{{flushed}}} -> {{{forcer}}}'
                pairs[key] += 1
                pair_reasons[key][f'{m.group(3)} @{m.group(8) or "?"}'] += 1
                continue
            m = ALIAS.match(line)
            if m:
                for item in m.group(2).split(' | ')[1:]:
                    item = item.strip()
                    if m.group(1) == 'top':
                        mm = re.match(r'(\d+)x (.*)', item)
                        if mm:
                            alias_top[mm.group(2)] += int(mm.group(1))
                    else:
                        mm = re.match(r'(\d+) MiB (\d+)x (.*)', item)
                        if mm:
                            alias_bytes[mm.group(3)] += int(mm.group(1))
                continue
            m = CLASS.match(line)
            if m:
                current = m.group(1)
                classes[current][0] += int(m.group(2))
                classes[current][1] += float(m.group(4))
                continue
            m = SITE.match(line)
            if m and current is not None:
                key = (current, m.group(3))
                sites[key][0] += int(m.group(1))
                sites[key][1] += int(m.group(1)) * float(m.group(2)) / 1000.0

    windows = max(1, len(commits))
    if commits:
        n = sum(c[0] for c in commits) / windows
        print(f'[commit] {len(commits)} windows: {n:.0f} draw commits per 10 s, held {sum(c[1] for c in commits) / windows:.2f} us, Graphics::Draw {sum(c[2] for c in commits) / windows:.2f} us per commit')
    print(f'\n== flush pairs ([flush] lines: {sum(pairs.values())})')
    for key, count in pairs.most_common(args.top):
        reasons = ', '.join(f'{r} {c}' for r, c in pair_reasons[key].most_common(3))
        print(f'{count:9} ({count / windows:.0f} per 10 s)  {key}  [{reasons}]')
    if alias_top:
        print('\n== [alias-flush] pairs by count (summed over windows)')
        for key, count in alias_top.most_common(args.top):
            print(f'{count:9}  {key}')
        print('\n== [alias-flush] pairs by MiB stored (summed)')
        for key, mib in alias_bytes.most_common(args.top):
            print(f'{mib:9} MiB  {key}')
    print(f'\n== [spec-refusals] classes (summed over windows; held ms per 10 s)')
    for name, (count, ms) in sorted(classes.items(), key=lambda kv: -kv[1][1]):
        print(f'{count:9} commits {ms / windows:8.1f} ms/10 s  {count and ms * 1000 / count:7.1f} us each  {name}')
        for (cls, site), (scount, sms) in sorted(((k, v) for k, v in sites.items() if k[0] == name), key=lambda kv: -kv[1][1])[:8]:
            print(f'        {scount:8} {sms / windows:7.1f} ms/10 s  {site}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
