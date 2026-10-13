"""A/B live flags inside one running race.

usage: liveab2.py <run tag> <dwell s> <rounds> name:flag=0,flag=1 name2:flag=1 ...

Each round runs every stage once, in order: the stage's flags are written over the live-flags file
(D:\\ps5\\gt7\\live-flags.txt, read by the driver about once a second), the first --settle seconds are
skipped, and the frame rate over the rest of the dwell is taken from the presents counter in the
run's samples.jsonl (the "(N)" after the title FPS). The file is restored at the end. Stops early when
the race ends (a VRR un-peg in the automation events)."""
import argparse, datetime, json, os, sys, time

FLAGS = r'D:\ps5\gt7\live-flags.txt'
ROOT = r'D:\ps5\gt7\diagnostics-rapid'


def read_flags():
    flags = {}
    try:
        for line in open(FLAGS, encoding='utf-8'):
            name, _, value = line.strip().partition('=')
            if name and value:
                flags[name] = int(value)
    except FileNotFoundError:
        pass
    return flags


def write_flags(flags):
    with open(FLAGS + '.tmp', 'w', encoding='utf-8') as f:
        f.write(''.join('%s=%d\n' % kv for kv in flags.items()))
    os.replace(FLAGS + '.tmp', FLAGS)


def samples(tag):
    rows = []
    for line in open(os.path.join(ROOT, tag, 'samples.jsonl'), encoding='utf-8', errors='replace'):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        title = row.get('title', '')
        if '(' in title and title.rstrip().endswith(')'):
            try:
                presents = int(title.rsplit('(', 1)[1].rstrip(') '))
            except ValueError:
                continue
            utc = datetime.datetime.fromisoformat(row['utc'].rstrip('Z')[:26])
            rows.append((utc, presents))
    return rows


def rate(tag, start, end):
    rows = [r for r in samples(tag) if start <= r[0] <= end]
    if len(rows) < 2:
        return None
    (t0, p0), (t1, p1) = rows[0], rows[-1]
    seconds = (t1 - t0).total_seconds()
    return (p1 - p0) / seconds if seconds > 0 else None


def race_ended(tag, since_seq):
    path = os.path.join(ROOT, tag, 'automation-events.jsonl')
    try:
        for line in open(path, encoding='utf-8', errors='replace'):
            event = json.loads(line)
            if event.get('type') == 'vrr' and event.get('op') == 'unpeg' and event.get('seq', 0) > since_seq:
                return True
    except (FileNotFoundError, ValueError):
        pass
    return False


def last_seq(tag):
    seq = 0
    try:
        for line in open(os.path.join(ROOT, tag, 'automation-events.jsonl'), encoding='utf-8', errors='replace'):
            try:
                seq = max(seq, json.loads(line).get('seq', 0))
            except ValueError:
                pass
    except FileNotFoundError:
        pass
    return seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tag')
    parser.add_argument('dwell', type=float)
    parser.add_argument('rounds', type=int)
    parser.add_argument('stages', nargs='+')
    parser.add_argument('--settle', type=float, default=6.0)
    args = parser.parse_args()
    base = read_flags()
    stages = []
    for spec in args.stages:
        name, _, kv = spec.partition(':')
        stages.append((name, dict((k, int(v)) for k, v in (p.split('=') for p in kv.split(',') if p))))
    results = {name: [] for name, _ in stages}
    start_seq = last_seq(args.tag)
    try:
        for round_index in range(args.rounds):
            for name, overrides in stages:
                if race_ended(args.tag, start_seq):
                    print('race ended; stopping', flush=True)
                    raise StopIteration
                flags = dict(base)
                flags.update(overrides)
                write_flags(flags)
                begin = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                time.sleep(args.dwell)
                end = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                fps = rate(args.tag, begin + datetime.timedelta(seconds=args.settle), end)
                if fps is not None:
                    results[name].append(fps)
                print('round %d %-10s %s FPS' % (round_index, name, '%.2f' % fps if fps is not None else 'n/a'), flush=True)
    except StopIteration:
        pass
    finally:
        write_flags(base)
    for name, values in results.items():
        mean = sum(values) / len(values) if values else float('nan')
        print('== %-10s mean %.3f FPS over %d windows: %s' % (name, mean, len(values), ' '.join('%.2f' % v for v in values)), flush=True)


if __name__ == '__main__':
    main()
