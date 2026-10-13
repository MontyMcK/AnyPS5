"""Race-by-race A/B in one running Music Rally session.

usage: racelab.py <run tag> <races> name:flag=0,flag=1 name2:flag=1 ...

After a race the game returns to the pre-race screen, and Start begins the same race again in a few
seconds (no reload). Each race here runs one stage: its flags are written over the live-flags file
(D:\\ps5\\gt7\\live-flags.txt), Start is held until the VRR peg (race start), and frames are counted
from the presents counter until the VRR un-peg (race end). Stages alternate race by race, so every
stage sees the same race content. If a race is already running when this starts, it waits for it to
end first. The live-flags file is restored at the end."""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aps5auto  # noqa: E402

FLAGS = r'D:\ps5\gt7\live-flags.txt'


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


def presents(game):
    status = game.status()
    for key in ('presents', 'present', 'frames'):
        if key in status:
            return int(status[key])
    raise RuntimeError('status has no present counter: %s' % status)


def last_vrr(events_path):
    """(op, seq) of the newest VRR event in the run's event log."""
    op, seq = None, 0
    try:
        for line in open(events_path, encoding='utf-8', errors='replace'):
            if '"vrr"' not in line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            op, seq = event.get('op', op), event.get('seq', seq)
    except FileNotFoundError:
        pass
    return op, seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tag')
    parser.add_argument('races', type=int)
    parser.add_argument('stages', nargs='+')
    parser.add_argument('--race-timeout', type=float, default=900)
    parser.add_argument('--skip', type=float, default=0.0, help='seconds after the race start not counted')
    parser.add_argument('--between', type=float, default=45.0, help='seconds after a race ends before Start (the pre-race screen and its Start label return)')
    args = parser.parse_args()
    base = read_flags()
    stages = []
    for spec in args.stages:
        name, _, kv = spec.partition(':')
        stages.append((name, dict((k, int(v)) for k, v in (p.split('=') for p in kv.split(',') if p))))
    game = aps5auto.connect(args.tag)
    results = {name: [] for name, _ in stages}
    try:
        # A race in progress (pegged, not yet un-pegged) is waited out.
        op, seq = last_vrr(game.status()['events'])
        if op == 'peg':
            print('a race is running; waiting for it to end', flush=True)
            game.wait_for('vrr', where=lambda e: e.get('op') == 'unpeg', since=seq, timeout=args.race_timeout)
            time.sleep(args.between)
        for race in range(args.races):
            name, overrides = stages[race % len(stages)]
            flags = dict(base)
            flags.update(overrides)
            write_flags(flags)
            time.sleep(2.0)
            started = None
            for attempt in range(3):
                since = game.cursor()
                game.press('cross', polls=6, frames=4, ms=3000)
                try:
                    started = game.wait_for('vrr', where=lambda e: e.get('op') == 'peg', since=since, timeout=120)
                    break
                except TimeoutError:
                    print('race %d: no start after press %d' % (race, attempt + 1), flush=True)
            if started is None:
                print('race %d: never started; stopping' % race, flush=True)
                break
            cursor = game.cursor()
            if args.skip > 0:
                time.sleep(args.skip)
            t0, p0 = time.monotonic(), presents(game)
            try:
                game.wait_for('vrr', where=lambda e: e.get('op') == 'unpeg', since=cursor, timeout=args.race_timeout)
            except TimeoutError:
                print('race %d: no end within %d s' % (race, args.race_timeout), flush=True)
            t1, p1 = time.monotonic(), presents(game)
            fps = (p1 - p0) / (t1 - t0) if t1 > t0 else 0.0
            results[name].append(fps)
            print('race %d %-10s %.3f FPS (%d frames in %.0f s)' % (race, name, fps, p1 - p0, t1 - t0), flush=True)
            time.sleep(args.between)
    finally:
        write_flags(base)
    for name, values in results.items():
        mean = sum(values) / len(values) if values else float('nan')
        print('== %-10s mean %.3f FPS over %d races: %s' % (name, mean, len(values), ' '.join('%.3f' % v for v in values)), flush=True)


if __name__ == '__main__':
    main()
