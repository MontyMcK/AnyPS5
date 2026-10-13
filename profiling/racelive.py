"""Fast paired A/B of live flags in a running Music Rally session.

usage: racelive.py <run tag> A:flag=1,flag2=0 B:flag=0 [--dwell 20] [--blocks 12] [--settle 5]

Windows of --dwell seconds alternate A,B,B,A (each block cancels a linear drift in the scene's own
load). In each window the flags of its stage are written over the live-flags file
(D:\\ps5\\gt7\\live-flags.txt; the driver re-reads it about once a second), the first --settle seconds
are skipped, and the frame rate over the rest is taken from the automation channel's presents counter.
A window that a race end falls into is dropped; when the race ends the next one is started (Start is
pressed once the pre-race screen is back), so measurement continues across races. After every block
the running paired difference B - A is printed with its standard error, so the run can be stopped as
soon as the answer is clear. The live-flags file is restored at the end."""
import argparse, json, math, os, sys, time

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


def vrr_events(path):
    events = []
    try:
        for line in open(path, encoding='utf-8', errors='replace'):
            if '"vrr"' not in line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('op') in ('peg', 'unpeg'):
                events.append((event['seq'], event['op']))
    except FileNotFoundError:
        pass
    return events


class Session:
    def __init__(self, tag, between):
        self.game = aps5auto.connect(tag)
        self.events_path = self.game.status()['events']
        self.between = between

    def in_race(self):
        events = vrr_events(self.events_path)
        return bool(events) and events[-1][1] == 'peg'

    def since_peg(self):
        # Seconds since the newest peg (the race start), from the events' own clock.
        last = None
        try:
            for line in open(self.events_path, encoding='utf-8', errors='replace'):
                if '"vrr"' in line and '"peg"' in line:
                    last = line
        except FileNotFoundError:
            return None
        if last is None:
            return None
        try:
            peg = json.loads(last)['wall'] / 1000.0
        except (ValueError, KeyError):
            return None
        return time.time() - peg

    def last_seq(self):
        events = vrr_events(self.events_path)
        return events[-1][0] if events else 0

    def ensure_race(self):
        if self.in_race():
            return
        print('starting a race', flush=True)
        time.sleep(self.between)
        # Intros (a track flyover) advance per frame at ~1 FPS and take minutes unless skipped, so
        # Cross is pressed again every 30 s.
        for attempt in range(12):
            since = self.game.cursor()
            # After a race the pre-race menu also offers "Play Music Replay" (a replay, no race
            # start); Left moves the focus back to Start, the leftmost button. Odd attempts skip an
            # intro first (Cross), even ones press Left before Start.
            if attempt % 2 == 1:
                for _ in range(3):
                    self.game.press('left', polls=3, frames=2, ms=300)
            self.game.press('cross', polls=6, frames=4, ms=3000)
            try:
                self.game.wait_for('vrr', where=lambda e: e.get('op') == 'peg', since=since, timeout=30)
                # The event log on disk can trail the pipe; a window must not see this peg land.
                deadline = time.monotonic() + 15
                while not self.in_race() and time.monotonic() < deadline:
                    time.sleep(0.5)
                return
            except TimeoutError:
                print('no race start after press %d' % (attempt + 1), flush=True)
        raise RuntimeError('the race does not start')

    def presents(self):
        return int(self.game.status()['presents'])


def mean_se(values):
    if not values:
        return float('nan'), float('nan')
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, float('nan')
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var / len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tag')
    parser.add_argument('a')
    parser.add_argument('b')
    parser.add_argument('--dwell', type=float, default=20.0)
    parser.add_argument('--settle', type=float, default=5.0)
    parser.add_argument('--blocks', type=int, default=12)
    parser.add_argument('--between', type=float, default=45.0)
    parser.add_argument('--heavy-until', type=float, default=85.0)
    parser.add_argument('--heavy-from', type=float, default=0.0)
    args = parser.parse_args()

    def stage(spec):
        name, _, kv = spec.partition(':')
        return name, dict((k, int(v)) for k, v in (p.split('=') for p in kv.split(',') if p))

    (name_a, flags_a), (name_b, flags_b) = stage(args.a), stage(args.b)
    base = read_flags()
    session = Session(args.tag, args.between)
    diffs, values = [], {name_a: [], name_b: []}
    # Per phase of the race (seconds since the peg at the block's start): the heavy phase is the
    # async-compute-bound part of the race, the light one the pegged results part after it.
    phases = {}
    try:
        block = 0
        while block < args.blocks:
            session.ensure_race()
            # Heavy-phase gating (--heavy-from/--heavy-until > 0): a block starts only inside the
            # race's own heavy window, and one that would not end inside it waits for the next race.
            if args.heavy_from > 0:
                since = session.since_peg()
                if since is None:
                    time.sleep(2)
                    continue
                if since < args.heavy_from:
                    time.sleep(args.heavy_from - since)
                    since = session.since_peg() or args.heavy_from
                if since + 4 * args.dwell > args.heavy_until:
                    print('waiting for the next race (%.0f s into this one)' % since, flush=True)
                    while session.in_race():
                        time.sleep(2)
                    continue
            measured = {}
            dropped = False
            block_start = session.since_peg()
            for position, (name, overrides) in enumerate(((name_a, flags_a), (name_b, flags_b), (name_b, flags_b), (name_a, flags_a))):
                flags = dict(base)
                flags.update(overrides)
                write_flags(flags)
                seq = session.last_seq()
                time.sleep(args.settle)
                # The rate from the moments the presents counter changed (polled every 0.2 s), not the
                # window edges: at ~1 FPS a window holds only ~10 presents, so edge timing alone
                # quantizes the rate to +-10%.
                end = time.monotonic() + max(0.0, args.dwell - args.settle)
                last = session.presents()
                changes = []
                while time.monotonic() < end:
                    time.sleep(0.2)
                    now, presents = time.monotonic(), session.presents()
                    if presents != last:
                        changes.append((now, presents))
                        last = presents
                if session.last_seq() != seq or not session.in_race():
                    dropped = True
                    break
                if len(changes) >= 2 and changes[-1][0] > changes[0][0]:
                    rate = (changes[-1][1] - changes[0][1]) / (changes[-1][0] - changes[0][0])
                else:
                    rate = len(changes) / max(1e-9, args.dwell - args.settle)
                measured.setdefault(name, []).append(rate)
            if dropped:
                print('block %d dropped (race boundary)' % block, flush=True)
                continue
            a = sum(measured[name_a]) / 2
            b = sum(measured[name_b]) / 2
            phase = 'heavy' if block_start is not None and block_start < args.heavy_until else 'light'
            phases.setdefault(phase, []).append(b - a)
            values[name_a].append(a)
            values[name_b].append(b)
            diffs.append(b - a)
            mean, se = mean_se(diffs)
            print('block %2d  %-5s %s %.3f  %s %.3f  diff %+.3f  | running %s-%s %+.3f +- %.3f FPS (%+.1f%% of %s, %d blocks)'
                  % (block, phase, name_a, a, name_b, b, b - a, name_b, name_a, mean, se, 100 * mean / max(1e-9, sum(values[name_a]) / len(values[name_a])), name_a, len(diffs)), flush=True)
            block += 1
    finally:
        write_flags(base)
    mean, se = mean_se(diffs)
    ma, _ = mean_se(values[name_a])
    mb, _ = mean_se(values[name_b])
    print('== %s %.3f FPS, %s %.3f FPS, %s-%s %+.3f +- %.3f over %d blocks' % (name_a, ma, name_b, mb, name_b, name_a, mean, se, len(diffs)), flush=True)
    for phase, phase_diffs in sorted(phases.items()):
        pm, ps = mean_se(phase_diffs)
        print('== %s phase: %s-%s %+.3f +- %.3f over %d blocks' % (phase, name_b, name_a, pm, ps, len(phase_diffs)), flush=True)


if __name__ == '__main__':
    main()
