"""Whole-race A/B of live flags in a running Music Rally session (one state per race).

usage: raceab.py <run tag> A:flag=1,flag2=0 B:flag=0 [--races 4] [--heavy-from 30] [--heavy-until 85]

Races alternate A,B,B,A,... The flags of a race's stage are written before the race is (re)started, so
pipelines and caches settle during the restart. Inside the heavy window (seconds after the race's VRR
peg) the frame rate is taken from the presents counter, and the live flag `mark` is set to 1 for the
window and back to 0 after it: the driver prints `[live] mark=` lines, so 10 s report lines (such as
[gputime] with APS5_PROFILE_GPU) can be split by race and state afterwards (gputab.py). Use this
rather than racelive.py when the evidence is in periodic report lines rather than the frame rate."""
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from racelive import Session, read_flags, write_flags, mean_se  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tag')
    parser.add_argument('a')
    parser.add_argument('b')
    parser.add_argument('--races', type=int, default=4)
    parser.add_argument('--between', type=float, default=10.0)
    parser.add_argument('--heavy-from', type=float, default=30.0)
    parser.add_argument('--heavy-until', type=float, default=85.0)
    args = parser.parse_args()

    def stage(spec):
        name, _, kv = spec.partition(':')
        return name, dict((k, int(v)) for k, v in (p.split('=') for p in kv.split(',') if p))

    stages = [stage(args.a), stage(args.b)]
    base = read_flags()
    base.pop('mark', None)
    session = Session(args.tag, args.between)
    rates = {stages[0][0]: [], stages[1][0]: []}
    try:
        race = 0
        while race < args.races:
            name, overrides = stages[0] if race % 4 in (0, 3) else stages[1]
            flags = dict(base)
            flags.update(overrides)
            flags['mark'] = 0
            # A race already past the start of its heavy window cannot be measured: wait it out so
            # the next one starts under this race's flags.
            since = session.since_peg() if session.in_race() else None
            if since is not None and since > args.heavy_from:
                print('waiting for the next race (%.0f s into this one)' % since, flush=True)
                write_flags(flags)
                while session.in_race():
                    time.sleep(2)
            write_flags(flags)
            session.ensure_race()
            seq = session.last_seq()
            since = session.since_peg()
            if since is None:
                time.sleep(2)
                continue
            if since < args.heavy_from:
                time.sleep(args.heavy_from - since)
            flags['mark'] = 1
            write_flags(flags)
            end = time.monotonic() + max(0.0, args.heavy_until - max(args.heavy_from, session.since_peg() or args.heavy_from))
            last = session.presents()
            changes = []
            while time.monotonic() < end:
                time.sleep(0.2)
                now, presents = time.monotonic(), session.presents()
                if presents != last:
                    changes.append((now, presents))
                    last = presents
            flags['mark'] = 0
            write_flags(flags)
            if session.last_seq() != seq or not session.in_race():
                print('race %d dropped (race ended inside the window)' % race, flush=True)
                continue
            if len(changes) >= 2 and changes[-1][0] > changes[0][0]:
                rate = (changes[-1][1] - changes[0][1]) / (changes[-1][0] - changes[0][0])
            else:
                rate = float('nan')
            rates[name].append(rate)
            ma, sa = mean_se(rates[stages[0][0]])
            mb, sb = mean_se(rates[stages[1][0]])
            print('race %2d  %-8s %.3f FPS  | %s %.3f +- %.3f (%d)  %s %.3f +- %.3f (%d)' % (race, name, rate, stages[0][0], ma, sa, len(rates[stages[0][0]]), stages[1][0], mb, sb, len(rates[stages[1][0]])), flush=True)
            race += 1
            while session.in_race():
                time.sleep(2)
    finally:
        write_flags(base)
    ma, sa = mean_se(rates[stages[0][0]])
    mb, sb = mean_se(rates[stages[1][0]])
    print('== %s %.3f +- %.3f FPS, %s %.3f +- %.3f FPS, %s-%s %+.3f (%+.1f%%)' % (stages[0][0], ma, sa, stages[1][0], mb, sb, stages[1][0], stages[0][0], mb - ma, 100 * (mb - ma) / max(1e-9, ma)), flush=True)


if __name__ == '__main__':
    main()
