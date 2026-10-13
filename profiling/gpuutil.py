"""GPU utilisation next to the frame rate in a running session.

usage: gpuutil.py <run tag> [--seconds 60] [--race]

Samples nvidia-smi (GPU busy %, SM clock, power) once a second together with the automation
channel's presents counter. With --race a race is started first (as racelive.py does)."""
import argparse, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import racelive  # noqa: E402


def gpu():
    out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,clocks.sm,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    util, clock, power = (x.strip() for x in out.strip().split(','))
    return int(util), int(clock), float(power)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tag')
    parser.add_argument('--seconds', type=float, default=60.0)
    parser.add_argument('--race', action='store_true')
    parser.add_argument('--between', type=float, default=45.0)
    args = parser.parse_args()
    session = racelive.Session(args.tag, args.between)
    if args.race:
        session.ensure_race()
    utils = []
    t0, p0 = time.monotonic(), session.presents()
    last_t, last_p = t0, p0
    while time.monotonic() - t0 < args.seconds:
        time.sleep(1.0)
        util, clock, power = gpu()
        now, presents = time.monotonic(), session.presents()
        utils.append(util)
        print('%5.1f s  gpu %3d%%  sm %4d MHz  %5.1f W  fps %.2f  race %d' % (now - t0, util, clock, power, (presents - last_p) / (now - last_t), session.in_race()), flush=True)
        last_t, last_p = now, presents
    t1, p1 = time.monotonic(), session.presents()
    print('== mean gpu %.1f%%, fps %.3f over %.0f s' % (sum(utils) / max(1, len(utils)), (p1 - p0) / (t1 - t0), t1 - t0), flush=True)


if __name__ == '__main__':
    main()
