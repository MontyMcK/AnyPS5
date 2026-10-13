"""Boots PPSA01317 into a Music Rally race through the automation channel and confirms the race start.

Start the game first with automation on, e.g. (from its own shell):
    best-run.ps1 -Tag mr1 -Seconds 2400 -ExtraEnv 'APS5_AUTOMATION=1','APS5_AUTOMATION_EVENTS=D:\\ps5\\gt7\\diagnostics-rapid\\mr1\\automation-events.jsonl'
then:  python -I nav-musicrally2.py mr1          (a pid works too)
Route: wait for the title screen (instant snapshots + title_focus) -> highlight MUSIC RALLY -> Cross -> wait for the
song list (screen changed, not loading, stable, no file opens) -> Cross on the first song -> wait for the pre-race
screen (same checks; when a reference from an earlier success exists, the lower UI band must match it, i.e. the Start
label is there) -> wait for the Start label -> long Cross press -> wait_for sceVideoOutVrrPegToFixedRate (race start).
Snapshots, nav.log and nav.json go to D:\\ps5\\gt7\\diagnostics-rapid\\<tag>\\nav2 (or --out). Exit 0 = race started.
"""

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aps5auto  # noqa: E402

REFERENCE = os.path.join(aps5auto.GAME_ROOT, 'automation', 'musicrally-prerace.bmp')
# The pre-race reference is compared on the lower band, where the menu's labels and buttons are; the
# preview above may animate.
LABEL_BAND = (0.0, 0.70, 1.0, 1.0)


class Route:
    def __init__(self, args):
        self.args = args
        self.out = args.out or os.path.join(aps5auto.RUNS_ROOT, args.target, 'nav2')
        os.makedirs(self.out, exist_ok=True)
        self.log_file = open(os.path.join(self.out, 'nav.log'), 'a', encoding='utf-8')
        self.started = time.monotonic()
        self.shot = 0
        self.steps = []
        self.game = None

    def log(self, text, **fields):
        stamp = time.strftime('%H:%M:%S')
        line = '%s +%5.0fs %s%s' % (stamp, time.monotonic() - self.started, text, (' ' + json.dumps(fields)) if fields else '')
        print(line, flush=True)
        self.log_file.write(line + '\n')
        self.log_file.flush()
        self.steps.append(dict(fields, t=round(time.monotonic() - self.started, 1), step=text))
        if self.game is not None:
            try:
                self.game.mark(text)
            except aps5auto.AutomationError:
                pass

    def snap(self, name, after=1):
        self.shot += 1
        path = os.path.join(self.out, '%03d-%s.bmp' % (self.shot, name))
        image = self.game.snapshot(path, after=after, timeout=self.args.frame_timeout)
        return image

    def try_snapshot(self, name):
        """A snapshot, or None (logged) when no frame came within --frame-timeout (long loads)."""
        try:
            return self.game.snapshot(os.path.join(self.out, name + '.bmp'), timeout=self.args.frame_timeout)
        except aps5auto.AutomationError as error:
            self.log('no snapshot: %s' % error, screen=self.game.screen())
            time.sleep(self.args.poll_seconds)
            return None

    def press(self, button, polls=2, frames=1, ms=150):
        ack = self.game.press(button, ms=ms, polls=polls, frames=frames, timeout=self.args.frame_timeout)
        self.log('pressed %s' % button, polls=ack.get('polls'), frames=ack.get('frames'), held_ms=ack.get('held_ms'))
        return ack

    def connect(self):
        deadline = time.monotonic() + self.args.boot_timeout
        while True:
            try:
                self.game = aps5auto.connect(self.args.target, connect_timeout=5.0)
                break
            except aps5auto.AutomationError as error:
                if time.monotonic() >= deadline:
                    raise
                print('waiting for the automation pipe: %s' % error, flush=True)
                time.sleep(5)
        self.log('connected', pipe=self.game.pipe)

    @staticmethod
    def loading(image):
        # Black frames and the loader (a small white piano on black) are dark overall.
        return image.luma() < 12.0

    def wait_title(self):
        deadline = time.monotonic() + self.args.boot_timeout
        hits = 0
        while time.monotonic() < deadline:
            image = self.try_snapshot('boot-latest')
            if image is None:
                continue
            state, wm, mr = aps5auto.title_focus(image)
            hits = hits + 1 if state != 'NOTTITLE' else 0
            if hits >= 2:
                shutil.copyfile(image.path, os.path.join(self.out, '000-title.bmp'))
                self.log('title screen', state=state, wm=wm, mr=mr, method=image.info.get('method'))
                return state
            time.sleep(self.args.poll_seconds)
        raise TimeoutError('the title screen did not appear within %d s' % self.args.boot_timeout)

    def highlight_music_rally(self, state):
        # The title moves a spatial cursor; where it starts varies (see nav-musicrally.ps1): World Map
        # -RIGHT-> Music Rally, otherwise try moves until MUSIC RALLY is lit.
        moves = ['down', 'up', 'up', 'down', 'left', 'right', 'down']
        index = 0
        for attempt in range(16):
            if state == 'MR':
                self.log('MUSIC RALLY highlighted', attempt=attempt)
                return
            if state == 'WM':
                move = 'right'
            elif state == 'NOTTITLE':
                raise RuntimeError('left the title screen while looking for MUSIC RALLY')
            else:
                move = moves[index % len(moves)]
                index += 1
            self.press(move)
            image = self.snap('title-%s' % move, after=2)
            state = aps5auto.title_focus(image)[0]
            self.log('title focus', state=state)
        raise RuntimeError('could not highlight MUSIC RALLY')

    def wait_ready(self, name, previous=None, timeout=None, reference=None):
        """Waits until the screen changed from `previous`, is not loading, two snapshots agree and no
        file opened for a few seconds; with `reference`, the label band must also match it."""
        timeout = timeout or self.args.screen_timeout
        deadline = time.monotonic() + timeout
        changed = previous is None
        last = None
        stable = 0
        while time.monotonic() < deadline:
            image = self.try_snapshot('%s-latest' % name)
            if image is None:
                continue
            if not changed:
                changed = self.loading(image) or image.diff(previous) > self.args.change_threshold
            elif not self.loading(image) and last is not None and image.diff(last) < self.args.stable_threshold:
                stable += 1
                # Settled: no guest file opens for a while, or (for titles that keep opening files)
                # the picture held still for several checks.
                if not self.game.recent(self.args.quiet_seconds, 'open') or stable >= self.args.stable_checks:
                    band = image.diff(reference, region=LABEL_BAND) if reference is not None else None
                    if band is None or band < self.args.reference_threshold:
                        self.shot += 1
                        keep = os.path.join(self.out, '%03d-%s.bmp' % (self.shot, name))
                        shutil.copyfile(image.path, keep)
                        image.path = keep
                        self.log('%s ready' % name, luma=round(image.luma(), 1), band_diff=band and round(band, 1), screen=self.game.screen()['guess'])
                        return image
            else:
                stable = 0
            last = image
            time.sleep(self.args.poll_seconds)
        raise TimeoutError('%s did not settle within %d s' % (name, timeout))

    def start_race(self, prerace):
        # Start only reacts once its labels have loaded (about 30 s after the pre-race screen appears),
        # and at ~1 FPS it needs a long hold. The race start is sceVideoOutVrrPegToFixedRate (rg6: the
        # peg came 14 s after the Start press, and none when the song list loaded).
        self.log('waiting for the Start label', seconds=self.args.prerace_settle)
        time.sleep(self.args.prerace_settle)
        prerace = self.snap('prerace-settled')
        for attempt in range(1, self.args.start_presses + 1):
            since = self.game.cursor()
            self.press('cross', polls=6, frames=4, ms=self.args.start_hold_ms)
            try:
                event = self.game.wait_for('vrr', where=lambda e: e.get('op') == 'peg', since=since, timeout=self.args.race_timeout)
            except TimeoutError:
                self.log('no VRR peg after Start press %d' % attempt, screen=self.game.screen())
                continue
            self.log('race started (sceVideoOutVrrPegToFixedRate)', seq=event['seq'], t=event['t'], attempt=attempt)
            if not os.path.exists(REFERENCE):
                os.makedirs(os.path.dirname(REFERENCE), exist_ok=True)
                shutil.copyfile(prerace.path, REFERENCE)
                self.log('saved the pre-race reference', path=REFERENCE)
            return True
        return False

    def run(self):
        self.connect()
        state = self.wait_title()
        self.highlight_music_rally(state)
        title = self.snap('title-mr')
        self.press('cross', polls=3, frames=1, ms=200)
        songs = self.wait_ready('songlist', previous=title)
        self.press('cross', polls=3, frames=1, ms=200)
        reference = aps5auto.Image.load(REFERENCE) if os.path.exists(REFERENCE) and not self.args.no_reference else None
        try:
            prerace = self.wait_ready('prerace', previous=songs, reference=reference)
        except TimeoutError:
            # The pre-race background keeps moving (flyover, scrolling text, idle UI fade), so it may never
            # hold still; take it as reached and wake the UI before waiting for the Start label.
            self.log('prerace did not settle; continuing')
            self.press('up', polls=3, frames=2, ms=300)
            prerace = self.snap('prerace-unsettled')
        started = self.start_race(prerace)
        return started

    def finish(self, ok, error=None):
        summary = {'ok': ok, 'error': error, 'seconds': round(time.monotonic() - self.started, 1), 'steps': self.steps}
        if self.game is not None:
            try:
                summary['status'] = self.game.status()
            except Exception as status_error:  # the game may be gone
                summary['status_error'] = str(status_error)
        with open(os.path.join(self.out, 'nav.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('target', help='run tag (diagnostics-rapid/<tag>/process.json), pid or pipe name')
    parser.add_argument('--out', help='output directory (default diagnostics-rapid/<tag>/nav2)')
    parser.add_argument('--boot-timeout', type=float, default=900, help='seconds to reach the title screen')
    parser.add_argument('--screen-timeout', type=float, default=900, help='seconds for each menu screen to settle')
    parser.add_argument('--race-timeout', type=float, default=180, help='seconds to wait for the VRR peg after a Start press')
    parser.add_argument('--frame-timeout', type=float, default=120, help='seconds a snapshot or press may wait for frames/reads')
    parser.add_argument('--poll-seconds', type=float, default=1.0, help='pause between screen checks')
    parser.add_argument('--quiet-seconds', type=float, default=4.0, help='no guest file opens for this long = settled')
    parser.add_argument('--change-threshold', type=float, default=8.0, help='mean brightness change that counts as a new screen')
    parser.add_argument('--stable-threshold', type=float, default=3.0, help='max change between two snapshots of a settled screen')
    parser.add_argument('--stable-checks', type=int, default=4, help='consecutive still snapshots that also count as settled')
    parser.add_argument('--reference-threshold', type=float, default=12.0, help='max label-band difference to the pre-race reference')
    parser.add_argument('--prerace-settle', type=float, default=35, help='seconds on the pre-race screen before pressing Start')
    parser.add_argument('--start-hold-ms', type=int, default=3000, help='Start press hold in ms')
    parser.add_argument('--start-presses', type=int, default=3, help='Cross presses to try for the race start')
    parser.add_argument('--race-frames', type=int, default=3, help='frames after the peg for the race snapshot')
    parser.add_argument('--no-reference', action='store_true', help='ignore the saved pre-race reference')
    args = parser.parse_args()
    route = Route(args)
    try:
        ok = route.run()
        route.finish(ok, None if ok else 'no race start')
        return 0 if ok else 1
    except Exception as error:
        route.log('failed: %s' % error)
        route.finish(False, str(error))
        return 1


if __name__ == '__main__':
    sys.exit(main())
