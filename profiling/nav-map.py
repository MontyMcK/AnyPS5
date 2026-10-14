"""Boots to the PPSA01317 world map: wait for the title screen, highlight WORLD MAP, Cross, wait for the
map to settle. Reuses nav-musicrally2's Route (snapshots, title focus, settle checks).
Prints "MAP READY" on success. Keeps the map awake afterwards with --keep-awake N (a Right/Left press pair
every N seconds, so the title's screensaver never starts during measurements).
"""
import argparse
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('nav2', os.path.join(HERE, 'nav-musicrally2.py'))
nav2 = importlib.util.module_from_spec(spec)
sys.argv_saved = sys.argv
spec.loader.exec_module(nav2)


class MapRoute(nav2.Route):
    def highlight_world_map(self, state):
        moves = ['down', 'up', 'up', 'down', 'right', 'left', 'down']
        index = 0
        for attempt in range(16):
            if state == 'WM':
                self.log('WORLD MAP highlighted', attempt=attempt)
                return
            if state == 'MR':
                move = 'left'
            elif state == 'NOTTITLE':
                raise RuntimeError('left the title screen while looking for WORLD MAP')
            else:
                move = moves[index % len(moves)]
                index += 1
            self.press(move)
            image = self.snap('title-%s' % move, after=2)
            state = nav2.aps5auto.title_focus(image)[0]
            self.log('title focus', state=state)
        raise RuntimeError('could not highlight WORLD MAP')

    def run(self):
        self.connect()
        state = self.wait_title()
        self.highlight_world_map(state)
        title = self.snap('title-wm')
        self.press('cross', polls=3, frames=1, ms=200)
        self.wait_ready('worldmap', previous=title)
        # The map's 3D scene runs well under 30 FPS; light screens (photo/loading) run at 60. Ready only
        # once the frame rate stays low for three checks in a row.
        deadline = time.monotonic() + 300
        low = 0
        while low < 3:
            if time.monotonic() > deadline:
                raise TimeoutError('the map scene (low frame rate) never appeared')
            time.sleep(2)
            low = low + 1 if self.game.screen()['fps'] < 30 else 0
        self.log('map scene running', fps=self.game.screen()['fps'])
        print('MAP READY', flush=True)
        return True

    def keep_awake(self, every, total):
        end = time.monotonic() + total
        flip = 1
        while time.monotonic() < end:
            time.sleep(every)
            try:
                self.game.press('right' if flip > 0 else 'left', ms=120, wait=False)
                flip = -flip
            except Exception as error:
                print('keep-awake stopped: %s' % error, flush=True)
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target')
    parser.add_argument('--keep-awake', type=float, default=0, help='nudge the right stick every N seconds after the map is up')
    parser.add_argument('--keep-awake-total', type=float, default=3600)
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], args.target] + rest
    nav_args = argparse.Namespace(target=args.target, out=None, boot_timeout=900, screen_timeout=900, race_timeout=180, frame_timeout=120,
                                  poll_seconds=1.0, quiet_seconds=4.0, change_threshold=8.0, stable_threshold=3.0, stable_checks=4,
                                  reference_threshold=12.0, prerace_settle=35, start_hold_ms=3000, start_presses=3, race_frames=3, no_reference=True)
    route = MapRoute(nav_args)
    try:
        ok = route.run()
        route.finish(ok, None)
    except Exception as error:
        route.finish(False, str(error))
        print('MAP FAILED: %s' % error, flush=True)
        return 1
    if args.keep_awake > 0:
        route.keep_awake(args.keep_awake, args.keep_awake_total)
    return 0


if __name__ == '__main__':
    sys.exit(main())
