"""nav-map-blind.py TAG: reaches the world map without snapshots (screen locked or window covered).
Waits for the automation pipe, then for the title screen by frame rate (presenting steadily for 20 s
after at least MIN_BOOT seconds), presses Cross (WORLD MAP is highlighted at boot), dismisses a
screensaver with Circle if the frame rate jumps to ~60, and prints MAP READY once the frame rate stays
under 30 with navigation presses for 20 s. Uses snapshots for the title when they work."""
import sys
import time

sys.path.insert(0, __import__('os').path.dirname(__file__))
import aps5auto

tag = sys.argv[1]
MIN_BOOT = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
deadline = time.monotonic() + 1200
game = None
while game is None:
    try:
        game = aps5auto.Game(tag)
    except Exception as error:
        if time.monotonic() > deadline:
            print('MAP FAILED: no pipe (%s)' % error, flush=True)
            sys.exit(1)
        time.sleep(3)
start = time.monotonic()


def fps():
    return game.screen()['fps']


def title_by_snapshot():
    try:
        image = game.snapshot(r'D:\ps5\work\blind-title.bmp', timeout=15)
    except Exception:
        return None
    return aps5auto.title_focus(image)[0]


# Title screen.
steady = 0
while True:
    if time.monotonic() > deadline:
        print('MAP FAILED: no title', flush=True)
        sys.exit(1)
    time.sleep(2)
    state = title_by_snapshot()
    if state in ('WM', 'MR'):
        print('title (snapshot) %s' % state, flush=True)
        if state == 'MR':
            game.press('left', ms=150, wait=False)
            time.sleep(1)
        break
    rate = fps()
    steady = steady + 1 if 5 < rate < 40 and game.screen()['opens_last_5s'] == 0 else 0
    if state is None and time.monotonic() - start > MIN_BOOT and steady >= 10:
        print('title (frame rate %.1f, %d s after the pipe)' % (rate, time.monotonic() - start), flush=True)
        break

game.press('cross', ms=200, wait=False)
low = 0
flip = True
last_press = time.monotonic()
while low < 10:
    if time.monotonic() > deadline:
        print('MAP FAILED: map scene never steady', flush=True)
        sys.exit(1)
    time.sleep(2)
    rate = fps()
    if rate > 50:
        game.press('circle', ms=150, wait=False)
        low = 0
        continue
    low = low + 1 if 3 < rate < 30 else 0
    if time.monotonic() - last_press > 15:
        game.press('right' if flip else 'left', ms=120, wait=False)
        flip = not flip
        last_press = time.monotonic()
print('MAP READY (%.1f FPS)' % fps(), flush=True)
