"""keep-awake.py TAG [EVERY] [TOTAL]: presses Right then Left alternately every EVERY seconds (default 15)
for TOTAL seconds (default 7200), so the title's screensaver never starts during map measurements."""
import sys
import time

sys.path.insert(0, __import__('os').path.dirname(__file__))
import aps5auto

tag = sys.argv[1]
every = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
total = float(sys.argv[3]) if len(sys.argv) > 3 else 7200.0
game = aps5auto.Game(tag)
end = time.monotonic() + total
flip = True
while time.monotonic() < end:
    try:
        game.press('right' if flip else 'left', ms=120, wait=False)
    except Exception as error:
        print('keep-awake stopped: %s' % error, flush=True)
        break
    flip = not flip
    time.sleep(every)
