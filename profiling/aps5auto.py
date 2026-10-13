"""aps5auto: drive an AnyPS5 game through its automation channel (stdlib only, run with python -I).

Enable it in the game: APS5_AUTOMATION=1 serves commands on \\\\.\\pipe\\aps5-<pid>; APS5_AUTOMATION=<name>
uses \\\\.\\pipe\\<name>. APS5_AUTOMATION_EVENTS=<file> names the JSON-lines event log (default
automation-events.jsonl in the game folder). With the run scripts:
    best-run.ps1 -Tag t1 -ExtraEnv 'APS5_AUTOMATION=1','APS5_AUTOMATION_EVENTS=D:\\ps5\\gt7\\diagnostics-rapid\\t1\\automation-events.jsonl'
Inputs go straight into the emulated pad (no window focus, merged with keyboard/mouse). A press is held
for `ms` of wall-clock time AND at least `polls` guest pad reads (and `frames` presents), so it is never
missed at 1 FPS; two presses of one button are always separated by a read that sees it released.

Python:
    import aps5auto
    g = aps5auto.connect('t1')            # run tag (diagnostics-rapid/<tag>/process.json), pid or pipe name
    g.press('cross', ms=300, polls=3)     # blocks until released; returns the ack (polls, frames, held_ms)
    g.press('l1+cross'); g.stick('left', 0.0, -1.0, ms=500); g.trigger('r2', 1.0, ms=2000)
    img = g.snapshot(r'D:\\ps5\\gt7\\shot.bmp') # the next presented frame, written before this returns
    g.wait_for('vrr', where=lambda e: e['op'] == 'peg', timeout=120)   # events since the last input
    g.fps(); g.screen()                   # presents/s; event-based "what screen" hints
CLI: python -I aps5auto.py <target> <command...>   (commands: ping status fps screen press hold release
    stick trigger wait snapshot mark events [--follow] wait-for <type[:substring]> | raw "<line>"),
    python -I aps5auto.py title-focus <image>, python -I aps5auto.py to-png <in.bmp> <out.png>
Events: {"seq","t","wall","type",...}; types: automation open module sysmodule dialog vrr savedata movie audio
fps input snapshot mark. Snapshot needs the window not minimized (PrintWindow, screen copy when visible; falls
back to the newest APS5_DUMP_FRAMES dump when the capture is black).
"""

import collections
import glob
import json
import os
import struct
import sys
import tempfile
import threading
import time
import zlib

GAME_ROOT = r'D:\ps5\gt7'
RUNS_ROOT = os.path.join(GAME_ROOT, 'diagnostics-rapid')
PIPE_PREFIX = '\\\\.\\pipe\\'


class AutomationError(RuntimeError):
    def __init__(self, message, reply=None):
        super().__init__(message)
        self.reply = reply or {}


def pipe_name(target):
    """Pipe path for a pipe name/path, a pid, or a run tag with process.json."""
    target = str(target)
    if target.startswith(PIPE_PREFIX):
        return target
    if target.isdigit():
        return PIPE_PREFIX + 'aps5-' + target
    identity = os.path.join(RUNS_ROOT, target, 'process.json')
    if os.path.exists(identity):
        with open(identity, encoding='utf-8-sig') as f:
            return PIPE_PREFIX + 'aps5-%d' % json.load(f)['pid']
    return PIPE_PREFIX + target


class PipeConnection:
    """One client connection: a command line out, one JSON reply line back."""

    def __init__(self, name, connect_timeout=10.0):
        deadline = time.monotonic() + connect_timeout
        while True:
            try:
                self.fd = os.open(name, os.O_RDWR | getattr(os, 'O_BINARY', 0))
                break
            except OSError as error:
                if time.monotonic() >= deadline:
                    raise AutomationError('cannot connect to %s: %s (is the game running with APS5_AUTOMATION?)' % (name, error))
                time.sleep(0.2)
        self.name = name
        self.buffer = b''
        self.lock = threading.Lock()

    def request(self, line):
        with self.lock:
            os.write(self.fd, (line.rstrip('\n') + '\n').encode('utf-8'))
            while b'\n' not in self.buffer:
                chunk = os.read(self.fd, 1 << 20)
                if not chunk:
                    raise AutomationError('the game closed the automation pipe')
                self.buffer += chunk
            reply, self.buffer = self.buffer.split(b'\n', 1)
        return json.loads(reply.decode('utf-8', errors='replace'))

    def close(self):
        try:
            os.close(self.fd)
        except OSError:
            pass


def quote(text):
    return '"' + str(text).replace('\\', '\\\\').replace('"', '\\"') + '"'


def _axis(value):
    """-1.0..1.0 (float) or 0..255 (int) -> 0..255."""
    if isinstance(value, float):
        return max(0, min(255, int(round((value + 1.0) * 127.5))))
    return max(0, min(255, int(value)))


def _matcher(match, where=None):
    if callable(match):
        base = match
    elif isinstance(match, dict):
        base = lambda e: all(e.get(k) == v for k, v in match.items())
    elif ':' in match:
        kind, part = match.split(':', 1)
        part = part.lower()
        base = lambda e: e.get('type') == kind and part in json.dumps(e).lower()
    else:
        base = lambda e: e.get('type') == match
    if where is None:
        return base
    return lambda e: base(e) and where(e)


class Game:
    def __init__(self, target, connect_timeout=10.0, frames_dir=GAME_ROOT):
        self.pipe = pipe_name(target)
        self.commands = PipeConnection(self.pipe, connect_timeout)
        self.event_link = None
        self.frames_dir = frames_dir
        self.events_cache = collections.deque(maxlen=50000)
        self.next_seq = 0
        self.checkpoint = 0
        self._snapshots = 0
        self._pump(0)
        self.checkpoint = self.next_seq

    # -- commands ---------------------------------------------------------------------------
    def cmd(self, line, check=True):
        reply = self.commands.request(line)
        if check and not reply.get('ok'):
            raise AutomationError('%s -> %s' % (line, reply.get('error', reply)), reply)
        return reply

    def ping(self):
        return self.cmd('ping')

    def status(self):
        return self.cmd('status')

    def fps(self):
        return self.status()['fps']

    def _mark_checkpoint(self):
        self._pump(0)
        self.checkpoint = self.next_seq

    def press(self, buttons, ms=150, polls=2, frames=0, timeout=30.0, wait=True):
        """Presses buttons ('cross', 'l1+cross' or a list) until ms elapsed, polls reads and frames presents saw it."""
        if not isinstance(buttons, str):
            buttons = '+'.join(buttons)
        self._mark_checkpoint()
        line = 'press %s %d polls=%d frames=%d timeout=%d%s' % (buttons, ms, polls, frames, int(timeout * 1000), '' if wait else ' nowait')
        return self.cmd(line)

    def hold(self, buttons, polls=0):
        if not isinstance(buttons, str):
            buttons = '+'.join(buttons)
        self._mark_checkpoint()
        return self.cmd('hold %s polls=%d' % (buttons, polls))

    def release(self, target='all'):
        return self.cmd('release %s' % target)

    def stick(self, side, x, y, ms=None, polls=1, frames=0, timeout=30.0):
        """x, y: floats -1..1 (up is y=-1) or ints 0..255; ms=None holds until changed (128,128 releases)."""
        self._mark_checkpoint()
        line = 'stick %s %d %d' % (side, _axis(x), _axis(y))
        if ms is not None:
            line += ' %d polls=%d frames=%d timeout=%d' % (ms, polls, frames, int(timeout * 1000))
        return self.cmd(line)

    def trigger(self, side, value, ms=None, polls=1, frames=0, timeout=30.0):
        """value: float 0..1 or int 0..255; ms=None holds (0 releases)."""
        level = int(round(value * 255)) if isinstance(value, float) else int(value)
        self._mark_checkpoint()
        line = 'trigger %s %d' % (side, max(0, min(255, level)))
        if ms is not None:
            line += ' %d polls=%d frames=%d timeout=%d' % (ms, polls, frames, int(timeout * 1000))
        return self.cmd(line)

    def wait_polls(self, count, timeout=30.0):
        return self.cmd('wait polls=%d timeout=%d' % (count, int(timeout * 1000)))

    def wait_presents(self, count, timeout=60.0):
        return self.cmd('wait presents=%d timeout=%d' % (count, int(timeout * 1000)))

    def mark(self, text):
        return self.cmd('mark %s' % text)

    def snapshot(self, path=None, after=1, timeout=60.0, dump_fallback=True):
        """Saves the after-th frame presented from now (.bmp or .png) and returns it as an Image.

        When the window capture comes back black or fails and APS5_DUMP_FRAMES dumps exist, the newest
        frame dump written after the request is used instead (img.info['method'] == 'frame-dump')."""
        if path is None:
            self._snapshots += 1
            path = os.path.join(tempfile.gettempdir(), 'aps5-snapshot-%d-%d.bmp' % (os.getpid(), self._snapshots))
        path = os.path.abspath(path)
        requested = time.time()
        reply = self.cmd('snapshot %s after=%d timeout=%d' % (quote(path), after, int(timeout * 1000)), check=False)
        if not reply.get('ok') and 'minimized' in str(reply.get('error', '')) and restore_window(self.status()['pid']):
            # A minimized window presents nothing; restore it without taking focus and ask again.
            time.sleep(1.0)
            reply = self.cmd('snapshot %s after=%d timeout=%d' % (quote(path), after, int(timeout * 1000)), check=False)
        if reply.get('ok') and reply.get('method') == 'screen' and reply.get('occluded'):
            # A screen copy of a covered window shows whatever is on top of it, not the game.
            reply = dict(reply, ok=False, error='the output window is covered (screen copy refused)')
        if reply.get('ok') and float(reply.get('mean', 1.0)) < 0.5:
            reply = dict(reply, black=True)  # an all-black capture (e.g. PrintWindow of a Vulkan window)
        if reply.get('ok') and not reply.get('black'):
            image = Image.load(path)
            image.info = reply
            return image
        # Only a run that is dumping frames (a dump in the last two minutes) is worth waiting for.
        if dump_fallback and newest_frame_dump(self.frames_dir, newer_than=requested - 120.0):
            # A covered window can wait for the next dump (one every APS5_DUMP_FRAMES_EVERY frames, which
            # at the race's frame rate can be tens of seconds away); a black capture needs only a moment.
            covered = 'covered' in str(reply.get('error', ''))
            dump = newest_frame_dump(self.frames_dir, newer_than=requested, wait=min(timeout, 45.0 if covered else 10.0))
            if dump:
                image = Image.load(dump)
                image.info = dict(reply, ok=True, method='frame-dump', path=dump)
                return image
        if reply.get('ok'):
            image = Image.load(path)
            image.info = reply
            return image
        raise AutomationError('snapshot failed: %s' % reply.get('error', reply), reply)

    # -- events -----------------------------------------------------------------------------
    def _events_request(self, line):
        if self.event_link is None:
            self.event_link = PipeConnection(self.pipe)
        return self.event_link.request(line)

    def _pump(self, timeout):
        """Fetches new events into the cache; waits up to timeout seconds for the first one."""
        reply = self._events_request('events since=%d timeout=%d max=20000' % (self.next_seq, int(timeout * 1000)))
        if not reply.get('ok'):
            raise AutomationError('events -> %s' % reply.get('error'), reply)
        for event in reply['events']:
            self.events_cache.append(event)
        self.next_seq = reply['next']
        return reply['events']

    def cursor(self):
        """The sequence number the next event will get (for wait_for(since=...))."""
        self._pump(0)
        return self.next_seq

    def events(self, since=None, timeout=0.0):
        """Cached and new events with seq >= since (default: since the last input command)."""
        since = self.checkpoint if since is None else since
        self._pump(timeout)
        return [e for e in self.events_cache if e['seq'] >= since]

    def wait_for(self, match, timeout=60.0, since=None, where=None):
        """First event with seq >= since (default: the last input command) matching `match`: an event type
        ('vrr'), 'type:substring' ('open:music'), a dict of field values, or a predicate on the event dict."""
        test = _matcher(match, where)
        since = self.checkpoint if since is None else since
        deadline = time.monotonic() + timeout
        scanned = since
        while True:
            for event in list(self.events_cache):
                if event['seq'] >= scanned and test(event):
                    return event
            scanned = max(scanned, self.next_seq)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('no %r event within %.0f s' % (match, timeout))
            self._pump(min(remaining, 1.0))

    def recent(self, seconds=10.0, kind=None):
        self._pump(0)
        if not self.events_cache:
            return []
        latest = self.events_cache[-1]['t']
        return [e for e in self.events_cache if e['t'] >= latest - seconds and (kind is None or e['type'] == kind)]

    def screen(self, window=10.0):
        """Event-based hints about the current screen: race (VRR pegged), dialog, movie, loading, menu."""
        self._pump(0)
        events = list(self.events_cache)
        latest = events[-1]['t'] if events else 0.0
        vrr = next((e for e in reversed(events) if e['type'] == 'vrr'), None)
        fps_event = next((e for e in reversed(events) if e['type'] == 'fps'), None)
        movie = next((e for e in reversed(events) if e['type'] == 'movie'), None)
        dialogs = {}
        for e in events:
            if e['type'] == 'dialog':
                dialogs[e.get('dialog')] = e
        open_dialogs = [k for k, e in dialogs.items() if e.get('op') == 'open']
        opens = sum(1 for e in events if e['type'] == 'open' and e['t'] >= latest - 5.0)
        in_race = vrr is not None and vrr.get('op') == 'peg'
        fps = fps_event['fps'] if fps_event else 0.0
        if in_race:
            guess = 'race'
        elif open_dialogs:
            guess = 'dialog:' + ','.join(open_dialogs)
        elif movie is not None and movie['t'] >= latest - window:
            guess = 'movie'
        elif opens > 20 or fps < 0.3:
            guess = 'loading'
        else:
            guess = 'menu'
        return {'guess': guess, 'in_race': in_race, 'fps': fps, 'opens_last_5s': opens,
                'open_dialogs': open_dialogs, 'last_movie': movie and movie.get('path'),
                'last_vrr': vrr and vrr.get('op'), 't': latest}

    def wait_settled(self, quiet=3.0, min_presents=2, timeout=120.0):
        """Waits until no guest file opened for `quiet` seconds and min_presents frames were presented."""
        deadline = time.monotonic() + timeout
        start_presents = self.status()['presents']
        last_open = time.monotonic()
        while time.monotonic() < deadline:
            new = self._pump(0.5)
            if any(e['type'] == 'open' for e in new):
                last_open = time.monotonic()
            if time.monotonic() - last_open >= quiet and self.status()['presents'] - start_presents >= min_presents:
                return True
        return False

    def close(self):
        self.commands.close()
        if self.event_link:
            self.event_link.close()


def restore_window(pid):
    """Restores the minimized top-level windows of process pid without activating them; True if any was."""
    if os.name != 'nt':
        return False
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    restored = []
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def visit(hwnd, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd) and user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 4)  # SW_SHOWNOACTIVATE
            restored.append(hwnd)
        return True

    user32.EnumWindows(proc(visit), 0)
    return bool(restored)


def connect(target, connect_timeout=10.0, frames_dir=GAME_ROOT):
    return Game(target, connect_timeout, frames_dir)


def newest_frame_dump(directory, newer_than=0.0, wait=0.0):
    """Newest complete frame_*.bmp in directory modified after newer_than (epoch s), waiting up to wait s."""
    deadline = time.monotonic() + wait
    while True:
        best = None
        for path in glob.glob(os.path.join(directory, 'frame_*.bmp')):
            try:
                st = os.stat(path)
            except OSError:
                continue
            if st.st_mtime > newer_than and (best is None or st.st_mtime > best[0]):
                best = (st.st_mtime, path, st.st_size)
        if best:
            try:
                with open(best[1], 'rb') as f:
                    head = f.read(54)
                if len(head) == 54 and head[:2] == b'BM' and struct.unpack_from('<I', head, 2)[0] <= best[2]:
                    return best[1]
            except OSError:
                pass
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.25)


class Image:
    """RGB pixels of a BMP (24/32-bit) or PNG (8-bit RGB/RGBA); coordinates are top-down."""

    def __init__(self, width, height, rgb, path=None):
        self.width, self.height, self.rgb, self.path = width, height, rgb, path
        self.info = {}

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = f.read()
        if data[:2] == b'BM':
            return cls._bmp(data, path)
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            return cls._png(data, path)
        raise ValueError('%s: not a BMP or PNG' % path)

    @classmethod
    def _bmp(cls, data, path):
        offset = struct.unpack_from('<I', data, 10)[0]
        width, height = struct.unpack_from('<ii', data, 18)
        bpp = struct.unpack_from('<H', data, 28)[0]
        if bpp not in (24, 32):
            raise ValueError('%s: %d-bit BMP not supported' % (path, bpp))
        top_down = height < 0
        height = abs(height)
        bytes_per = bpp // 8
        stride = (width * bytes_per + 3) & ~3
        rgb = bytearray(width * height * 3)
        for y in range(height):
            src_row = y if top_down else height - 1 - y
            row = data[offset + src_row * stride: offset + src_row * stride + width * bytes_per]
            out = y * width * 3
            rgb[out + 0: out + width * 3: 3] = row[2::bytes_per]
            rgb[out + 1: out + width * 3: 3] = row[1::bytes_per]
            rgb[out + 2: out + width * 3: 3] = row[0::bytes_per]
        return cls(width, height, bytes(rgb), path)

    @classmethod
    def _png(cls, data, path):
        pos, idat, width = 8, b'', 0
        while pos < len(data):
            length, kind = struct.unpack_from('>I4s', data, pos)
            body = data[pos + 8: pos + 8 + length]
            if kind == b'IHDR':
                width, height, depth, color = struct.unpack_from('>IIBB', body)
                if depth != 8 or color not in (2, 6):
                    raise ValueError('%s: only 8-bit RGB/RGBA PNG supported' % path)
                channels = 3 if color == 2 else 4
            elif kind == b'IDAT':
                idat += body
            pos += 12 + length
        raw = zlib.decompress(idat)
        stride = width * channels
        rgb = bytearray(width * height * 3)
        previous = bytearray(stride)
        for y in range(height):
            filt = raw[y * (stride + 1)]
            line = bytearray(raw[y * (stride + 1) + 1: (y + 1) * (stride + 1)])
            for i in range(stride if filt else 0):
                a = line[i - channels] if i >= channels else 0
                b = previous[i]
                c = previous[i - channels] if i >= channels else 0
                if filt == 1:
                    line[i] = (line[i] + a) & 255
                elif filt == 2:
                    line[i] = (line[i] + b) & 255
                elif filt == 3:
                    line[i] = (line[i] + (a + b) // 2) & 255
                elif filt == 4:
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
            out = y * width * 3
            for ch in range(3):
                rgb[out + ch: out + width * 3: 3] = line[ch::channels]
            previous = line
        return cls(width, height, bytes(rgb), path)

    def pixel(self, x, y):
        i = (y * self.width + x) * 3
        return self.rgb[i], self.rgb[i + 1], self.rgb[i + 2]

    def mean(self, x0, y0, x1, y1, step=2):
        """Mean of (R+G+B)/3 over the pixel box [x0,x1) x [y0,y1)."""
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(self.width, int(x1)), min(self.height, int(y1))
        total, count = 0, 0
        for y in range(y0, y1, step):
            row = self.rgb[(y * self.width + x0) * 3: (y * self.width + x1) * 3]
            sample = row[::3 * step], row[1::3 * step], row[2::3 * step]
            total += sum(sample[0]) + sum(sample[1]) + sum(sample[2])
            count += len(sample[0]) * 3
        return total / count if count else 0.0

    def box1080(self, cx, cy, half_w, half_h, step=2):
        """mean() of a box given in 1920x1080 coordinates (scaled to this image)."""
        sx, sy = self.width / 1920.0, self.height / 1080.0
        return self.mean((cx - half_w) * sx, (cy - half_h) * sy, (cx + half_w) * sx, (cy + half_h) * sy, step)

    def luma(self):
        return self.mean(0, 0, self.width, self.height, step=max(1, self.width // 160))

    def diff(self, other, grid=48, region=(0.0, 0.0, 1.0, 1.0)):
        """Mean absolute brightness difference over a grid x grid sample of region (x0, y0, x1, y1 as
        fractions of the frame; 0 = identical). Works across resolutions."""
        x0, y0, x1, y1 = region
        total = 0.0
        for gy in range(grid):
            for gx in range(grid):
                fx = x0 + (gx + 0.5) * (x1 - x0) / grid
                fy = y0 + (gy + 0.5) * (y1 - y0) / grid
                a = self.pixel(min(self.width - 1, int(fx * self.width)), min(self.height - 1, int(fy * self.height)))
                b = other.pixel(min(other.width - 1, int(fx * other.width)), min(other.height - 1, int(fy * other.height)))
                total += abs(sum(a) - sum(b)) / 3.0
        return total / (grid * grid)

    def save_png(self, path):
        raw = b''.join(b'\x00' + self.rgb[y * self.width * 3: (y + 1) * self.width * 3] for y in range(self.height))
        chunk = lambda kind, body: struct.pack('>I', len(body)) + kind + body + struct.pack('>I', zlib.crc32(kind + body) & 0xffffffff)
        with open(path, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', self.width, self.height, 8, 2, 0, 0, 0))
                    + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def title_focus(image):
    """Which PPSA01317 title-screen button is lit: ('WM'|'MR'|'NONE'|'NOTTITLE', wm, mr) (titlefocus.py logic)."""
    wm, mr = image.box1080(760, 586, 150, 22), image.box1080(1160, 586, 150, 22)
    title = image.box1080(960, 330, 150, 22) > 60
    if not title:
        state = 'NOTTITLE'
    elif mr > wm + 12:
        state = 'MR'
    elif wm > mr + 12:
        state = 'WM'
    else:
        state = 'NONE'
    return state, round(wm, 1), round(mr, 1)


def _print(obj):
    print(json.dumps(obj, indent=None if isinstance(obj, dict) and len(json.dumps(obj)) < 200 else 1))


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    if argv[0] == 'title-focus':
        print(*title_focus(Image.load(argv[1])))
        return 0
    if argv[0] == 'to-png':
        Image.load(argv[1]).save_png(argv[2])
        return 0
    game = connect(argv[0])
    verb, rest = argv[1].lower(), argv[2:]
    if verb == 'fps':
        print(game.fps())
    elif verb == 'screen':
        _print(game.screen())
    elif verb == 'snapshot':
        image = game.snapshot(rest[0] if rest else None)
        _print(dict(image.info, luma=round(image.luma(), 1), title=title_focus(image)[0]))
    elif verb == 'events':
        follow = '--follow' in rest
        kind = rest[rest.index('--type') + 1] if '--type' in rest else None
        since = int(rest[rest.index('--since') + 1]) if '--since' in rest else 0
        game.next_seq = since
        while True:
            for event in game._pump(1.0 if follow else 0):
                if kind is None or event['type'] == kind:
                    print(json.dumps(event), flush=True)
            if not follow:
                break
    elif verb == 'wait-for':
        timeout = float(rest[rest.index('--timeout') + 1]) if '--timeout' in rest else 60.0
        _print(game.wait_for(rest[0], timeout=timeout, since=0 if '--any' in rest else None))
    elif verb == 'raw':
        _print(game.cmd(' '.join(rest), check=False))
    else:
        _print(game.cmd(' '.join([verb] + [quote(a) if ' ' in a else a for a in rest]), check=False))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
