"""Tests for aps5auto.py: image decoding, matchers, title focus, and the client against the real channel
served by build/tests/automation_channel_tests.exe (serve mode, a fake game). Run: python -I test_aps5auto.py"""

import os
import struct
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import aps5auto  # noqa: E402

REPO = os.path.dirname(HERE)
SERVER = os.path.join(REPO, 'build', 'tests', 'automation_channel_tests.exe')
DLL_DIRS = [os.path.join(REPO, 'build', 'core', 'libs', 'libs', 'unpatched'), r'C:\tools\mingw64\bin']


def write_bmp(path, width, height, pixel, bpp=24, top_down=False):
    bytes_per = bpp // 8
    stride = (width * bytes_per + 3) & ~3
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            r, g, b = pixel(x, y)
            row += bytes([b, g, r] + ([255] if bpp == 32 else []))
        rows.append(bytes(row) + b'\0' * (stride - len(row)))
    if not top_down:
        rows.reverse()
    data = b''.join(rows)
    header = b'BM' + struct.pack('<IHHI', 54 + len(data), 0, 0, 54)
    header += struct.pack('<IiiHHIIiiII', 40, width, -height if top_down else height, 1, bpp, 0, len(data), 2835, 2835, 0, 0)
    with open(path, 'wb') as f:
        f.write(header + data)


class ImageTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='aps5auto-test-')

    def test_bmp_layouts(self):
        pixel = lambda x, y: (x * 10, y * 20, 7)
        for bpp in (24, 32):
            for top_down in (False, True):
                path = os.path.join(self.dir, 'p%d%d.bmp' % (bpp, top_down))
                write_bmp(path, 5, 3, pixel, bpp, top_down)
                image = aps5auto.Image.load(path)
                self.assertEqual((image.width, image.height), (5, 3))
                self.assertEqual(image.pixel(4, 2), (40, 40, 7))
                self.assertEqual(image.pixel(1, 0), (10, 0, 7))

    def test_png_roundtrip_and_mean(self):
        image = aps5auto.Image(4, 2, bytes([10, 20, 30] * 4 + [90, 90, 90] * 4))
        path = os.path.join(self.dir, 'r.png')
        image.save_png(path)
        back = aps5auto.Image.load(path)
        self.assertEqual(back.rgb, image.rgb)
        self.assertAlmostEqual(back.mean(0, 0, 4, 1, step=1), 20.0)
        self.assertAlmostEqual(back.mean(0, 1, 4, 2, step=1), 90.0)
        self.assertEqual(back.diff(image), 0.0)
        self.assertGreater(back.diff(aps5auto.Image(4, 2, bytes(24))), 0.0)

    def test_title_focus(self):
        def title(lit):
            def pixel(x, y):
                X, Y = x * 1920 // 192, y * 1080 // 108
                if abs(X - 960) < 150 and abs(Y - 330) < 22:
                    return (220, 220, 220)
                if lit and abs(X - lit) < 150 and abs(Y - 586) < 22:
                    return (200, 200, 200)
                return (40, 40, 40)
            path = os.path.join(self.dir, 't%d.bmp' % lit)
            write_bmp(path, 192, 108, pixel)
            return aps5auto.Image.load(path)
        self.assertEqual(aps5auto.title_focus(title(1160))[0], 'MR')
        self.assertEqual(aps5auto.title_focus(title(760))[0], 'WM')
        self.assertEqual(aps5auto.title_focus(title(0))[0], 'NONE')
        dark = os.path.join(self.dir, 'dark.bmp')
        write_bmp(dark, 192, 108, lambda x, y: (0, 0, 0))
        self.assertEqual(aps5auto.title_focus(aps5auto.Image.load(dark))[0], 'NOTTITLE')


class HelperTests(unittest.TestCase):
    def test_matchers(self):
        event = {'type': 'open', 'path': '/app0/Music/Track01.dat', 'fd': 3}
        self.assertTrue(aps5auto._matcher('open')(event))
        self.assertFalse(aps5auto._matcher('vrr')(event))
        self.assertTrue(aps5auto._matcher('open:track01')(event))
        self.assertFalse(aps5auto._matcher('open:track02')(event))
        self.assertTrue(aps5auto._matcher({'type': 'open', 'fd': 3})(event))
        self.assertTrue(aps5auto._matcher('open', where=lambda e: e['fd'] > 2)(event))
        self.assertFalse(aps5auto._matcher(lambda e: False)(event))

    def test_pipe_names_and_axes(self):
        self.assertEqual(aps5auto.pipe_name(1234), '\\\\.\\pipe\\aps5-1234')
        self.assertEqual(aps5auto.pipe_name('\\\\.\\pipe\\custom'), '\\\\.\\pipe\\custom')
        self.assertEqual(aps5auto.pipe_name('no-such-tag-xyz'), '\\\\.\\pipe\\no-such-tag-xyz')
        self.assertEqual(aps5auto._axis(-1.0), 0)
        self.assertEqual(aps5auto._axis(1.0), 255)
        self.assertEqual(aps5auto._axis(0.0), 128)
        self.assertEqual(aps5auto._axis(300), 255)
        self.assertEqual(aps5auto.quote('a "b" c\\d'), '"a \\"b\\" c\\\\d"')


@unittest.skipUnless(os.path.exists(SERVER), 'build automation_channel_tests first')
class ChannelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.name = 'aps5auto-test-%d' % os.getpid()
        env = dict(os.environ, PATH=os.pathsep.join(DLL_DIRS + [os.environ.get('PATH', '')]))
        env.pop('APS5_AUTOMATION_EVENTS', None)
        cls.dir = tempfile.mkdtemp(prefix='aps5auto-chan-')
        env['APS5_AUTOMATION_EVENTS'] = os.path.join(cls.dir, 'events.jsonl')
        cls.server = subprocess.Popen([SERVER, 'serve', cls.name, '30'], env=env, stdout=subprocess.PIPE)
        cls.server.stdout.readline()
        cls.game = aps5auto.connect(aps5auto.PIPE_PREFIX + cls.name, frames_dir=cls.dir)

    @classmethod
    def tearDownClass(cls):
        cls.game.close()
        cls.server.kill()
        cls.server.wait()

    def test_press_then_wait_for_event(self):
        ack = self.game.press('cross', ms=200, polls=3)
        self.assertEqual(ack['state'], 'done')
        self.assertGreaterEqual(ack['polls'], 3)
        self.assertGreaterEqual(ack['held_ms'], 200)
        event = self.game.wait_for('vrr', where=lambda e: e['op'] == 'peg', timeout=5)
        self.assertGreaterEqual(event['seq'], self.game.checkpoint)
        self.assertTrue(self.game.screen()['in_race'])
        with self.assertRaises(TimeoutError):
            self.game.wait_for('vrr', since=self.game.cursor(), timeout=0.5)

    def test_snapshot(self):
        image = self.game.snapshot(os.path.join(self.dir, 'shot.bmp'), after=2)
        self.assertEqual((image.width, image.height), (64, 36))
        self.assertEqual(image.info['method'], 'fake')
        self.assertEqual(image.pixel(10, 5), (40, 40, 40))
        png = self.game.snapshot(os.path.join(self.dir, 'shot.png'))
        self.assertEqual(png.path, os.path.join(self.dir, 'shot.png'))

    def test_inputs_and_status(self):
        self.assertEqual(self.game.stick('left', 0.0, -1.0, ms=100)['state'], 'done')
        self.assertEqual(self.game.trigger('r2', 1.0, ms=50)['value'], 255)
        held = self.game.hold('l1', polls=1)
        self.assertEqual(held['state'], 'active')
        self.assertEqual(self.game.release('l1')['released'], 1)
        with self.assertRaises(aps5auto.AutomationError):
            self.game.press('bogus')
        status = self.game.status()
        self.assertGreater(status['presents'], 0)
        self.assertGreater(status['polls'], 0)
        self.game.wait_presents(3)
        time.sleep(1.2)
        self.assertGreater(self.game.fps(), 5)
        self.game.mark('hello there')
        self.assertTrue(any(e['type'] == 'mark' and e['text'] == 'hello there' for e in self.game.events(since=0)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
