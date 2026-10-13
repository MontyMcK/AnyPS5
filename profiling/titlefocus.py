# Prints which title-screen button is highlighted in the newest frame dump: WM, MR, or NONE.
import glob, os, sys
from PIL import Image, ImageStat
fs = sorted(glob.glob('D:/ps5/gt7/frame_*.bmp'), key=os.path.getmtime)
im = Image.open(fs[-1]).convert('RGB'); w, h = im.size
def box(cx, cy):
    x0, x1 = int((cx - 150) * w / 1920), int((cx + 150) * w / 1920)
    y0, y1 = int((cy - 22) * h / 1080), int((cy + 22) * h / 1080)
    return sum(ImageStat.Stat(im.crop((x0, y0, x1, y1))).mean) / 3
wm, mr = box(760, 586), box(1160, 586)
title = box(960, 330) > 60  # the white logo area
state = 'NONE'
if title and mr > wm + 12: state = 'MR'
elif title and wm > mr + 12: state = 'WM'
elif not title: state = 'NOTTITLE'
print(state, round(wm, 1), round(mr, 1))
