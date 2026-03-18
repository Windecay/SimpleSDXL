# imghdr.py - Determine the type of an image
# Copied from Python standard library

import io
import os

def what(file, h=None):
    if h is None:
        if isinstance(file, str):
            f = None
            try:
                f = open(file, 'rb')
                h = f.read(32)
            finally:
                if f:
                    f.close()
        elif hasattr(file, 'read'):
            pos = file.tell()
            h = file.read(32)
            file.seek(pos)
        else:
            return None
    for name, test in tests:
        res = test(h)
        if res:
            return name
    return None

def test_jpeg(h):
    if h[0:3] == b'\xff\xd8\xff':
        return 'jpeg'
def test_png(h):
    if h[:8] == b'\211PNG\r\n\032\n':
        return 'png'
def test_gif(h):
    if h[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'
def test_tiff(h):
    if h[:2] in (b'MM', b'II'):
        return 'tiff'
def test_bmp(h):
    if h[:2] == b'BM':
        return 'bmp'
def test_webp(h):
    if h[0:4] == b'RIFF' and h[8:12] == b'WEBP':
        return 'webp'

tests = [
    ('jpeg', test_jpeg),
    ('png', test_png),
    ('gif', test_gif),
    ('tiff', test_tiff),
    ('bmp', test_bmp),
    ('webp', test_webp),
]
