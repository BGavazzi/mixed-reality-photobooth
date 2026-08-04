"""
Minimal Spout receiver window, so you can see what the bridge is sending
without Resolume open. Displays whatever sender name you point it at
(default: the bridge's own "ComfyBridge").

    python spout_viewer.py
    python spout_viewer.py --sender ComfyBridge

Press q or Esc to close.
"""

import argparse
from array import array
from itertools import repeat

import cv2
import numpy as np
import SpoutGL
from OpenGL import GL

parser = argparse.ArgumentParser()
parser.add_argument("--sender", default="ComfyBridge")
args = parser.parse_args()

with SpoutGL.SpoutReceiver() as receiver:
    receiver.setReceiverName(args.sender)
    print(f"[viewer] waiting for Spout sender '{args.sender}'... (q/Esc to quit)")

    buffer = None
    window_name = f"Spout: {args.sender}"

    while True:
        result = receiver.receiveImage(buffer, GL.GL_RGBA, False, 0)

        if receiver.isUpdated():
            width = receiver.getSenderWidth()
            height = receiver.getSenderHeight()
            buffer = array("B", repeat(0, width * height * 4))
            print(f"[viewer] connected: {width}x{height}")

        if buffer and result and not SpoutGL.helpers.isBufferEmpty(buffer):
            width = receiver.getSenderWidth()
            height = receiver.getSenderHeight()
            frame = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            cv2.imshow(window_name, bgr)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or Esc
            break

        receiver.waitFrameSync(args.sender, 10)

    cv2.destroyAllWindows()
