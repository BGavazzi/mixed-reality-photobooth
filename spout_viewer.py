"""
Minimal Spout receiver window, so you can see what the bridge is sending
without Resolume open. Displays whatever sender name you point it at
(default: the bridge's own "ComfyBridge").

    python spout_viewer.py
    python spout_viewer.py --sender ComfyBridge

Press q, Esc, or close the window to quit.
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

        window_created = False
        if buffer and result and not SpoutGL.helpers.isBufferEmpty(buffer):
            width = receiver.getSenderWidth()
            height = receiver.getSenderHeight()
            frame = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            cv2.imshow(window_name, bgr)
            window_created = True

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or Esc
            break

        # cv2 doesn't wire the window's own close (X) button to anything by
        # default — clicking it just hides the window and the loop above
        # would silently recreate it on the next frame. Checking visibility
        # here is what actually makes the X button work.
        if window_created and cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        receiver.waitFrameSync(args.sender, 10)

    cv2.destroyAllWindows()
