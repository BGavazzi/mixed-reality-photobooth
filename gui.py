"""
Minimal typing box for the bridge. No OSC/curl knowledge needed — type a
prompt, hit a button (or Enter), it fires the same triggers
test_trigger.py does.

    python gui.py
"""

import argparse
import tkinter as tk
from tkinter import ttk

from pythonosc.udp_client import SimpleUDPClient

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=9000)
args = parser.parse_args()

client = SimpleUDPClient(args.host, args.port)


def send(address: str, value):
    client.send_message(address, value)
    status_var.set(f"sent {address} {value!r}")


def generate_image(event=None):
    text = prompt_var.get().strip()
    if text:
        send("/comfybridge/generate", text)


def generate_video():
    text = prompt_var.get().strip()
    if text:
        send("/comfybridge/generate_video", text)


def resync():
    send("/comfybridge/resync", 1)


root = tk.Tk()
root.title("ComfyBridge")
root.geometry("520x150")
root.resizable(False, False)

frame = ttk.Frame(root, padding=16)
frame.pack(fill="both", expand=True)

prompt_var = tk.StringVar()
entry = ttk.Entry(frame, textvariable=prompt_var, font=("Segoe UI", 12))
entry.pack(fill="x", pady=(0, 10))
entry.bind("<Return>", generate_image)
entry.focus()

button_row = ttk.Frame(frame)
button_row.pack(fill="x")
ttk.Button(button_row, text="Generate Image", command=generate_image).pack(side="left", padx=(0, 8))
ttk.Button(button_row, text="Generate Video", command=generate_video).pack(side="left", padx=(0, 8))
ttk.Button(button_row, text="Resync from Resolume", command=resync).pack(side="left")

status_var = tk.StringVar(value=f"ready — sending to {args.host}:{args.port}")
ttk.Label(frame, textvariable=status_var, foreground="#666").pack(fill="x", pady=(12, 0))

root.mainloop()
