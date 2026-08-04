"""
Simulate OSC triggers so you can exercise the bridge without Resolume open.

Examples:
    python test_trigger.py --clip 1 1        # simulate Resolume layer 1 / clip 1 connect
    python test_trigger.py --prompt "a cat made of stained glass"
"""

import argparse

from pythonosc.udp_client import SimpleUDPClient

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=9000)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--clip", nargs=2, type=int, metavar=("LAYER", "CLIP"),
                    help="simulate Resolume's own clip-connect OSC address")
group.add_argument("--prompt", type=str, help="send freeform prompt text directly")
args = parser.parse_args()

client = SimpleUDPClient(args.host, args.port)

if args.clip:
    layer, clip = args.clip
    address = f"/composition/layers/{layer}/clips/{clip}/connect"
    client.send_message(address, 1.0)
    print(f"sent {address} 1.0")
else:
    client.send_message("/comfybridge/generate", args.prompt)
    print(f"sent /comfybridge/generate {args.prompt!r}")
