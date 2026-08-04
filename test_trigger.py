"""
Simulate OSC triggers so you can exercise the bridge without Resolume open.

Examples:
    python test_trigger.py --clip 1 1        # simulate Resolume layer 1 / clip 1 connect
    python test_trigger.py --prompt "a cat made of stained glass"
    python test_trigger.py --video "a cat made of stained glass, slow pan"  # ComfyUI backend only
    python test_trigger.py --play-file "D:\\path\\to\\clip.mp4"            # play a pre-rendered file as-is
    python test_trigger.py --resync           # pull live state from Resolume's REST API
"""

import argparse

from pythonosc.udp_client import SimpleUDPClient

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=9000)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--clip", nargs=2, type=int, metavar=("LAYER", "CLIP"),
                    help="simulate Resolume's own clip-connect OSC address")
group.add_argument("--prompt", type=str, help="send freeform prompt text directly (still image)")
group.add_argument("--video", type=str, help="send freeform prompt text for video generation (ComfyUI backend only)")
group.add_argument("--play-file", type=str, help="play a local pre-rendered video file as-is, no generation")
group.add_argument("--resync", action="store_true",
                    help="trigger a regeneration from Resolume's current live state")
args = parser.parse_args()

client = SimpleUDPClient(args.host, args.port)

if args.clip:
    layer, clip = args.clip
    address = f"/composition/layers/{layer}/clips/{clip}/connect"
    client.send_message(address, 1.0)
    print(f"sent {address} 1.0")
elif args.resync:
    client.send_message("/comfybridge/resync", 1)
    print("sent /comfybridge/resync")
elif args.video:
    client.send_message("/comfybridge/generate_video", args.video)
    print(f"sent /comfybridge/generate_video {args.video!r}")
elif args.play_file:
    client.send_message("/comfybridge/play_file", args.play_file)
    print(f"sent /comfybridge/play_file {args.play_file!r}")
else:
    client.send_message("/comfybridge/generate", args.prompt)
    print(f"sent /comfybridge/generate {args.prompt!r}")
