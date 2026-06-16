"""Add the missing top-level "type": "smolvla" discriminator to lerobot-train
checkpoint config.json files so SmolVLAPolicy.from_pretrained / draccus can load them."""
import json, glob, sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
for f in glob.glob(f"{root}/checkpoints/*/pretrained_model/config.json"):
    c = json.load(open(f))
    if c.get("type") != "smolvla":
        json.dump({"type": "smolvla", **c}, open(f, "w"), indent=2)
        print("patched", f)
    else:
        print("ok", f)
