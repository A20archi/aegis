"""Merge a saved LoRA adapter into the base -> from_pretrained-compatible full ckpt for eval."""
import sys, argparse
sys.path.insert(0, ".")
import torch
from peft import PeftModel
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
ap=argparse.ArgumentParser(); ap.add_argument("--base",required=True); ap.add_argument("--adapter",required=True); ap.add_argument("--out",required=True)
a=ap.parse_args()
p=SmolVLAPolicy.from_pretrained(a.base)
p=PeftModel.from_pretrained(p, a.adapter)
m=p.merge_and_unload()
m.save_pretrained(a.out)
print("merged ->", a.out)
