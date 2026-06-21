"""Modal app for GR00T N1.5 + AEGIS. Image mirrors the known-good local gr00t env
(torch 2.5.1+cu124, flash-attn 2.7.1.post4, transformers 4.51.3, Isaac-GR00T@4ea96a1).

Cost discipline:
  modal run gr00t_modal.py::main --stage validate   # CHEAP CPU: confirm image builds/imports
  modal run gr00t_modal.py::main --stage smoke       # cheap L4 GPU: real-model wiring smoke
RAM-OOM-proof: every function declares `memory=` so the container can't exceed it.
"""
import sys
import modal

FLASH_WHL = ("https://github.com/Dao-AILab/flash-attention/releases/download/"
             "v2.7.1.post4/flash_attn-2.7.1.post4+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install("torch==2.5.1", "torchvision==0.20.1",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(FLASH_WHL)
    .pip_install_from_requirements("gr00t_requirements_smoke.txt", extra_options="--no-deps")
    .run_commands("pip install --no-deps 'git+https://github.com/NVIDIA/Isaac-GR00T.git@4ea96a1'")
    .add_local_file("gr00t_aegis.py", "/root/gr00t_aegis.py")
)

app = modal.App("gr00t-aegis")
hf_cache = modal.Volume.from_name("gr00t-hf-cache", create_if_missing=True)


@app.function(image=image, cpu=2.0, memory=8192, timeout=900)
def validate_image():
    """CPU-only: confirm the image deps import + versions match local. Pennies."""
    out = {}
    import torch; out["torch"] = torch.__version__
    try:
        import flash_attn; out["flash_attn"] = flash_attn.__version__
    except Exception as e:
        out["flash_attn"] = f"(import deferred to GPU: {type(e).__name__})"
    import transformers, diffusers, peft, accelerate
    out["transformers"] = transformers.__version__
    out["diffusers"] = diffusers.__version__
    import gr00t                                              # import only (no model build)
    from gr00t.model.gr00t_n1 import GR00T_N1_5               # class import is CPU-safe
    out["gr00t_import"] = "ok"
    sys.path.insert(0, "/root")
    import gr00t_aegis
    from gr00t_aegis import RobustIB, AdaptiveSpectralFilter
    # tiny CPU module identity re-check (parity with local tier-1)
    import torch.nn as nn
    lin = nn.Linear(2048, 1536)
    from gr00t_aegis import FusedRobustIBProjector
    f = FusedRobustIBProjector(lin, D_out=1536).eval()
    x = torch.randn(1, 3, 2048)
    out["rib_identity_cpu"] = bool(torch.allclose(f(x), lin(x), atol=1e-5))
    out["rib_params_M"] = round(f.rib.n_params() / 1e6, 2)
    return out


@app.function(image=image, gpu="L4", memory=32768, timeout=1800,
              volumes={"/cache": hf_cache})
def wiring_smoke():
    """L4 GPU: load REAL GR00T N1.5, inject RIB, verify identity-at-init + RASF dims."""
    import os, torch, torch.nn as nn
    os.environ["HF_HOME"] = "/cache/hf"
    sys.path.insert(0, "/root")
    from gr00t_aegis import inject_rib, FusedRobustIBProjector, AdaptiveSpectralFilter
    from gr00t.model.gr00t_n1 import GR00T_N1_5

    res = {}
    model = GR00T_N1_5.from_pretrained("nvidia/GR00T-N1.5-3B",
                                       torch_dtype=torch.float32).eval().cuda()
    # RIB locus is the real vision->LLM connector mlp1 (eagle_linear is nn.Identity).
    conn = model.backbone.eagle_model.mlp1
    lins = [m for m in conn.modules() if isinstance(m, nn.Linear)]
    d_in = lins[0].in_features
    cdtype = next(conn.parameters()).dtype          # weights may be bf16
    res["connector"] = f"mlp1 {type(conn).__name__} {d_in}->{lins[-1].out_features} ({cdtype})"
    x = torch.randn(1, 4, d_in).cuda().to(cdtype)   # match connector dtype
    with torch.no_grad():
        base = conn(x)
    fused = inject_rib(model)
    res["rib_swapped"] = isinstance(model.backbone.eagle_model.mlp1, FusedRobustIBProjector)
    with torch.no_grad():
        out = model.backbone.eagle_model.mlp1(x)
    res["rib_identity_at_init"] = bool(out.shape == base.shape and torch.allclose(out, base, atol=1e-3))
    res["rib_params_M"] = round(fused.rib.n_params() / 1e6, 2)
    res["fusion_tanh"] = round(fused.ib_contribution, 3)

    H = int(model.action_horizon); d = int(model.action_dim)
    res["action_HxD"] = f"{H}x{d}"
    rasf = AdaptiveSpectralFilter(H=H, d=d, gate_max=0.6).cuda().eval()
    A = torch.randn(2, H, d).cuda()
    with torch.no_grad():
        Ah = rasf(A)["A_hat"]
    res["rasf_identity_at_init"] = bool(torch.allclose(Ah, A, atol=1e-5))
    res["rasf_gate_strength"] = rasf.gate_strength
    hf_cache.commit()
    return res


@app.function(image=image, gpu="L4", memory=32768, timeout=1800,
              volumes={"/cache": hf_cache})
def dump_structure():
    """L4 GPU: load REAL GR00T N1.5 and dump the backbone/action-head module tree so we
    can locate the TRUE 2048->1536 projection (eagle_linear turned out to be nn.Identity)."""
    import os, torch, torch.nn as nn
    os.environ["HF_HOME"] = "/cache/hf"
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    model = GR00T_N1_5.from_pretrained("nvidia/GR00T-N1.5-3B",
                                       torch_dtype=torch.float32).eval().cuda()
    res = {"action_horizon": int(getattr(model, "action_horizon", -1)),
           "action_dim": int(getattr(model, "action_dim", -1))}
    # every Linear whose (in,out) could be the connector, plus all backbone submodule names
    linears = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            linears.append((name, m.in_features, m.out_features))
    # candidate projections feeding the action head (out==1536 or in==2048)
    res["candidate_projections"] = [
        f"{n}: {i}->{o}" for (n, i, o) in linears if o == 1536 or i == 2048 or i == 1536]
    # top-level structure of backbone + action_head (1-2 levels deep)
    def shallow(mod, prefix, depth=2):
        out = []
        for n, c in mod.named_children():
            out.append(f"{prefix}{n}: {type(c).__name__}")
            if depth > 1:
                for n2, c2 in c.named_children():
                    out.append(f"{prefix}  {n}.{n2}: {type(c2).__name__}")
        return out
    res["backbone_tree"] = shallow(model.backbone, "")
    res["eagle_linear_type"] = type(model.backbone.eagle_linear).__name__
    res["n_linears_total"] = len(linears)
    hf_cache.commit()
    return res


@app.local_entrypoint()
def main(stage: str = "validate"):
    if stage == "validate":
        print("VALIDATE:", validate_image.remote())
    elif stage == "smoke":
        print("WIRING SMOKE:", wiring_smoke.remote())
    elif stage == "dump":
        import json as _j
        print("STRUCTURE:", _j.dumps(dump_structure.remote(), indent=2))
    else:
        print("unknown stage; use validate|smoke|dump")
