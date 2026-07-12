"""[nested_ssm] R rotation + v4 tiered decode for vLLM MambaMixer2.

Master copy lives in the repo (scale/vllm_v4_patch.py); install copies it to
  ~/vllm_env/.../vllm/model_executor/layers/mamba/v4_nested.py
and applies 5 small marked edits (see scale/install_vllm_v4.sh):
  mamba_mixer2.py  __init__ : v4R/v4cfg/_v4buf attrs
  mamba_mixer2.py  prefill  : rotate B_p/C_p after split; snapshot after state write
  mamba_mixer2.py  decode   : rotate B_d/C_d after split; tiered readout after SSU
  nemotron_h.py    load     : apply_ckpt(self) at end of ForCausalLM.load_weights

Config via env (so `vllm serve` CLI stays stock):
  NESTED_SSM_CKPT=/path/to/nemo9b_rot_*.pt   (enables R + tuned decay = fresh arm)
  NESTED_SSM_MODE=fresh|v4                   (default fresh)
  NESTED_SSM_PB=32  NESTED_SSM_C=4           (v4 tiering knobs)
  NESTED_SSM_COLD=fp32|bf16|fp8              (cold snapshot dtype, default bf16)

Semantics mirror scale/v4_native_decode.py (gate: bit-exact vs HF cuda path):
prefill fresh + snapshot; decode y = hot(fresh) + cold(stale snapshot, decay-
compensated), snapshot refresh every c tokens. State itself stays fully fresh
(SSU in-place) — this port is ACCURACY-grade; the speed path lands separately.
All bookkeeping is vectorized per cache slot -> CUDA-graph safe.
"""
import os
import torch
import torch.nn.functional as F


def _cfg_from_env():
    ck = os.environ.get("NESTED_SSM_CKPT")
    if not ck:
        return None
    return dict(
        ckpt=ck,
        mode=os.environ.get("NESTED_SSM_MODE", "fresh"),
        pb=int(os.environ.get("NESTED_SSM_PB", "32")),
        c=int(os.environ.get("NESTED_SSM_C", "4")),
        cold=os.environ.get("NESTED_SSM_COLD", "bf16"),
    )


def rot_flat(x, R):
    """x: (T, G*N) flat B or C; R: (G, N, N).  Same math as ActRotMask.rotmask."""
    T = x.shape[0]
    G, N, _ = R.shape
    return torch.einsum("tgn,gmn->tgm", x.view(T, G, N), R.to(x.dtype)).reshape(T, G * N)


def apply_ckpt(model):
    """Called at the end of NemotronHForCausalLM.load_weights."""
    cfg = _cfg_from_env()
    if cfg is None:
        return
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
    mixers = [m for m in model.modules() if isinstance(m, MambaMixer2)]
    saved = torch.load(cfg["ckpt"], map_location="cpu")
    dev = mixers[0].A.device
    for i, m in enumerate(mixers):
        m.v4R = saved[i].to(dev, torch.float32)
        if "decay" in saved:
            m.A.data.copy_(-torch.exp(saved["decay"]["A_log"][i].float()).to(dev))
            m.dt_bias.data.copy_(saved["decay"]["dt_bias"][i].to(dev))
        if cfg["mode"] == "v4":
            m.v4cfg = dict(pb=cfg["pb"], c=cfg["c"], cold=cfg["cold"])
    print(f"[nested_ssm] ckpt {cfg['ckpt']} applied to {len(mixers)} mixers "
          f"(mode={cfg['mode']} pb={cfg['pb']} c={cfg['c']} cold={cfg['cold']})",
          flush=True)


_COLD_DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}


def _bufs(mixer, ssm_state):
    if mixer._v4buf is not None:
        return mixer._v4buf
    slots, H, P, N = ssm_state.shape
    pb, cold = mixer.v4cfg["pb"], mixer.v4cfg["cold"]
    dev = ssm_state.device
    mixer._v4buf = dict(
        snap=torch.zeros(slots, H, P, N - pb, dtype=_COLD_DT[cold], device=dev),
        scale=(torch.ones(slots, H, P, 1, dtype=torch.float32, device=dev)
               if cold == "fp8" else None),
        glog=torch.zeros(slots, H, dtype=torch.float32, device=dev),
        t=torch.zeros(slots, dtype=torch.int32, device=dev),
    )
    return mixer._v4buf


def _quant_store(buf, slots, s_cold):
    """Write cold snapshot (fp32) into the slot buffer in its storage dtype."""
    if buf["scale"] is not None:                       # fp8: absmax/448 per (h,p)
        sc = s_cold.abs().amax(dim=-1, keepdim=True) / 448.0 + 1e-12
        buf["scale"][slots] = sc
        buf["snap"][slots] = (s_cold / sc).to(buf["snap"].dtype)
    else:
        buf["snap"][slots] = s_cold.to(buf["snap"].dtype)


def _read_snap(buf, slots):
    s = buf["snap"][slots].float()
    if buf["scale"] is not None:
        s = s * buf["scale"][slots]
    return s


def prefill_snapshot(mixer, ssm_state, slots):
    """After prefill wrote fresh states: cold snapshot @ t=0 (warmup=prefill)."""
    if mixer.v4cfg is None:
        return
    if slots.dim() > 1:
        slots = slots.reshape(-1)
    buf = _bufs(mixer, ssm_state)
    pb = mixer.v4cfg["pb"]
    s = ssm_state[slots].float()
    _quant_store(buf, slots, s[..., pb:])
    buf["glog"][slots] = 0.0
    buf["t"][slots] = 0


def decode_readout(mixer, ssm_state, out, x_head, dt_raw, C_d, slots):
    """Override SSU's fresh readout with tiered hot+cold; state stays fresh.

    out:    (b, H, P) preallocated ssm out written by SSU (overwritten here)
    x_head: (b, H, P) conv output hidden states, head view
    dt_raw: (b, H) pre-softplus dt
    C_d:    (b, G, N) post-rotation
    slots:  (b,) destination cache slot per request
    """
    if mixer.v4cfg is None:
        return
    if slots.dim() > 1:        # non-cache-all decode passes (b, q_len=1) 2D indices
        slots = slots[:, -1]
    cfg = mixer.v4cfg
    pb, c = cfg["pb"], cfg["c"]
    buf = _bufs(mixer, ssm_state)
    H = mixer.num_heads
    G = C_d.shape[1]
    Ch = C_d.float().repeat_interleave(H // G, dim=1)          # (b,H,N)
    Sf = ssm_state[slots].float()                              # (b,H,P,N) post-SSU
    glog = buf["glog"][slots]                                  # (b,H)
    y = torch.einsum("bhpn,bhn->bhp", Sf[..., :pb], Ch[..., :pb]) \
        + torch.einsum("bhpn,bhn->bhp", _read_snap(buf, slots), Ch[..., pb:]) \
        * torch.exp(glog)[..., None]
    y = y + x_head.float() * mixer.D.float().view(1, H, 1)   # D: (H,) skip conn
    out.copy_(y.to(out.dtype))
    # bookkeeping (vectorized, graph-safe): refresh every c tokens else decay
    t = buf["t"][slots] + 1
    buf["t"][slots] = t
    flush = (t % c == 0)
    s_cold_new = Sf[..., pb:]
    old = _read_snap(buf, slots)
    _quant_store(buf, slots, torch.where(flush[:, None, None, None], s_cold_new, old))
    dtg = F.softplus(dt_raw.float() + mixer.dt_bias.float().view(1, -1))   # (b,H)
    dtA = dtg * mixer.A.float().view(1, -1)                                # A = -exp
    buf["glog"][slots] = torch.where(flush[:, None], torch.zeros_like(glog),
                                     glog + dtA)
