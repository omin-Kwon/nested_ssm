"""T10: rotation-only nested retrofit on nvidia/NVIDIA-Nemotron-Nano-9B-v2
(REAL public pretrained, Mamba2/SSD hybrid — 27 mamba2 + 4 attn layers).
Scalar per-head decay => rotation-invariance in the B/C (key/query) state basis
=> retrofit trains ONLY per-group rotations R (identity-init, orth penalty),
backbone fully frozen. Nesting axis = d_state (128), menu {16,32,64,128}.

Injection without touching remote code: mixer.torch_forward applies `self.act`
once (train path) to the conv'd [x|B|C] tensor -> we wrap self.act to
rotate+mask the B and C slices per group. R=I & w=128 == original model
(validation gate)."""
import argparse, math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

class ActRotMask(nn.Module):
    def __init__(self, act, inter, ngroups, dstate):
        super().__init__()
        self.act = act
        self.inter, self.ng, self.ds = inter, ngroups, dstate
        self.conv_dim = inter + 2 * ngroups * dstate
        self.R = nn.Parameter(torch.eye(dstate).repeat(ngroups, 1, 1))  # (G,N,N)
        self.width = dstate

    def rotmask(self, x):                                  # x: (B,T,G*N)
        b, t, _ = x.shape
        x = x.view(b, t, self.ng, self.ds)
        x = torch.einsum('btgn,gmn->btgm', x, self.R.to(x.dtype))
        w = self.width
        if torch.is_tensor(w):
            idx = torch.arange(self.ds, device=x.device)
            x = x * (idx[None, :] < w[:, None]).to(x.dtype)[:, None, None, :]
        elif w < self.ds:
            x = torch.cat([x[..., :w], torch.zeros_like(x[..., w:])], -1)
        return x.reshape(b, t, self.ng * self.ds)

    def forward(self, x):
        x = self.act(x)
        if x.dim() == 3 and x.shape[-1] == self.conv_dim:
            h, B, C = torch.split(
                x, [self.inter, self.ng * self.ds, self.ng * self.ds], dim=-1)
            x = torch.cat([h, self.rotmask(B), self.rotmask(C)], dim=-1)
        return x

def get_wrappers(model):
    return [m for m in model.modules() if isinstance(m, ActRotMask)]

def set_width(model, w):
    for m in get_wrappers(model):
        m.width = w

def batches(train, bsz, seqlen, gen, device):
    ix = torch.randint(0, len(train) - seqlen - 1, (bsz,), generator=gen, device=device).cpu()
    x = torch.stack([torch.from_numpy(train[i:i + seqlen].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(train[i + 1:i + 1 + seqlen].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def vppl(model, val, w, seqlen=1024, n=6, device="cuda"):
    model.eval(); set_width(model, w); tot = 0.0
    for b in range(n):
        s = b * seqlen * 4
        row = torch.from_numpy(val[s:s + seqlen + 1].astype(np.int64))[None].to(device)
        logits = model(row[:, :-1]).logits
        tot += F.cross_entropy(logits.float().view(-1, logits.shape[-1]),
                               row[:, 1:].reshape(-1)).item()
    model.train(); return math.exp(tot / n)

def _stub_mamba_ssm():
    """Remote code hard-requires mamba_ssm only for rmsnorm_fn; stub it with an
    exact pure-torch equivalent (group RMS-norm, gate via silu). Fast-path
    detection uses importlib.metadata, so kernels stay disabled -> torch_forward."""
    import sys, types
    def rmsnorm_fn(x, weight, bias, z=None, eps=1e-6, group_size=None,
                   norm_before_gate=True):
        dt = x.dtype
        x = x.float()
        if z is not None and not norm_before_gate:
            x = x * F.silu(z.float())
        gs = group_size or x.shape[-1]
        s = x.shape
        xg = x.view(*s[:-1], s[-1] // gs, gs)
        x = (xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)).view(s)
        x = x * weight.float()
        if bias is not None:
            x = x + bias.float()
        if z is not None and norm_before_gate:
            x = x * F.silu(z.float())
        return x.to(dt)
    import importlib.machinery
    for name in ["mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton",
                 "mamba_ssm.ops.triton.layernorm_gated"]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__spec__ = importlib.machinery.ModuleSpec(name, None)
            m.__path__ = []
            sys.modules[name] = m
    sys.modules["mamba_ssm.ops.triton.layernorm_gated"].rmsnorm_fn = rmsnorm_fn

def main():
    # NOTE: run with ~/nemo_env/bin/python3 (transformers 5.13 native NemotronH).
    # The remote-code path (trust_remote_code + mamba_ssm stub) produced a BROKEN
    # forward (ppl ~3700, degenerate generation); native torch path gives ppl 8.3.
    from transformers import AutoModelForCausalLM
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seqlen", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--orth", type=float, default=1e-2)
    p.add_argument("--widths", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--fixed_only", type=int, default=0)
    p.add_argument("--save", default="nemo9b_rot.pt")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    train = np.load("wt103_train_nemo.npy", mmap_mode="r")
    val = np.load("wt103_val_nemo.npy")
    gen = torch.Generator(device=device); gen.manual_seed(args.seed)
    tag = "nemo9b-rot" if not args.fixed_only else f"nemo9b-fixed{args.fixed_only}"
    model = AutoModelForCausalLM.from_pretrained(
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2", dtype=torch.bfloat16).to(device)
    for q in model.parameters():
        q.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    # baseline ppl BEFORE wrapping
    with torch.no_grad():
        tot = 0.0
        for b in range(6):
            s = b * args.seqlen * 4
            row = torch.from_numpy(val[s:s + args.seqlen + 1].astype(np.int64))[None].to(device)
            logits = model(row[:, :-1]).logits
            tot += F.cross_entropy(logits.float().view(-1, logits.shape[-1]),
                                   row[:, 1:].reshape(-1)).item()
    p_orig = math.exp(tot / 6)
    # wrap every mamba2 mixer's act
    nmix = 0
    for m in model.modules():
        if type(m).__name__ == "NemotronHMamba2Mixer":
            w = ActRotMask(m.act, m.intermediate_size, m.n_groups, m.ssm_state_size)
            w.R.data = w.R.data.float()
            m.act = w.to(device)
            nmix += 1
    print(f"[{tag}] wrapped {nmix} mamba2 mixers", flush=True)
    p_patch = vppl(model, val, 128, device=device)
    print(f"[{tag}] VALIDATION original={p_orig:.2f} patched-full={p_patch:.2f}", flush=True)
    print(f"[{tag}] PRE-FT ppl vs width: " +
          " ".join(f"k{w}:{vppl(model, val, w, device=device):.2f}" for w in args.widths),
          flush=True)
    Rs = [m.R for m in get_wrappers(model)]
    for R in Rs:
        R.requires_grad_(True)
    opt = torch.optim.AdamW(Rs, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    widths = args.widths if not args.fixed_only else [args.fixed_only]
    eye = torch.eye(128, device=device)
    model.train()
    t0 = time.time()
    for step in range(args.steps):
        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        for g_ in opt.param_groups: g_["lr"] = lr
        x, y = batches(train, args.batch, args.seqlen, gen, device)
        if len(widths) > 1:
            ws = torch.tensor(widths, device=device)[
                torch.randint(len(widths), (x.shape[0],), generator=gen, device=device)]
            set_width(model, ws)
        else:
            set_width(model, widths[0])
        logits = model(x).logits
        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.reshape(-1))
        orth = sum(((R.transpose(-1, -2) @ R) - eye).pow(2).mean() for R in Rs) / len(Rs)
        (loss + args.orth * orth).backward()
        torch.nn.utils.clip_grad_norm_(Rs, 1.0)
        opt.step(); opt.zero_grad()
        with torch.no_grad():                        # QR retraction: R stays exactly
            for R in Rs:                             # orthogonal -> full width EXACTLY
                Q, Rr = torch.linalg.qr(R.data)      # preserved (rotation invariance)
                sgn = torch.sign(torch.diagonal(Rr, dim1=-2, dim2=-1))
                R.data = Q * sgn[..., None, :]
        if (step + 1) % args.log_every == 0:
            print(f"[{tag}] step {step+1} loss {loss.item():.3f} orth {orth.item():.4f} "
                  f"({time.time()-t0:.0f}s) " +
                  " ".join(f"k{w}:{vppl(model, val, w, device=device):.2f}" for w in args.widths),
                  flush=True)
    print(f"[{tag}] FINAL " +
          " ".join(f"k{w}:{vppl(model, val, w, device=device):.2f}" for w in args.widths), flush=True)
    torch.save({i: R.detach().cpu() for i, R in enumerate(Rs)}, args.save)
    print(f"[{tag}] saved {args.save}", flush=True)

if __name__ == "__main__":
    main()
