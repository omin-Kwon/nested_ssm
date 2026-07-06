"""T4c: rotation-only retrofit on the REAL dedicated GDN (per-head R, id-init).
GDN decay is per-head scalar -> orthogonal key-rotation invariance holds
(the Qwen3.5 family case). Freeze backbone, train only R with Matryoshka loss."""
import torch, torch.nn as nn, torch.nn.functional as F, sys, time
sys.path.insert(0, "/home/omin/TTT-PNM/poc")
from nested_delta_mqar import make_imqar
from gdn_a4 import GDNLM, eval_grid
device = "cuda"
nk, nv, nq = 256, 128, 24
vocab = 1 + nk + nv
widths = [8, 16, 32, 64]
Ds = [8, 16, 32, 64]
model = GDNLM(vocab).to(device).bfloat16()
model.load_state_dict(torch.load("gdn_a4_dedic.pt"))
for p in model.parameters(): p.requires_grad_(False)
rots = []
for b in model.blocks:
    H, K = b.attn.num_heads, b.attn.head_k_dim
    b.attn.rot = nn.Parameter(torch.eye(K, device=device, dtype=torch.bfloat16)
                              .expand(H, K, K).clone())
    rots.append(b.attn.rot)
opt = torch.optim.AdamW(rots, lr=1e-2, weight_decay=0.0)
gen = torch.Generator(device=device); gen.manual_seed(0)
eye = torch.eye(64, device=device, dtype=torch.float32)
print("BEFORE adapt (dedicated truncated):")
g = eval_grid(model, Ds, widths, nk, nv, nq, device, gen, batch=512)
for D in Ds: print(f"{D:<4d} " + " ".join(f"{g[D][w]:6.3f}" for w in widths), flush=True)
t0 = time.time()
for step in range(3000):
    D = Ds[torch.randint(len(Ds), (1,), generator=gen, device=device).item()]
    inp, tgt = make_imqar(192, D, nq, nk, nv, device, gen)
    ws = torch.tensor(widths, device=device)[
        torch.randint(len(widths), (inp.shape[0],), generator=gen, device=device)]
    opt.zero_grad()
    logits = model(inp, width=ws)
    loss = F.cross_entropy(logits.float().view(-1, vocab), tgt.view(-1), ignore_index=-100)
    loss = loss + 0.1 * sum(((R.float().transpose(-1, -2) @ R.float() - eye) ** 2).sum()
                            for R in rots)
    loss.backward(); opt.step()
    if (step + 1) % 1500 == 0:
        print(f"step {step+1} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
print("AFTER rot-only Matryoshka FT:")
g = eval_grid(model, Ds, widths, nk, nv, nq, device, gen, batch=1024)
for D in Ds: print(f"{D:<4d} " + " ".join(f"{g[D][w]:6.3f}" for w in widths), flush=True)
print("(reference from-scratch nested FINAL: k8 col 0.999/0.994/0.970/0.871)")
