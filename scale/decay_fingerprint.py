"""A2: per-dim decay statistics of a pretrained GLA (zero training).
Question: do dims already separate into slow (stale-eligible/cold) vs
fast (recency/hot) timescales — a free placement signal?"""
import torch, fla, inspect, math
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "fla-hub/gla-340M-15B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                             trust_remote_code=True).cuda().eval()
cfg = model.config
# find the exact gk transform in the forward source
src = inspect.getsource(type(model.model.layers[0].attn).forward)
norm = cfg.gate_logit_normalizer if hasattr(cfg, "gate_logit_normalizer") else 16
print("gate_logit_normalizer:", norm)
print("forward uses:", [l.strip() for l in src.splitlines() if "logsigmoid" in l or "gk" in l][:6])

TEXT = ("The history of computing spans centuries, from mechanical calculators to "
        "modern datacenters. Charles Babbage designed the Analytical Engine in the "
        "nineteenth century, and Ada Lovelace wrote what many consider the first "
        "algorithm intended for a machine. During the Second World War, electronic "
        "computers such as Colossus and ENIAC were built to solve urgent numerical "
        "problems. The invention of the transistor at Bell Labs in 1947 transformed "
        "the field, enabling smaller and more reliable machines. Integrated circuits "
        "followed, and with them Moore's law, the observation that transistor counts "
        "double roughly every two years. Operating systems, compilers, and networks "
        "grew alongside the hardware. The internet emerged from ARPANET research, "
        "linking universities and laboratories before reaching homes worldwide. "
        "Personal computers brought computing to individuals, while smartphones put "
        "it in their pockets. Today, machine learning models trained on vast corpora "
        "run in datacenters filled with accelerators, and researchers explore new "
        "memory hierarchies to feed them efficiently. ") * 12

ids = tok(TEXT, return_tensors="pt").input_ids[:, :3072].cuda()
print("tokens:", ids.shape)

# hook gk_proj outputs per layer
gks = {}
hooks = []
for li, layer in enumerate(model.model.layers):
    def mk(li):
        def h(mod, i, o):
            # o: (B, L, key_dim_total) pre-logsigmoid gate logits
            g = torch.nn.functional.logsigmoid(o.float()) / norm   # log alpha
            gks[li] = g.detach()[0]                                 # (L, K)
        return h
    hooks.append(layer.attn.gk_proj.register_forward_hook(mk(li)))
with torch.no_grad():
    model(ids)
for h in hooks: h.remove()

# stats: per-dim mean log-decay -> timescale tau = -1/mean_log_alpha
import numpy as np
all_tau = []
print(f"\n{'layer':>5s} {'tau p5':>8s} {'p25':>8s} {'median':>8s} {'p75':>8s} {'p95':>8s} {'max':>9s}  spread(p95/p5)")
for li in sorted(gks):
    la = gks[li].mean(0)                       # (K,) mean log alpha per dim
    tau = (-1.0 / la.clamp(max=-1e-6)).cpu().numpy()   # tokens
    all_tau.append(tau)
    q = np.percentile(tau, [5, 25, 50, 75, 95])
    if li % 4 == 0 or li == 23:
        print(f"{li:>5d} {q[0]:>8.1f} {q[1]:>8.1f} {q[2]:>8.1f} {q[3]:>8.1f} {q[4]:>8.1f} {tau.max():>9.1f}  {q[4]/q[0]:>6.1f}x")
A = np.stack(all_tau)                          # (layers, K)
print(f"\nglobal: dims with tau<8 tok: {(A<8).mean()*100:.0f}% | 8-64: {((A>=8)&(A<64)).mean()*100:.0f}% "
      f"| 64-512: {((A>=64)&(A<512)).mean()*100:.0f}% | >=512: {(A>=512).mean()*100:.0f}%")
np.save("gla340m_tau.npy", A)
print("saved gla340m_tau.npy")
