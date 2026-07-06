"""Debug: can the (non-nested) gated-delta recurrence solve MQAR D=8 at all?
Isolates architecture/data from the Matryoshka machinery."""
import argparse, time, torch, torch.nn.functional as F
from nested_delta_mqar import make_mqar, NestedDeltaLM

def run(D, width, heads, head_dim, layers, steps, lr, nk, nv, nq, batch, seed=0):
    dev = "cuda"
    vocab = 1 + nk + nv
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    m = NestedDeltaLM(vocab, 128, layers, heads, head_dim).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    # show one sample
    inp, tgt = make_mqar(2, D, nq, nk, nv, dev, gen)
    print("sample inp[0]:", inp[0].tolist())
    print("sample tgt[0]:", [t for t in tgt[0].tolist()])
    for s in range(steps):
        inp, tgt = make_mqar(batch, D, nq, nk, nv, dev, gen)
        opt.zero_grad()
        logits = m(inp, width)
        loss = F.cross_entropy(logits.view(-1, vocab), tgt.view(-1), ignore_index=-100)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s+1) % (steps//8) == 0:
            m.eval()
            with torch.no_grad():
                ii, tt = make_mqar(1024, D, nq, nk, nv, dev, gen)
                lg = m(ii, width); mask = tt != -100
                acc = (lg.argmax(-1)[mask] == tt[mask]).float().mean().item()
            m.train()
            print(f"  step {s+1:4d} loss {loss.item():.3f} recall {acc:.3f}", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--D", type=int, default=8)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--head_dim", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--nk", type=int, default=64)
    p.add_argument("--nv", type=int, default=32)
    p.add_argument("--nq", type=int, default=16)
    p.add_argument("--batch", type=int, default=256)
    a = p.parse_args()
    print(f"D={a.D} width={a.width} heads={a.heads} head_dim={a.head_dim} "
          f"layers={a.layers} lr={a.lr}")
    run(a.D, a.width, a.heads, a.head_dim, a.layers, a.steps, a.lr,
        a.nk, a.nv, a.nq, a.batch)
