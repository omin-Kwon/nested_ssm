"""Motivation experiment: GPU decode throughput is STATE-CAPACITY-BOUND.

For production-adopted linear-recurrent families (GDN = Qwen3.5/Kimi/Nemotron;
GLA = channel-gate family), sweep per-sequence recurrent-state size and measure
REAL decode-step latency with fla fused_recurrent kernels at the max batch that
fits a fixed HBM budget.  Two effects compound:
  (a) capacity: max_batch = (HBM - weights) / state_bytes_per_seq  ~ 1/m
  (b) bandwidth: the state-update kernel touches every state byte per token
      (OI~1), so step time stays pinned at ~budget/BW regardless of m.
=> tokens/s ~ 1/m.  We also report achieved kernel bandwidth to show the
kernel is memory-bound (more FLOPs would not help).

Output: bench_state_bound_{fam}.json  (plot with plot_state_bound.py)
"""
import argparse, json, math, time
import torch
import torch.nn.functional as F

def gdn_step(q, k, v, g, beta, states, Wq, Wk, Wv, Wo, Wup, Wdn, x):
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    outs = []
    for S in states:
        # projections (real GEMMs at this batch)
        qh = (x @ Wq).view(*q.shape); kh = (x @ Wk).view(*k.shape)
        vh = (x @ Wv).view(*v.shape)
        o, S2 = fused_recurrent_gated_delta_rule(
            qh, kh, vh, g=g, beta=beta, initial_state=S,
            output_final_state=True, use_qk_l2norm_in_kernel=True)
        S.copy_(S2)
        y = o.reshape(x.shape[0], 1, -1) @ Wo
        h = (y.squeeze(1) @ Wup); h = F.silu(h) @ Wdn
        outs.append(h)
    return outs

def gla_step(q, k, v, gk, states, Wq, Wk, Wv, Wo, Wup, Wdn, x):
    from fla.ops.gla import fused_recurrent_gla
    outs = []
    for S in states:
        qh = (x @ Wq).view(*q.shape); kh = (x @ Wk).view(*k.shape)
        vh = (x @ Wv).view(*v.shape)
        o, S2 = fused_recurrent_gla(qh, kh, vh, gk=gk, initial_state=S,
                                    output_final_state=True)
        S.copy_(S2)
        y = o.reshape(x.shape[0], 1, -1) @ Wo
        h = (y.squeeze(1) @ Wup); h = F.silu(h) @ Wdn
        outs.append(h)
    return outs

def run_point(fam, m, B, L, d, H, dk0, dv, device, axis="heads",
              iters=30, warmup=8):
    K = dk0 * (m if axis == "dk" else 1)
    H = H * (m if axis == "heads" else 1)
    dt = torch.bfloat16
    x = torch.randn(B, d, device=device, dtype=dt)
    q = torch.empty(B, 1, H, K, device=device, dtype=dt)
    k = torch.empty(B, 1, H, K, device=device, dtype=dt)
    v = torch.empty(B, 1, H, dv, device=device, dtype=dt)
    Wq = torch.randn(d, H * K, device=device, dtype=dt) * 0.02
    Wk = torch.randn(d, H * K, device=device, dtype=dt) * 0.02
    Wv = torch.randn(d, H * dv, device=device, dtype=dt) * 0.02
    Wo = torch.randn(H * dv, d, device=device, dtype=dt) * 0.02
    Wup = torch.randn(d, 4 * d, device=device, dtype=dt) * 0.02
    Wdn = torch.randn(4 * d, d, device=device, dtype=dt) * 0.02
    states = [torch.zeros(B, H, K, dv, device=device, dtype=torch.float32)
              for _ in range(L)]
    if fam == "gdn":
        g = torch.full((B, 1, H), -0.05, device=device, dtype=torch.float32)
        beta = torch.full((B, 1, H), 0.9, device=device, dtype=dt)
        step = lambda: gdn_step(q, k, v, g, beta, states, Wq, Wk, Wv, Wo, Wup, Wdn, x)
    else:
        gk = torch.full((B, 1, H, K), -0.05, device=device, dtype=torch.float32)
        step = lambda: gla_step(q, k, v, gk, states, Wq, Wk, Wv, Wo, Wup, Wdn, x)
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
    t0.record()
    for _ in range(iters):
        step()
    t1.record(); torch.cuda.synchronize()
    ms = t0.elapsed_time(t1) / iters
    # kernel-only timing (one layer) for achieved-bandwidth evidence
    if fam == "gdn":
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        kern = lambda: fused_recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=states[0],
            output_final_state=True, use_qk_l2norm_in_kernel=True)
    else:
        from fla.ops.gla import fused_recurrent_gla
        kern = lambda: fused_recurrent_gla(q, k, v, gk=gk,
                                           initial_state=states[0],
                                           output_final_state=True)
    for _ in range(warmup):
        kern()
    torch.cuda.synchronize()
    t0.record()
    for _ in range(iters):
        kern()
    t1.record(); torch.cuda.synchronize()
    kms = t0.elapsed_time(t1) / iters
    state_bytes = L * H * K * dv * 4                      # fp32 state (fla)
    kernel_traffic = 2 * B * (H * K * dv * 4)             # read+write, 1 layer
    return dict(step_ms=ms, toks=B / (ms / 1e3),
                state_gb_seq=state_bytes / 1e9, kernel_ms=kms,
                bw_tbs=kernel_traffic / (kms / 1e3) / 1e12,
                kern_frac=kms * L / ms)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", choices=["gdn", "gla"], default="gdn")
    ap.add_argument("--budgets", type=float, nargs="+", default=[80.0, 160.0])
    ap.add_argument("--ms", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--L", type=int, default=24)
    ap.add_argument("--d", type=int, default=2048)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--dk0", type=int, default=128)
    ap.add_argument("--dv", type=int, default=256)
    ap.add_argument("--cap", type=int, default=4096, help="max batch cap")
    ap.add_argument("--axis", choices=["heads", "dk"], default="heads")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.budgets, args.ms, args.L = [4.0], [1, 4], 4
    device = "cuda"
    out = {"cfg": vars(args), "points": []}
    for m in args.ms:
        K = args.dk0 * (m if args.axis == "dk" else 1)
        H = args.H * (m if args.axis == "heads" else 1)
        # weights of the full L-layer model (bf16), charged against the budget
        w_layer = (2 * args.d * H * K + args.d * H * args.dv
                   + H * args.dv * args.d + 8 * args.d * args.d)
        wb = 2 * w_layer * args.L
        st_seq = args.L * H * K * args.dv * 4
        for budget in args.budgets:
            free = budget * 1e9 - wb - 2e9                # 2GB activation slack
            B = min(args.cap, int(free // st_seq))
            if B < 1:
                print(f"[{args.fam}] m={m} budget={budget}: batch=0, skip", flush=True)
                continue
            try:
                r = run_point(args.fam, m, B, args.L, args.d, args.H,
                              args.dk0, args.dv, device, axis=args.axis)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[{args.fam}] m={m} B={B} OOM on this GPU, skip", flush=True)
                continue
            r.update(m=m, B=B, budget=budget)
            out["points"].append(r)
            print(f"[{args.fam}] m={m:>2} budget={budget:>5.0f}GB B={B:>5} "
                  f"state/seq={r['state_gb_seq']:.3f}GB step={r['step_ms']:.2f}ms "
                  f"tok/s={r['toks']:,.0f} kernelBW~{r['bw_tbs']:.2f}TB/s", flush=True)
            torch.cuda.empty_cache()
    json.dump(out, open(f"bench_state_bound_{args.fam}_{args.axis}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
