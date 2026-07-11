"""B200 measured decode speed of v4 tiered execution vs fresh, at Nemotron-9B
mixer dims (L=27 mamba2 mixers, H=128 heads, P=80 head_dim, N=128 d_state;
state/seq = 27*128*80*128*4B = 141.6 MB fp32).

Arms (state-op path only — the component §8 targets; projections/attention/MLP
are identical across arms and analyzed separately as dilution):
  fresh   : fla fused_recurrent_simple_gla, dk=128 (STRONGEST baseline: fused
            read+write ~2S traffic per token)
  v4      : fused hot kernel dk=pb (fresh hot tier) + eager cold readout
            (decay-compensated einsum vs chunk-start snapshot, reads cold 1x)
            + exact cold flush every c tokens (chunk-batched matmul, amortized)
  hotonly : fused kernel dk=pb, cold DISCARDED (the iso-speed narrow baseline;
            same needle 0.46-vs-0.92 comparison point as the acc suite)
  eager3  : fresh in unfused torch (update + readout separate ~3-5S) — what a
            naive engine pays; reported for engine-dependence honesty.

Semantics match nemo9b_eval.naive_mixer_forward mode=v4: hot exact every token,
cold frozen at snapshot with scalar/head decay compensation, exact replay at
chunk boundary.  Speed-only: values random, correctness of v4 established
elsewhere; flush replay here uses the buffered outer-product sum.

Run with SYSTEM python3 (needs fla), e.g.:
  CUDA_VISIBLE_DEVICES=3 python3 bench_v4_decode.py --Bs 64 256 512 1024
"""
import argparse, json
import torch
import triton
import triton.language as tl

L, H, P, N = 27, 128, 80, 128     # Nemotron-9B mamba2 mixer dims


@triton.jit
def _fp8_matvec(Q, S8, SC, Y, Nc: tl.constexpr, Pd: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr):
    """y[bh, p] = sum_n q[bh, n] * scale[bh, n] * fp8(S8[bh, n, p]).
    Fused dequant-matvec: reads cold snapshot at 1 byte/elem — the kernel that
    turns the analytic fp8-cold speedup into a measured one."""
    pid = tl.program_id(0)                                # over B*H
    offs_p = tl.arange(0, BLOCK_P)
    acc = tl.zeros([BLOCK_P], dtype=tl.float32)
    for n0 in range(0, Nc, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        m_n = offs_n < Nc
        qv = tl.load(Q + pid * Nc + offs_n, mask=m_n, other=0.).to(tl.float32)
        sc = tl.load(SC + pid * Nc + offs_n, mask=m_n, other=0.).to(tl.float32)
        s8 = tl.load(S8 + pid * Nc * Pd + offs_n[:, None] * Pd + offs_p[None, :],
                     mask=m_n[:, None] & (offs_p[None, :] < Pd),
                     other=0.).to(tl.float32)
        acc += tl.sum((qv * sc)[:, None] * s8, 0)
    tl.store(Y + pid * Pd + offs_p, acc, mask=offs_p < Pd)


def fp8_cold_readout(qc, S8, scale, out):
    BH = qc.shape[0]
    _fp8_matvec[(BH,)](qc, S8, scale, out, Nc=S8.shape[1], Pd=S8.shape[2],
                       BLOCK_N=32, BLOCK_P=128)
    return out


def timed(fn, iters, warmup, sync_buf=None):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def run_point(B, pb, c, device, iters=20, warmup=5):
    from fla.ops.simple_gla import fused_recurrent_simple_gla as sgla
    dt = torch.bfloat16
    q = torch.randn(B, 1, H, N, device=device, dtype=dt)
    k = torch.randn(B, 1, H, N, device=device, dtype=dt)
    v = torch.randn(B, 1, H, P, device=device, dtype=dt)
    g = torch.full((B, 1, H), -0.05, device=device, dtype=torch.float32)
    res = {}

    # ---- fresh (fused, dk=128) ----
    S = [torch.zeros(B, H, N, P, device=device) for _ in range(L)]
    def fresh():
        for l in range(L):
            o, s2 = sgla(q, k, v, g=g, initial_state=S[l], output_final_state=True)
            S[l].copy_(s2)
    res["fresh_ms"] = timed(fresh, iters, warmup)
    del S; torch.cuda.empty_cache()

    # ---- hot-only (fused, dk=pb, cold discarded) ----
    qh, kh = q[..., :pb].contiguous(), k[..., :pb].contiguous()
    Sh = [torch.zeros(B, H, pb, P, device=device) for _ in range(L)]
    def hotonly():
        for l in range(L):
            o, s2 = sgla(qh, kh, v, g=g, initial_state=Sh[l], output_final_state=True)
            Sh[l].copy_(s2)
    res["hotonly_ms"] = timed(hotonly, iters, warmup)

    # ---- v4: fused hot + eager cold readout + amortized exact flush ----
    Nc = N - pb
    Snap = [torch.zeros(B, H, Nc, P, device=device) for _ in range(L)]  # cold snapshot
    kc_f32 = k[..., pb:].float().squeeze(1)              # (B,H,Nc)
    qc_f32 = q[..., pb:].float().squeeze(1)
    v_f32 = v.float().squeeze(1)                         # (B,H,P)
    comp = torch.ones(B, H, 1, device=device)            # decay compensation exp(Glog)
    # per-token buffers for the chunk flush (written each token, replayed every c)
    bufK = torch.zeros(c, B, H, Nc, device=device)
    bufV = torch.zeros(c, B, H, P, device=device)
    step_idx = [0]
    def v4():
        t = step_idx[0]
        for l in range(L):
            o, s2 = sgla(qh, kh, v, g=g, initial_state=Sh[l], output_final_state=True)
            Sh[l].copy_(s2)                                            # hot tier fresh
            yc = torch.einsum('bhn,bhnp->bhp', qc_f32, Snap[l]) * comp # stale cold read
        bufK[t % c].copy_(kc_f32); bufV[t % c].copy_(v_f32)            # buffer write
        if (t + 1) % c == 0:                                           # exact chunk flush
            for l in range(L):
                Snap[l].mul_(0.98).add_(
                    torch.einsum('cbhn,cbhp->bhnp', bufK, bufV))
        step_idx[0] = t + 1
    it_c = max(iters, 4 * c)                    # exact flush amortization
    res["v4_ms"] = timed(v4, it_c, warmup)
    del Snap, bufK, bufV; torch.cuda.empty_cache()

    # ---- v4-async: flush hidden on side stream, snapshot lagged ONE chunk ----
    # Write path stays exact (flush is exact replay whenever it lands); readout
    # age becomes (c, 2c] instead of (0, c] -> pair async-c with sync-2c acc.
    # Ping-pong snapshots: readers of chunk i+1 read through-(i-1) buffer while
    # flush_i writes the other; chunk-parity double buffers for bufK/bufV.
    side = torch.cuda.Stream()
    SnapP = [[torch.zeros(B, H, Nc, P, device=device) for _ in range(L)]
             for _ in range(2)]
    bufKP = [torch.zeros(c, B, H, Nc, device=device) for _ in range(2)]
    bufVP = [torch.zeros(c, B, H, P, device=device) for _ in range(2)]
    flush_ev = torch.cuda.Event()
    st = {"t": 0, "cur": 0}
    def v4async():
        t, cur = st["t"], st["cur"]
        rd, par = SnapP[cur], (t // c) % 2
        for l in range(L):
            o, s2 = sgla(qh, kh, v, g=g, initial_state=Sh[l], output_final_state=True)
            Sh[l].copy_(s2)
            yc = torch.einsum('bhn,bhnp->bhp', qc_f32, rd[l]) * comp
        bufKP[par][t % c].copy_(kc_f32); bufVP[par][t % c].copy_(v_f32)
        if (t + 1) % c == 0:
            torch.cuda.current_stream().wait_event(flush_ev)  # prev flush landed
            cur = 1 - cur                       # readers advance to through-(i-1)
            wr = SnapP[1 - cur]
            ev = torch.cuda.Event(); ev.record()
            with torch.cuda.stream(side):
                side.wait_event(ev)
                for l in range(L):
                    torch.mul(SnapP[cur][l], 0.98, out=wr[l])
                    wr[l].add_(torch.einsum('cbhn,cbhp->bhnp', bufKP[par], bufVP[par]))
                flush_ev.record(side)
            st["cur"] = cur
        st["t"] = t + 1
    res["v4async_ms"] = timed(v4async, it_c, warmup)
    torch.cuda.synchronize()
    del SnapP, bufKP, bufVP; torch.cuda.empty_cache()

    # ---- v4 bf16-cold: cold snapshot stored bf16 (read-only between flushes)
    # -> halves cold readout + flush bytes; the only lever that works in the
    # BW-saturated regime (async falsified overlap).
    SnapH = [torch.zeros(B, H, Nc, P, device=device, dtype=dt) for _ in range(L)]
    kc_bf = k[..., pb:].contiguous().squeeze(1)          # (B,H,Nc) bf16
    qc_bf = q[..., pb:].contiguous().squeeze(1)
    v_bf = v.squeeze(1)                                  # (B,H,P) bf16
    comp_bf = comp.to(dt)
    bufKh = torch.zeros(c, B, H, Nc, device=device, dtype=dt)
    bufVh = torch.zeros(c, B, H, P, device=device, dtype=dt)
    step_idx2 = [0]
    def v4bf16():
        t = step_idx2[0]
        for l in range(L):
            o, s2 = sgla(qh, kh, v, g=g, initial_state=Sh[l], output_final_state=True)
            Sh[l].copy_(s2)
            yc = torch.einsum('bhn,bhnp->bhp', qc_bf, SnapH[l]) * comp_bf
        bufKh[t % c].copy_(kc_bf); bufVh[t % c].copy_(v_bf)
        if (t + 1) % c == 0:
            for l in range(L):
                SnapH[l].mul_(0.98).add_(
                    torch.einsum('cbhn,cbhp->bhnp', bufKh, bufVh))
        step_idx2[0] = t + 1
    res["v4bf16_ms"] = timed(v4bf16, it_c, warmup)
    del SnapH, bufKh, bufVh; torch.cuda.empty_cache()

    # ---- v4 fp8-cold: scaled fp8 snapshot + Triton fused dequant-matvec readout
    # (reads cold at 1 byte/elem; flush quantizes once per chunk — the asymmetric
    # precision license, measured. Baseline CANNOT do per-token fp8: +5.1% ppl.)
    f8 = torch.float8_e4m3fn
    S8 = [torch.zeros(B * H, Nc, P, device=device, dtype=f8) for _ in range(L)]
    SC8 = [torch.ones(B * H, Nc, device=device) for _ in range(L)]
    qc2 = q[..., pb:].float().reshape(B * H, Nc).contiguous()
    y8 = torch.empty(B * H, P, device=device)
    bufK8 = torch.zeros(c, B, H, Nc, device=device, dtype=dt)
    bufV8 = torch.zeros(c, B, H, P, device=device, dtype=dt)
    acc_cold = torch.zeros(B, H, Nc, P, device=device)   # fp32 master (flush-side)
    step_idx3 = [0]
    def v4fp8():
        t = step_idx3[0]
        for l in range(L):
            o, s2 = sgla(qh, kh, v, g=g, initial_state=Sh[l], output_final_state=True)
            Sh[l].copy_(s2)
            fp8_cold_readout(qc2, S8[l], SC8[l], y8)     # 1B/elem cold read
        bufK8[t % c].copy_(kc_bf); bufV8[t % c].copy_(v_bf)
        if (t + 1) % c == 0:                             # flush: fp32 accum -> fp8 store
            for l in range(L):
                acc_cold.mul_(0.98).add_(
                    torch.einsum('cbhn,cbhp->bhnp', bufK8.float(), bufV8.float()))
                sc = acc_cold.abs().amax(-1).clamp(min=1e-6).div(448.).reshape(B * H, Nc)
                SC8[l].copy_(sc)
                S8[l].copy_((acc_cold.reshape(B * H, Nc, P)
                             / sc[..., None]).to(f8))
        step_idx3[0] = t + 1
    res["v4fp8_ms"] = timed(v4fp8, it_c, warmup)
    del Sh, S8, SC8, bufK8, bufV8, acc_cold; torch.cuda.empty_cache()

    # ---- eager 3-pass fresh (unfused engine reference) ----
    S = [torch.zeros(B, H, N, P, device=device) for _ in range(L)]
    kf, qf = k.float().squeeze(1), q.float().squeeze(1)
    dec = g.exp().squeeze(1)[..., None, None]
    def eager3():
        for l in range(L):
            S[l] = S[l] * dec + torch.einsum('bhn,bhp->bhnp', kf, v_f32)
            y = torch.einsum('bhn,bhnp->bhp', qf, S[l])
    res["eager3_ms"] = timed(eager3, iters, warmup)
    del S; torch.cuda.empty_cache()

    state_gb = L * B * H * N * P * 4 / 1e9
    res.update(B=B, pb=pb, c=c, state_gb=state_gb,
               fresh_toks=B / res["fresh_ms"] * 1e3,
               v4_toks=B / res["v4_ms"] * 1e3,
               v4async_toks=B / res["v4async_ms"] * 1e3,
               v4bf16_toks=B / res["v4bf16_ms"] * 1e3,
               hotonly_toks=B / res["hotonly_ms"] * 1e3,
               eager3_toks=B / res["eager3_ms"] * 1e3,
               speedup_vs_fresh=res["fresh_ms"] / res["v4_ms"],
               speedup_async=res["fresh_ms"] / res["v4async_ms"],
               speedup_bf16=res["fresh_ms"] / res["v4bf16_ms"],
               speedup_fp8=res["fresh_ms"] / res["v4fp8_ms"],
               v4fp8_toks=B / res["v4fp8_ms"] * 1e3,
               speedup_vs_eager3=res["eager3_ms"] / res["v4_ms"],
               fresh_bw_tbs=2 * state_gb / res["fresh_ms"])   # fused ~2S traffic
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Bs", type=int, nargs="+", default=[64, 256, 512, 1024])
    ap.add_argument("--pb", type=int, default=32)
    ap.add_argument("--cs", type=int, nargs="+", default=[4, 8, 16, 64])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.Bs, args.cs = [16], [4]
    device = "cuda"
    out = {"dims": dict(L=L, H=H, P=P, N=N), "cfg": vars(args), "points": []}
    for B in args.Bs:
        for c in args.cs:
            try:
                r = run_point(B, args.pb, c, device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"B={B} c={c}: OOM, skip", flush=True)
                continue
            out["points"].append(r)
            print(f"B={B:>5} pb={args.pb} c={c:>3} | state {r['state_gb']:6.1f}GB | "
                  f"fresh {r['fresh_ms']:7.2f}ms ({r['fresh_toks']:>9,.0f} tok/s, "
                  f"BW~{r['fresh_bw_tbs']:.2f}TB/s) | v4 {r['v4_ms']:7.2f}ms "
                  f"({r['v4_toks']:>9,.0f} tok/s) | v4-ASYNC {r['v4async_ms']:7.2f}ms | "
                  f"v4-BF16COLD {r['v4bf16_ms']:7.2f}ms ({r['v4bf16_toks']:>9,.0f} tok/s) | "
                  f"hot-only {r['hotonly_ms']:6.2f}ms | eager3 {r['eager3_ms']:7.2f}ms | "
                  f"speedup: sync {r['speedup_vs_fresh']:.2f}x ASYNC {r['speedup_async']:.2f}x "
                  f"BF16COLD {r['speedup_bf16']:.2f}x FP8COLD {r['speedup_fp8']:.2f}x "
                  f"(vs fused-fresh)", flush=True)
    json.dump(out, open("bench_v4_decode.json", "w"), indent=1)
    print("saved bench_v4_decode.json", flush=True)


if __name__ == "__main__":
    main()
