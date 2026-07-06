"""
PoC: Nested (Matryoshka) Gated-DeltaNet state on MQAR.

Hypothesis under test
---------------------
If we train a gated delta-rule recurrent LM with *nested dropout on the key
dimension of the recurrent state* (Matryoshka-style), then at inference a single
trained model can be run at any state width k, and:
  (H1) MQAR recall degrades gracefully & monotonically as k shrinks
       -> a genuine *runtime dial* exists (no retraining).
  (H2) larger #associations D needs larger k for the same recall
       -> the width behaves like a *capacity* knob (matches Zoology:
          state size == recall capacity).
  (H3) the nested model at width k is close to a model *trained* at fixed
       width k (small "Matryoshka tax").

State is S in R^{H x Dv x Dk}. Nesting truncates the KEY dim Dk -> first m cols.

This is a decisive small-scale test, not a scaled model.
"""
import argparse, math, time
import torch, torch.nn as nn, torch.nn.functional as F

# --------------------------- MQAR data (Zoology-style) ---------------------------
def make_mqar(batch, D, n_query, n_keys, n_vals, device, gen, return_age=False):
    """Sequence = [k1,v1,...,kD,vD, q1,...,qQ].
    Predict value_of(qi) from the model output at qi's position (next-token style).
    keys in [1, n_keys], values in [n_keys+1, n_keys+n_vals], 0 = blank.
    Returns input_ids (B,L), targets (B,L) with -100 except at query positions.
    return_age: also return (B,L) age = query_pos - value_write_pos (-1 elsewhere)."""
    KEY0, VAL0 = 1, 1 + n_keys
    L = 2 * D + n_query
    inp = torch.zeros(batch, L, dtype=torch.long, device=device)
    tgt = torch.full((batch, L), -100, dtype=torch.long, device=device)
    age = torch.full((batch, L), -1, dtype=torch.long, device=device)
    for b in range(batch):
        keys = (torch.randperm(n_keys, generator=gen, device=device)[:D] + KEY0)
        vals = (torch.randint(n_vals, (D,), generator=gen, device=device) + VAL0)
        # kv block
        inp[b, 0:2 * D:2] = keys
        inp[b, 1:2 * D:2] = vals
        # queries: sample n_query of the D keys (with replacement if n_query>D)
        qidx = torch.randint(D, (n_query,), generator=gen, device=device)
        qpos = torch.arange(2 * D, 2 * D + n_query, device=device)
        inp[b, qpos] = keys[qidx]
        tgt[b, qpos] = vals[qidx]
        age[b, qpos] = qpos - (2 * qidx + 1)               # dist from value write
    return (inp, tgt, age) if return_age else (inp, tgt)

def make_imqar(batch, D, n_query, n_keys, n_vals, device, gen, return_age=False):
    """INTERLEAVED MQAR: writes and queries are mixed along the sequence, so
    queries have small ages (young targets) — required to test read-recency /
    staleness effects (the separated layout above puts all writes before all
    queries, making stale READOUT trivially safe: a layout artifact)."""
    KEY0, VAL0 = 1, 1 + n_keys
    B, nq = batch, n_query
    L = 2 * D + nq
    E = D + nq
    # batched sample-without-replacement via argsort trick
    keys = torch.argsort(torch.rand(B, n_keys, generator=gen, device=device),
                         dim=1)[:, :D] + KEY0
    vals = torch.randint(n_vals, (B, D), generator=gen, device=device) + VAL0
    # event order (0=write, 1=query), shuffled; force a write into slot 0
    ev = torch.cat([torch.zeros(B, D, dtype=torch.long, device=device),
                    torch.ones(B, nq, dtype=torch.long, device=device)], dim=1)
    perm = torch.argsort(torch.rand(B, E, generator=gen, device=device), dim=1)
    ev = torch.gather(ev, 1, perm)
    bar = torch.arange(B, device=device)
    first_w = torch.argmax((ev == 0).long(), dim=1)
    ev[bar, first_w] = ev[:, 0]
    ev[:, 0] = 0
    cost = 2 - ev                                          # write=2 tokens, query=1
    start = torch.cumsum(cost, dim=1) - cost               # token pos of each event
    nw_before = torch.cumsum((ev == 0).long(), dim=1) - (ev == 0).long()
    inp = torch.zeros(B, L, dtype=torch.long, device=device)
    tgt = torch.full((B, L), -100, dtype=torch.long, device=device)
    age = torch.full((B, L), -1, dtype=torch.long, device=device)
    bidx = bar.unsqueeze(1).expand(B, E)
    wmask = ev == 0
    bw, sw, wi = bidx[wmask], start[wmask], nw_before[wmask]
    inp[bw, sw] = keys[bw, wi]
    inp[bw, sw + 1] = vals[bw, wi]
    wpos = torch.zeros(B, D, dtype=torch.long, device=device)
    wpos[bw, wi] = sw + 1                                  # value-write positions
    qmask = ev == 1
    bq, sq, nwq = bidx[qmask], start[qmask], nw_before[qmask]
    j = (torch.rand(nwq.shape, generator=gen, device=device) * nwq).long()
    inp[bq, sq] = keys[bq, j]
    tgt[bq, sq] = vals[bq, j]
    age[bq, sq] = sq - wpos[bq, j]
    return (inp, tgt, age) if return_age else (inp, tgt)

DATA_FN = make_mqar        # set by --task (mqar | imqar)

# --------------------------- Nested gated delta layer ---------------------------
def causal_shortconv(u, conv):
    # u: (B,L,C) -> depthwise causal conv over time
    B, L, C = u.shape
    x = u.transpose(1, 2)                       # (B,C,L)
    k = conv.weight.shape[-1]
    x = F.conv1d(F.pad(x, (k - 1, 0)), conv.weight, conv.bias, groups=C)
    return F.silu(x.transpose(1, 2))            # (B,L,C)

def _grans(head_dim):
    g, w = [], 2
    while w <= head_dim:
        g.append(w); w *= 2
    return g                                   # [2,4,8,...,head_dim]

class NestedGatedDelta(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, mode="additive", kconv=4, pipe_b=8):
        super().__init__()
        self.H, self.Dh, self.mode = n_heads, head_dim, mode
        self.pipe_b = pipe_b                   # hot/cold boundary for pipedelta
        self.grans = _grans(head_dim)          # nested block boundaries
        self.spans = list(zip([0] + self.grans[:-1], self.grans))  # [(0,2),(2,4),...]
        inner = n_heads * head_dim
        self.q = nn.Linear(d_model, inner, bias=False)
        self.k = nn.Linear(d_model, inner, bias=False)
        self.v = nn.Linear(d_model, inner, bias=False)
        self.qc = nn.Conv1d(inner, inner, kconv, groups=inner)  # short conv (GDN)
        self.kc = nn.Conv1d(inner, inner, kconv, groups=inner)
        self.vc = nn.Conv1d(inner, inner, kconv, groups=inner)
        self.beta = nn.Linear(d_model, n_heads, bias=True)    # write strength
        self.alpha = nn.Linear(d_model, n_heads, bias=True)   # forgetting gate
        self.o = nn.Linear(inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.rot = None                        # optional key-space rotation (E8)
        self.tier_w = False                    # v4 tier-local writes (stale eval)
        if mode == "blockdelta":               # per-block write gates
            self.beta_b = nn.Linear(d_model, n_heads * len(self.spans), bias=True)

    def add_rotation(self):
        """E8 retrofit: learnable key-space rotation, identity-init.
        Delta recurrence is invariant under orthogonal R on (q,k) at full width,
        so training starts function-preserving; R only has to learn ORDERING."""
        self.rot = nn.Parameter(torch.eye(self.Dh, device=self.o.weight.device))

    def forward(self, x, width, read_width=None, stale_c=0, stale_pb=8):
        B, L, _ = x.shape
        H, Dh, m = self.H, self.Dh, width
        h = self.norm(x)
        q = causal_shortconv(self.q(h), self.qc).view(B, L, H, Dh)
        k = causal_shortconv(self.k(h), self.kc).view(B, L, H, Dh)
        v = causal_shortconv(self.v(h), self.vc).view(B, L, H, Dh)
        alpha = torch.sigmoid(self.alpha(h))               # (B,L,H) in (0,1) decay
        if self.mode == "blockdelta":
            return x + self.o(self._blockdelta(q, k, v, alpha, h, m, read_width))
        q = F.normalize(q, dim=-1)          # L2 norm (GDN)
        k = F.normalize(k, dim=-1)
        if self.rot is not None:                           # E8: reorder key space
            q = q @ self.rot.T
            k = k @ self.rot.T
        beta = torch.sigmoid(self.beta(h))                 # (B,L,H) in (0,1)
        qm = q[..., :m]; km = k[..., :m]                   # nest: first m key dims
        if self.mode == "nesteddelta":
            return x + self.o(self._nesteddelta(qm, km, v, alpha, beta, m))
        if stale_c > 0 and self.mode == "delta":
            return x + self.o(self._staledelta(qm, km, v, alpha, beta, m,
                                               stale_c, stale_pb,
                                               corr_fresh=(stale_pb == 0)))
        S = x.new_zeros(B, H, Dh, m)                        # value(full) x key(m)
        pb = min(self.pipe_b, m)                            # hot/cold boundary (pipedelta)
        outs = []
        for t in range(L):
            k_t = km[:, t]                                  # (B,H,m)
            q_t = qm[:, t]
            v_t = v[:, t]                                   # (B,H,Dh)
            a_t = alpha[:, t]                               # (B,H)
            b_t = beta[:, t]
            if self.mode in ("delta", "pipedelta"):
                Sk = (S * k_t.unsqueeze(2)).sum(-1)         # S@k (pre-update, exact)
                write = (b_t[..., None] * (v_t - a_t.unsqueeze(-1) * Sk))
            else:                                           # additive gated linear attn
                write = b_t[..., None] * v_t
            if self.mode == "pipedelta" and m > pb:
                # cold readout on PRE-update (decayed) state -> single pipelined
                # GPU<->PNM exchange per token; write visible in cold dims at t+1
                y_cold = a_t.unsqueeze(-1) * (
                    S[..., pb:] * q_t[..., pb:].unsqueeze(2)).sum(-1)
                S = a_t[..., None, None] * S + write.unsqueeze(-1) * k_t.unsqueeze(2)
                y_hot = (S[..., :pb] * q_t[..., :pb].unsqueeze(2)).sum(-1)
                y_t = y_hot + y_cold
            else:
                S = a_t[..., None, None] * S + write.unsqueeze(-1) * k_t.unsqueeze(2)
                y_t = (S * q_t.unsqueeze(2)).sum(-1)       # (B,H,Dh)  S@q
            outs.append(y_t)
        y = torch.stack(outs, 1).reshape(B, L, H * Dh)
        return x + self.o(y)

    def _blockdelta(self, q, k, v, alpha, h, m, read_width=None):
        """Block-diagonal nested delta: every active block runs its OWN
        self-correcting delta over its key slice on the SAME v (parallel sum,
        no residual chain). Per-block L2 norm + per-block write gate make each
        block's trajectory independent of total active width -> exact prefix
        consistency; blocks map 1:1 to hardware tiers (hot=GPU, cold=PNM) with
        zero cross-tier coupling. read_width: sum readouts only up to this
        boundary (state updates still run up to m) — used for the prefix test."""
        B, L, H, Dh = v.shape
        spans = [(s, e) for s, e in self.spans if e <= m]
        assert spans and spans[-1][1] == m, f"width {m} must be a block boundary"
        rw = m if read_width is None else read_width
        bb = torch.sigmoid(self.beta_b(h)).view(B, L, H, len(self.spans))
        Ks = [F.normalize(k[..., s:e], dim=-1) for s, e in spans]   # per-block norm
        Qs = [F.normalize(q[..., s:e], dim=-1) for s, e in spans]
        Sb = [v.new_zeros(B, H, Dh, e - s) for s, e in spans]
        outs = []
        for t in range(L):
            a_t = alpha[:, t]; v_t = v[:, t]
            y_t = 0.0
            for j, (s, e) in enumerate(spans):
                k_j = Ks[j][:, t]; q_j = Qs[j][:, t]                # (B,H,w_j)
                b_j = bb[:, t, :, j]                                # (B,H)
                Sk = (Sb[j] * k_j.unsqueeze(2)).sum(-1)             # S_j @ k_j
                w = b_j[..., None] * (v_t - a_t.unsqueeze(-1) * Sk) # own-block delta
                Sb[j] = a_t[..., None, None] * Sb[j] + w.unsqueeze(-1) * k_j.unsqueeze(2)
                if e <= rw:
                    y_t = y_t + (Sb[j] * q_j.unsqueeze(2)).sum(-1)
            outs.append(y_t)
        return torch.stack(outs, 1).reshape(B, L, H * Dh)

    def _staledelta(self, qm, km, v, alpha, beta, m, c, pb, corr_fresh=False):
        """E6 eval-time chunk-refresh staleness (deployment-honest semantics):
        dims >= pb READ from a snapshot published every c steps (decay-compensated
        by running gate product G). Dims < pb stay fully fresh/sequential.
        pb=0 = ALL state stale = Config B ('uniform chunking + fresh conv').
        corr_fresh: delta correction uses the exact sequential state (honest
        Config B — PNM replays the chunk exactly at the boundary, only READOUTS
        are stale). corr_fresh=False: correction's stale dims also use the
        snapshot (honest for pb>0, where GPU forms the write w with only stale
        cold info and PNM applies w as received). c=1,pb=8 ~= pipedelta."""
        B, L, H, Dh = v.shape
        pb = min(pb, m)
        S = v.new_zeros(B, H, Dh, m)
        Snap = S.clone()
        G = v.new_ones(B, H)                      # prod of alphas since snapshot
        outs = []
        for t in range(L):
            if t % c == 0:
                Snap = S.clone(); G = v.new_ones(B, H)
            k_t = km[:, t]; q_t = qm[:, t]; v_t = v[:, t]
            a_t = alpha[:, t]; b_t = beta[:, t]
            if corr_fresh:                        # exact sequential correction
                Sk = (S * k_t.unsqueeze(2)).sum(-1)
            else:                                 # fresh hot + stale cold corr
                Sk = (S[..., :pb] * k_t[..., :pb].unsqueeze(2)).sum(-1) if pb > 0 else 0.0
                if m > pb:
                    Sk = Sk + G.unsqueeze(-1) * (
                        Snap[..., pb:] * k_t[..., pb:].unsqueeze(2)).sum(-1)
            write = b_t[..., None] * (v_t - a_t.unsqueeze(-1) * Sk)
            if self.tier_w and pb > 0 and m > pb and not corr_fresh:
                # v4 tier-local writes: cold columns get the PNM-exact write
                # (PNM replays with GPU-shipped fresh r_hot + its own exact cold
                # state); only hot columns use the stale-corrected write.
                Sk_ex = (S[..., :pb] * k_t[..., :pb].unsqueeze(2)).sum(-1) \
                    + (S[..., pb:] * k_t[..., pb:].unsqueeze(2)).sum(-1)
                w_ex = b_t[..., None] * (v_t - a_t.unsqueeze(-1) * Sk_ex)
                S = torch.cat([
                    a_t[..., None, None] * S[..., :pb]
                    + write.unsqueeze(-1) * k_t[..., :pb].unsqueeze(2),
                    a_t[..., None, None] * S[..., pb:]
                    + w_ex.unsqueeze(-1) * k_t[..., pb:].unsqueeze(2)], dim=-1)
            else:
                S = a_t[..., None, None] * S + write.unsqueeze(-1) * k_t.unsqueeze(2)
            y_t = (S[..., :pb] * q_t[..., :pb].unsqueeze(2)).sum(-1) if pb > 0 else 0.0
            if m > pb:                            # stale readout (pre-update)
                y_t = y_t + (a_t * G).unsqueeze(-1) * (
                    Snap[..., pb:] * q_t[..., pb:].unsqueeze(2)).sum(-1)
            G = G * a_t
            outs.append(y_t)
        return torch.stack(outs, 1).reshape(B, L, H * Dh)

    def _nesteddelta(self, qm, km, v, alpha, beta, m):
        """Hierarchical residual delta: each nested block fits the residual not
        explained by lower blocks -> leading blocks are independent of higher
        ones (structurally tax-free nesting for the coupled delta rule)."""
        B, L, H, _ = v.shape
        bounds = [b for b in self.grans if b <= m]
        if bounds[-1] != m: bounds.append(m)
        spans = list(zip([0] + bounds[:-1], bounds))       # [(0,2),(2,4),(4,8),...]
        Sb = [v.new_zeros(B, H, self.Dh, e - s) for s, e in spans]
        outs = []
        for t in range(L):
            a_t = alpha[:, t]; b_t = beta[:, t]; v_t = v[:, t]
            residual = v_t
            y_t = 0.0
            for j, (s, e) in enumerate(spans):
                Kj = km[:, t, :, s:e]                       # (B,H,span)
                Qj = qm[:, t, :, s:e]
                read = (Sb[j] * Kj.unsqueeze(2)).sum(-1)    # readout of block j (B,H,Dh)
                write = b_t[..., None] * (residual - a_t.unsqueeze(-1) * read)
                Sb[j] = a_t[..., None, None] * Sb[j] + write.unsqueeze(-1) * Kj.unsqueeze(2)
                residual = residual - (Sb[j] * Kj.unsqueeze(2)).sum(-1)  # pass down
                y_t = y_t + (Sb[j] * Qj.unsqueeze(2)).sum(-1)
            outs.append(y_t)
        return torch.stack(outs, 1).reshape(B, L, H * self.Dh)

class NestedDeltaLM(nn.Module):
    def __init__(self, vocab, d_model=128, n_layers=2, n_heads=2, head_dim=32,
                 mode="additive", max_len=512, pipe_b=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)          # absolute position
        self.layers = nn.ModuleList(
            [NestedGatedDelta(d_model, n_heads, head_dim, mode, pipe_b=pipe_b)
             for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.emb.weight  # tie
        self.head_dim = head_dim

    def forward(self, ids, width, stale_c=0, stale_pb=8):
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device)
        x = self.emb(ids) + self.pos(pos)[None]
        for lyr in self.layers:
            x = lyr(x, width, stale_c=stale_c, stale_pb=stale_pb)
        return self.head(self.norm_f(x))

# --------------------------- train / eval ---------------------------
def recall_at(model, D, nq, nk, nv, widths, device, gen, batch=512,
              stale_c=0, stale_pb=8):
    model.eval()
    inp, tgt = DATA_FN(batch, D, nq, nk, nv, device, gen)
    res = {}
    with torch.no_grad():
        for w in widths:
            logits = model(inp, w, stale_c=stale_c, stale_pb=stale_pb)
            mask = tgt != -100
            pred = logits.argmax(-1)
            acc = (pred[mask] == tgt[mask]).float().mean().item()
            res[w] = acc
    model.train()
    return res

AGE_BINS = [(1, 4), (5, 8), (9, 16), (17, 32), (33, 10 ** 6)]

def recall_by_age(model, D, nq, nk, nv, w, device, gen, batch=1024,
                  stale_c=0, stale_pb=8):
    """Accuracy bucketed by query-target age (write->query distance)."""
    model.eval()
    inp, tgt, age = DATA_FN(batch, D, nq, nk, nv, device, gen, return_age=True)
    with torch.no_grad():
        logits = model(inp, w, stale_c=stale_c, stale_pb=stale_pb)
    pred = logits.argmax(-1)
    ok = (pred == tgt)
    res = {}
    for lo, hi in AGE_BINS:
        mask = (tgt != -100) & (age >= lo) & (age <= hi)
        res[(lo, hi)] = ok[mask].float().mean().item() if mask.any() else float("nan")
    model.train()
    return res

def stale_sweep(model, Ds, widths, args, device, gen, cs=(1, 2, 4, 8, 16),
                pbs=(8, 0), batch=1024):
    """E6: recall under chunk-refresh staleness.
    pb=8 (a): cold-only stale, GPU hot fresh (our design).
    pb=0 (c1): ALL state stale, conv fresh, exact-corr = honest Config B
    ('uniform chunking + fresh conv' — the strongest rival deployment).
    Age-resolved rows attribute any drop: conv covers age<=4 only, so a
    (c1) hole at age in (4, c] means fresh hot state is what buys large c."""
    nk, nv, nq = args.n_keys, args.n_vals, args.n_query
    for lyr in model.layers:                   # v4 flag (affects pb>0 only)
        lyr.tier_w = getattr(args, "stale_tier_w", False)
    wmax = max(widths)
    for pb in pbs:
        scope = (f"(a) cold-only stale (dims>{pb}, hot fresh)" if pb > 0
                 else "(c1) ALL state stale, conv fresh, exact-corr [Config B]")
        print(f"\n===== E6 STALE SWEEP — {scope} =====", flush=True)
        for c in cs:
            print(f"--- stale c={c} ---")
            grid = {D: recall_at(model, D, nq, nk, nv, widths, device, gen,
                                 batch=batch, stale_c=c, stale_pb=pb) for D in Ds}
            print_grid(grid, Ds, widths)
            for D in (16, 32):
                if D in Ds:
                    ab = recall_by_age(model, D, nq, nk, nv, wmax, device, gen,
                                       batch=batch, stale_c=c, stale_pb=pb)
                    print(f"    age(D={D},k={wmax}): " + "  ".join(
                        f"{lo}-{hi if hi < 10**6 else '+'}:{v:.2f}"
                        for (lo, hi), v in ab.items()), flush=True)

def train_one(D, widths, args, device, nested=True, fixed_w=None, tag=""):
    nk, nv, nq = args.n_keys, args.n_vals, args.n_query
    vocab = 1 + nk + nv
    gen = torch.Generator(device=device); gen.manual_seed(args.seed)
    model = NestedDeltaLM(vocab, args.d_model, args.n_layers, args.n_heads,
                          args.head_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    train_widths = widths if nested else [fixed_w]
    t0 = time.time()
    for step in range(args.steps):
        inp, tgt = make_mqar(args.batch, D, nq, nk, nv, device, gen)
        opt.zero_grad()
        loss = 0.0
        for w in train_widths:                       # Matryoshka: sum over widths
            logits = model(inp, w)
            loss = loss + F.cross_entropy(
                logits.view(-1, vocab), tgt.view(-1), ignore_index=-100)
        loss = loss / len(train_widths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.log_every == 0:
            r = recall_at(model, D, nq, nk, nv, widths, device, gen)
            rc = " ".join(f"k{w}:{r[w]:.2f}" for w in widths)
            print(f"[{tag} D={D}] step {step+1:4d} loss {loss.item():.3f} | {rc}",
                  flush=True)
    r = recall_at(model, D, nq, nk, nv, widths, device, gen, batch=1024)
    print(f"[{tag} D={D}] FINAL ({time.time()-t0:.0f}s): "
          + " ".join(f"k{w}:{r[w]:.3f}" for w in widths), flush=True)
    return r

def eval_grid(model, Ds, widths, args, device, gen, batch=1024):
    nk, nv, nq = args.n_keys, args.n_vals, args.n_query
    grid = {}
    for D in Ds:
        grid[D] = recall_at(model, D, nq, nk, nv, widths, device, gen, batch=batch)
    return grid

def print_grid(grid, Ds, widths):
    print("D\\k  " + " ".join(f"{w:>6d}" for w in widths))
    for D in Ds:
        print(f"{D:<4d} " + " ".join(f"{grid[D][w]:6.3f}" for w in widths))

def train_mixed(Ds, widths, args, device, nested=True, fixed_w=None, tag=""):
    """One model trained on a MIX of D (sample D per step). If nested, sum loss
    over all widths (Matryoshka); else train only at fixed_w."""
    nk, nv, nq = args.n_keys, args.n_vals, args.n_query
    vocab = 1 + nk + nv
    gen = torch.Generator(device=device); gen.manual_seed(args.seed)
    model = NestedDeltaLM(vocab, args.d_model, args.n_layers, args.n_heads,
                          args.head_dim, mode=args.mode, pipe_b=args.pipe_b).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    train_widths = widths if nested else [fixed_w]
    Ds_t = torch.tensor(Ds)
    t0 = time.time()
    for step in range(args.steps):
        D = Ds[torch.randint(len(Ds), (1,), generator=gen, device=device).item()]
        inp, tgt = DATA_FN(args.batch, D, nq, nk, nv, device, gen)
        opt.zero_grad()
        loss = 0.0
        wsum = 0.0
        for w in train_widths:
            cw = float(w) ** args.loss_pow          # loss_pow<0 emphasizes small widths
            logits = model(inp, w)
            loss = loss + cw * F.cross_entropy(logits.view(-1, vocab), tgt.view(-1),
                                               ignore_index=-100)
            wsum += cw
        loss = loss / wsum
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.log_every == 0:
            g = eval_grid(model, Ds, widths, args, device, gen, batch=256)
            print(f"[{tag}] step {step+1:4d} loss {loss.item():.3f}", flush=True)
            print_grid(g, Ds, widths)
    g = eval_grid(model, Ds, widths, args, device, gen, batch=1024)
    print(f"[{tag}] FINAL ({time.time()-t0:.0f}s):", flush=True)
    print_grid(g, Ds, widths)
    return model, g

def adapt_rotation(model, Ds, widths, args, device, tag="e8-rot"):
    """E8: freeze pretrained backbone; train ONLY per-layer key-space rotations
    (identity-init) with the Matryoshka multi-width objective (+ soft
    orthogonality penalty). Tests whether nesting is a cheap RETROFIT."""
    nk, nv, nq = args.n_keys, args.n_vals, args.n_query
    vocab = 1 + nk + nv
    gen = torch.Generator(device=device); gen.manual_seed(args.seed + 1)
    for p in model.parameters():
        p.requires_grad_(False)
    rots = []
    for lyr in model.layers:
        lyr.add_rotation()
        rots.append(lyr.rot)
    opt = torch.optim.AdamW(rots, lr=args.lr_rot, weight_decay=0.0)
    eye = torch.eye(model.head_dim, device=device)
    t0 = time.time()
    for step in range(args.adapt_steps):
        D = Ds[torch.randint(len(Ds), (1,), generator=gen, device=device).item()]
        inp, tgt = DATA_FN(args.batch, D, nq, nk, nv, device, gen)
        opt.zero_grad()
        loss = 0.0
        for w in widths:
            logits = model(inp, w)
            loss = loss + F.cross_entropy(logits.view(-1, vocab), tgt.view(-1),
                                          ignore_index=-100)
        loss = loss / len(widths)
        loss = loss + args.orth_lam * sum(((R.T @ R - eye) ** 2).sum() for R in rots)
        loss.backward()
        opt.step()
        if (step + 1) % args.log_every == 0:
            g = eval_grid(model, Ds, widths, args, device, gen, batch=256)
            print(f"[{tag}] step {step+1:4d} loss {loss.item():.3f}", flush=True)
            print_grid(g, Ds, widths)
    g = eval_grid(model, Ds, widths, args, device, gen, batch=1024)
    print(f"[{tag}] FINAL ({time.time()-t0:.0f}s):", flush=True)
    print_grid(g, Ds, widths)
    with torch.no_grad():
        for i, R in enumerate(rots):
            dev = ((R.T @ R - eye) ** 2).sum().sqrt().item()
            print(f"  layer{i} ||R^T R - I||_F = {dev:.3f}")
    return g

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=1)
    p.add_argument("--head_dim", type=int, default=32)
    p.add_argument("--mode", choices=["additive", "delta", "nesteddelta", "blockdelta",
                                      "pipedelta"], default="additive")
    p.add_argument("--pipe_b", type=int, default=8,
                   help="hot/cold boundary for pipedelta (cold dims read pre-update)")
    p.add_argument("--n_keys", type=int, default=128)
    p.add_argument("--n_vals", type=int, default=64)
    p.add_argument("--n_query", type=int, default=16)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--loss_pow", type=float, default=0.0,
                   help="per-width loss weight = width**loss_pow (neg emphasizes small widths)")
    p.add_argument("--Ds", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--widths", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--tax", action="store_true", help="also train fixed-width baselines")
    p.add_argument("--fixed_only", type=int, default=0,
                   help="train ONLY a single fixed-width baseline at this width (for parallel tax)")
    p.add_argument("--task", choices=["mqar", "imqar"], default="mqar")
    p.add_argument("--e8", action="store_true",
                   help="E8: pretrain fixed-width -> freeze -> rotation-only Matryoshka FT")
    p.add_argument("--adapt_steps", type=int, default=2000)
    p.add_argument("--lr_rot", type=float, default=1e-2)
    p.add_argument("--orth_lam", type=float, default=0.01)
    p.add_argument("--stale_tier_w", action="store_true",
                   help="v4: cold columns updated with PNM-exact write during stale eval")
    p.add_argument("--eval_stale", action="store_true",
                   help="after training, run E6 staleness sweep (chunk-refresh)")
    args = p.parse_args()
    global DATA_FN
    DATA_FN = make_imqar if args.task == "imqar" else make_mqar
    NestedGatedDelta.TIER_W_DEFAULT = args.stale_tier_w
    device = "cuda"
    widths = [w for w in args.widths if w <= args.head_dim]
    torch.manual_seed(args.seed)
    print(f"device={device} mode={args.mode} widths={widths} head_dim={args.head_dim} "
          f"layers={args.n_layers} heads={args.n_heads} Ds={args.Ds}")

    gen_eval = torch.Generator(device=device); gen_eval.manual_seed(args.seed + 777)

    if args.e8:
        wmax = max(widths)
        print(f"\n=== E8: 'pretrained' proxy = fixed-{wmax} model ===")
        model, _ = train_mixed(args.Ds, [wmax], args, device, nested=False,
                               fixed_w=wmax, tag=f"pretrain{wmax}")
        print("\n--- BEFORE adaptation: pretrained model truncated to each width ---")
        print_grid(eval_grid(model, args.Ds, widths, args, device, gen_eval), args.Ds, widths)
        print("\n--- rotation-only Matryoshka FT (backbone frozen) ---")
        adapt_rotation(model, args.Ds, widths, args, device)
        if args.eval_stale:                    # NAIL B: does the retrofit also
            # reproduce STALENESS TOLERANCE (property Y), not just ordering (X)?
            stale_sweep(model, args.Ds, [w for w in widths if w > 8] or widths,
                        args, device, gen_eval)
        return

    if args.fixed_only:
        w = args.fixed_only
        print(f"\n=== FIXED-ONLY baseline width={w} ({args.mode}) ===")
        model, _ = train_mixed(args.Ds, [w], args, device, nested=False,
                               fixed_w=w, tag=f"fixed{w}")
        if args.eval_stale:
            stale_sweep(model, args.Ds, [w], args, device, gen_eval)
        return

    print("\n=== H1/H2: ONE nested model, recall(width, D) grid ===")
    model, table = train_mixed(args.Ds, widths, args, device, nested=True, tag="nested")
    if args.eval_stale:
        stale_sweep(model, args.Ds, [w for w in widths if w > 8] or widths,
                    args, device, gen_eval)

    if args.tax:
        print("\n=== H3: Matryoshka tax vs fixed-width models ===")
        fixed = {}
        for w in widths:
            _, g = train_mixed(args.Ds, [w], args, device, nested=False,
                               fixed_w=w, tag=f"fixed{w}")
            fixed[w] = g
        print("\n===== TAX (fixed - nested), per (D,k) =====")
        print("D\\k  " + " ".join(f"{w:>6d}" for w in widths))
        for D in args.Ds:
            print(f"{D:<4d} " + " ".join(
                f"{fixed[w][D][w]-table[D][w]:+6.3f}" for w in widths))

if __name__ == "__main__":
    main()
