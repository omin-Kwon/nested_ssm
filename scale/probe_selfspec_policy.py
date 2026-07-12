"""Overnight E1+E2: self-speculative run-length distribution + verification
trigger policy signals.

Per decode position t (teacher trajectory = full model, greedy):
  hit_t       argmax(hot-only logits) == argmax(full logits)   [same state]
  margin_t    top1-top2 of HOT logits (deployable, free)
  stamp_t     sum_l sum_h ||B_cold||*||dt x|| of THIS token     (deployable)
  dtA_t       mean decay increment (for pending-mass decay)
Offline: run-length distribution, AUC of signals vs miss events, policy sim
(fixed-k vs margin vs pending-mass vs combined) -> verify rate & waste.

Run: CUDA_VISIBLE_DEVICES=0 ~/nemo_env/bin/python3 probe_selfspec_policy.py
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nemotron_retrofit import ActRotMask
import v4_native_decode as V

MID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
HUB = "/NHNHOME/ARC/arclab/shared/hub"
CKPT = "nemo9b_rot_longcot.pt"
GEN = 300
PB = 32

tok = AutoTokenizer.from_pretrained(MID, cache_dir=HUB)
model = AutoModelForCausalLM.from_pretrained(
    MID, dtype=torch.bfloat16, cache_dir=HUB).to("cuda")
model.config.use_cache = True
V.M.is_fast_path_available = False
mixers = [m for m in model.modules() if type(m).__name__ == "NemotronHMamba2Mixer"]
for m in mixers:
    m.act = ActRotMask(m.act, m.intermediate_size, m.n_groups,
                       m.ssm_state_size).to("cuda")
saved = torch.load(CKPT)
for i, m in enumerate(mixers):
    m.act.R.data.copy_(saved[i].to("cuda").float())
    m.A_log.data.copy_(saved["decay"]["A_log"][i].to("cuda"))
    m.dt_bias.data.copy_(saved["decay"]["dt_bias"][i].to("cuda"))
model.eval()

# ---- spy: capture per-token per-layer stamp norms inside the decode path ----
SPY = {"stamp": 0.0, "dtA": 0.0, "n": 0}
_orig = V._v4_decode
def spy_decode(self, input_states, cache_params, attention_mask):
    # replicate the preamble cheaply to read B/dt/x norms (full path pb>=N)
    out = _orig(self, input_states, cache_params, attention_mask)
    return out
# instead of duplicating math, hook on ActRotMask output via a light wrapper:
class StampSpy(torch.nn.Module):
    def __init__(self, act, mixer):
        super().__init__()
        self.act, self.mixer = act, mixer
    def forward(self, x):
        y = self.act(x)
        if y.dim() == 3 and y.shape[-1] == self.act.conv_dim:
            gn = self.act.ng * self.act.ds
            B = y[..., self.act.inter:self.act.inter + gn].float() \
                .view(-1, self.act.ng, self.act.ds)
            bcold = B[..., PB:].norm(dim=-1).sum().item()      # sum over groups
            SPY["stamp"] += bcold
            SPY["n"] += 1
        return y
    def rotmask(self, x): return self.act.rotmask(x)
    @property
    def R(self): return self.act.R
    @property
    def conv_dim(self): return self.act.conv_dim
    @property
    def inter(self): return self.act.inter
    @property
    def ng(self): return self.act.ng
    @property
    def ds(self): return self.act.ds

for m in mixers:
    m.act = StampSpy(m.act, m)

def clone_cache(cache):
    import copy
    new = copy.copy(cache)
    new.layers = [copy.copy(l) for l in cache.layers]
    for l in new.layers:
        for k, v in list(vars(l).items()):
            if torch.is_tensor(v):
                setattr(l, k, v.clone())
            elif isinstance(v, (list, tuple)) and v and torch.is_tensor(v[0]):
                setattr(l, k, type(v)(t.clone() for t in v))
    return new

PROMPTS = [
    "Explain why the sky is blue in three sentences.",
    "A train travels 60 km/h for 2.5 hours, then 80 km/h for 1.5 hours. Total distance? Show your work.",
    "Write a short Python function that reverses a linked list.",
    "Summarize the causes of World War I in one paragraph.",
    "If x^2 - 5x + 6 = 0, find both roots and verify them.",
    "Describe how a refrigerator works thermodynamically.",
    "Translate to French: 'The weather is beautiful today, let us walk along the river.'",
    "What is the time complexity of quicksort in the worst case and why?",
]

records = []
with torch.no_grad():
    for pi, p in enumerate(PROMPTS):
        msgs = [{"role": "user", "content": p}]
        ids = tok.apply_chat_template(msgs, return_tensors="pt",
                                      add_generation_prompt=True)
        if not torch.is_tensor(ids):
            ids = ids["input_ids"]
        ids = ids.to("cuda")
        # full-model trajectory with per-step dual readout
        V.install(model, pb=128, c=1 << 30, cold_bf16=0)      # full fresh
        r = model(ids, use_cache=True)
        cache = r.past_key_values
        logits_full = r.logits[0, -1]
        tokens = []
        for t in range(GEN):
            nxt = logits_full.argmax().item()
            if nxt == tok.eos_token_id:
                break
            tokens.append(nxt)
            # hot-only logits from the SAME state (clone, coldoff step)
            c2 = clone_cache(cache)
            for m in mixers:
                m.v4cfg["coldoff"] = 1
                m.v4cfg["pb"] = PB
                if m._v4state is not None:
                    m._v4state["t"] = 0
            SPY["stamp"] = 0.0
            rh = model(torch.tensor([[nxt]], device="cuda"),
                       past_key_values=c2, use_cache=True)
            lh = rh.logits[0, -1].float()
            stamp = SPY["stamp"]
            for m in mixers:
                m.v4cfg["coldoff"] = 0
                m.v4cfg["pb"] = 128
            # full step (advances the real cache)
            r = model(torch.tensor([[nxt]], device="cuda"),
                      past_key_values=cache, use_cache=True)
            logits_prev = logits_full
            logits_full = r.logits[0, -1]
            lf = logits_full.float()
            top2h = lh.topk(2).values
            records.append(dict(
                prompt=pi, t=t,
                hit=int(lh.argmax().item() == lf.argmax().item()),
                margin=float(top2h[0] - top2h[1]),
                stamp=float(stamp),
            ))
        print(f"prompt {pi}: {len(tokens)} tokens, "
              f"alpha={sum(r_['hit'] for r_ in records if r_['prompt']==pi)/max(1,len([r_ for r_ in records if r_['prompt']==pi])):.3f}",
              flush=True)

json.dump(records, open("results/selfspec_policy_records.json", "w"))
# ---- offline analysis ----
import statistics
hits = [r["hit"] for r in records]
alpha = sum(hits) / len(hits)
runs, cur = [], 0
for h in hits:
    if h: cur += 1
    else: runs.append(cur); cur = 0
runs.append(cur)
print(f"\nALPHA={alpha:.3f}  runs: mean={statistics.mean(runs):.1f} "
      f"median={statistics.median(runs)} p90={sorted(runs)[int(.9*len(runs))]}")
miss = [r for r in records if not r["hit"]]
hit_ = [r for r in records if r["hit"]]
for sig in ("margin", "stamp"):
    mh = statistics.mean(x[sig] for x in hit_)
    mm = statistics.mean(x[sig] for x in miss) if miss else float("nan")
    # AUC via rank comparison (margin: low=risky; stamp: high=risky)
    import random
    random.seed(0)
    pairs = [(a[sig], b[sig]) for a in random.sample(miss, min(200, len(miss)))
             for b in random.sample(hit_, min(200, len(hit_)))]
    if sig == "margin":
        auc = sum(a < b for a, b in pairs) / len(pairs)
    else:
        auc = sum(a > b for a, b in pairs) / len(pairs)
    print(f"signal {sig}: mean(hit)={mh:.3f} mean(miss)={mm:.3f} AUC={auc:.3f}")
print("RECORDS SAVED -> results/selfspec_policy_records.json")
