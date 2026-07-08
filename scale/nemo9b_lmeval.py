"""T12-4: downstream accuracy of the 9B retrofit under three deployment configs
via lm-eval-harness (0.4.12).

Configs (--config):
  orig        : untouched public Nemotron-9B (no rotation, native forward)  -> accuracy ceiling
  retro_fresh : ActRotMask+R patched, full width 128, native torch_forward  -> cost of retrofit
  retro_v4    : ActRotMask+R patched, naive_mixer_forward mode=v4 (c/pb)     -> cost of tiered v4 deploy

The rotation basis R comes from --ckpt (default the tiering-aware v4-aware retrofit).
Right-padded loglikelihood batches are safe for the causal SSM (pads sit after the
scored positions), so batch_size>1 is fine for all configs.

Run with ~/nemo_env/bin/python3 (native transformers 5.13 NemotronH)."""
import argparse, json
import numpy as np
import torch
from nemotron_retrofit import ActRotMask, set_width
from nemo9b_eval import naive_mixer_forward, CFG

MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"


def build_model(config, ckpt, c, pb, device, lag=0, cold_bf16=0, warm=0, cold_fp8=0):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device)
    model.config.use_cache = False
    if config == "orig":
        model.eval()
        return model
    mixers = [m for m in model.modules() if type(m).__name__ == "NemotronHMamba2Mixer"]
    for m in mixers:
        m.act = ActRotMask(m.act, m.intermediate_size, m.n_groups, m.ssm_state_size).to(device)
    saved = torch.load(ckpt)
    for i, m in enumerate(mixers):
        m.act.R.data.copy_(saved[i].to(device).float())
    if "decay" in saved:                                  # tune_decay ckpts
        for i, m in enumerate(mixers):
            m.A_log.data.copy_(saved["decay"]["A_log"][i].to(device))
            m.dt_bias.data.copy_(saved["decay"]["dt_bias"][i].to(device))
    set_width(model, 128)
    if config in ("retro_v4", "naive_fresh"):
        for m in mixers:
            m.forward = naive_mixer_forward.__get__(m)
        if config == "retro_v4":
            CFG.update(mode="v4", c=c, pb=pb, lag=lag, cold_bf16=cold_bf16,
                       warm=warm, cold_fp8=cold_fp8)
        else:               # fresh through the SAME naive engine (engine-consistent
            CFG.update(mode="fresh")   # comparison; also low peak memory for gen)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    choices=["orig", "retro_fresh", "retro_v4", "naive_fresh"])
    ap.add_argument("--ckpt", default="nemo9b_rot_v4aware.pt")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--c", type=int, default=16)
    ap.add_argument("--pb", type=int, default=32)
    ap.add_argument("--lag", type=int, default=0,
                    help="1 = async-flush semantics: readout snapshot lags one chunk (age (c,2c])")
    ap.add_argument("--cold_bf16", type=int, default=0,
                    help="1 = cold snapshot stored bf16 (halves cold readout bytes)")
    ap.add_argument("--warm", type=int, default=0,
                    help="first W tokens run fresh before v4 kicks in (one-time "
                         "cold-tier ship; tiering targets long sequences)")
    ap.add_argument("--cold_fp8", type=int, default=0,
                    help="1 = scaled-fp8 cold snapshot (asymmetric precision license)")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--tasks", nargs="+",
                    default=["lambada_openai", "piqa", "hellaswag",
                             "arc_easy", "arc_challenge", "winogrande"])
    args = ap.parse_args()
    tag = args.tag or f"nemo9b-{args.config}"
    device = "cuda"

    from transformers import AutoTokenizer
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = build_model(args.config, args.ckpt, args.c, args.pb, device,
                        lag=args.lag, cold_bf16=args.cold_bf16, warm=args.warm,
                        cold_fp8=args.cold_fp8)
    print(f"[{tag}] model built (config={args.config} ckpt={args.ckpt} c={args.c} "
          f"pb={args.pb} lag={args.lag} cold_bf16={args.cold_bf16} "
          f"bs={args.bs} limit={args.limit})", flush=True)

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.bs)
    results = lm_eval.simple_evaluate(model=lm, tasks=args.tasks, limit=args.limit,
                                      bootstrap_iters=0)
    out = {}
    for task, m in results["results"].items():
        metrics = {k.split(",")[0]: v for k, v in m.items()
                   if isinstance(v, (int, float)) and "stderr" not in k}
        out[task] = metrics
        print(f"[{tag}] {task}: " +
              " ".join(f"{k}={v}" for k, v in sorted(metrics.items())), flush=True)
    json.dump({"config": args.config, "ckpt": args.ckpt, "c": args.c, "pb": args.pb,
               "lag": args.lag, "cold_bf16": args.cold_bf16, "warm": args.warm,
               "limit": args.limit, "results": out},
              open(f"nemo9b_lmeval_{tag}.json", "w"), indent=1)
    print(f"[{tag}] DONE -> nemo9b_lmeval_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
