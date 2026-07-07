"""Sample a diverse web corpus (fineweb) with the Nemotron tokenizer.

Night finding (2026-07-08): continuing R/decay FT on wikitext-only improves
in-domain v4 metrics (ppl gap halved, needle 1.00) but drifts DOWNSTREAM acc
under v4 by ~1pt — the rotation overfits its training domain.  Remedy: mix
diverse web text.  Output: fineweb_train_nemo.npy + mixed_train_nemo.npy
(50/50 concat with wt103)."""
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

TARGET = 24_000_000
tok = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-Nano-9B-v2")
eos = tok.eos_token_id if tok.eos_token_id is not None else 0
ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                  split="train", streaming=True)
toks = []
for i, ex in enumerate(ds):
    toks.extend(tok(ex["text"]).input_ids)
    toks.append(eos)
    if i % 2000 == 0:
        print(f"docs {i} tokens {len(toks):,}", flush=True)
    if len(toks) >= TARGET:
        break
arr = np.array(toks[:TARGET], dtype=np.uint32)
np.save("fineweb_train_nemo.npy", arr)
print(f"saved fineweb_train_nemo.npy {arr.shape}", flush=True)
wt = np.load("wt103_train_nemo.npy", mmap_mode="r")
wt_slice = np.asarray(wt[:TARGET], dtype=np.uint32)      # 50/50 mix
np.save("mixed_train_nemo.npy", np.concatenate([wt_slice, arr]))
print(f"saved mixed_train_nemo.npy {len(wt_slice) + len(arr):,} tokens", flush=True)
