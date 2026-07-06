from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("fla-hub/gla-340M-15B")
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train",
                  streaming=True)
out, total, TARGET = [], 0, 1_500_000_000        # ~1.5B tokens
buf = []
for ex in ds:
    buf.append(ex["text"])
    if len(buf) >= 500:
        ids = tok("\n".join(buf), return_tensors="np").input_ids[0].astype(np.int32)
        out.append(ids); total += len(ids); buf = []
        if total // 100_000_000 != (total - len(ids)) // 100_000_000:
            print(f"{total/1e6:.0f}M tokens", flush=True)
        if total >= TARGET: break
arr = np.concatenate(out)
np.save("fineweb_1p5b.npy", arr)
print("DONE", len(arr), flush=True)
