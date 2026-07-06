from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
tok = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-Nano-9B-v2")
def tokenize_split(split):
    texts = [t for t in ds[split]["text"] if t.strip()]
    out = []
    CH = 2000
    for i in range(0, len(texts), CH):
        ids = tok("\n".join(texts[i:i+CH]), return_tensors="np").input_ids[0]
        out.append(ids.astype(np.int32))
        if i % 20000 == 0: print(split, i, "/", len(texts), flush=True)
    return np.concatenate(out)
val = tokenize_split("validation"); np.save("wt103_val_nemo.npy", val); print("val tokens:", len(val), flush=True)
train = tokenize_split("train"); np.save("wt103_train_nemo.npy", train); print("train tokens:", len(train), flush=True)
print("DONE", flush=True)
