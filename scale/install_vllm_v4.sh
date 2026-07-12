#!/bin/bash
# [nested_ssm] Install R-rotation + v4 tiering into a vLLM env (idempotent).
# Usage: bash install_vllm_v4.sh [/path/to/vllm_env]  (default ~/vllm_env)
set -e
ENVROOT=${1:-$HOME/vllm_env}
PKG=$(echo $ENVROOT/lib/python3*/site-packages/vllm)
REPO=$(cd "$(dirname "$0")" && pwd)

cp "$REPO/vllm_v4_patch.py" "$PKG/model_executor/layers/mamba/v4_nested.py"

"$ENVROOT/bin/python3" - "$PKG" <<'EOF'
import sys, re
pkg = sys.argv[1]

def patch(path, anchor, insert, marker="[nested_ssm]"):
    src = open(path).read()
    if insert.strip() in src:
        print(f"  already patched: {path.split('/')[-1]} ({insert.strip().splitlines()[0][:50]})")
        return
    assert anchor in src, f"anchor not found in {path}:\n{anchor}"
    open(path, "w").write(src.replace(anchor, anchor + insert, 1))
    print(f"  patched: {path.split('/')[-1]}")

mm = f"{pkg}/model_executor/layers/mamba/mamba_mixer2.py"
nh = f"{pkg}/model_executor/models/nemotron_h.py"

patch(mm,
"""        self.is_blackwell = current_platform.is_device_capability_family(100)
""",
"""
        # [nested_ssm] R rotation + v4 tiering hooks (set by nemotron_h apply_ckpt)
        self.v4R = None
        self.v4cfg = None
        self._v4buf = None
""")

patch(mm,
"""            hidden_states_p, B_p, C_p = self.split_hidden_states_B_C_fn(
                hidden_states_B_C_p
            )
""",
"""            if self.v4R is not None:  # [nested_ssm] rotate B/C per group
                from vllm.model_executor.layers.mamba import v4_nested as _v4n
                B_p = _v4n.rot_flat(B_p, self.v4R)
                C_p = _v4n.rot_flat(C_p, self.v4R)
""")

patch(mm,
"""                assert state_indices_tensor_p is not None
                ssm_state[state_indices_tensor_p] = varlen_states
""",
"""                if self.v4cfg is not None:  # [nested_ssm] cold snapshot @ t=0
                    from vllm.model_executor.layers.mamba import v4_nested as _v4n
                    _v4n.prefill_snapshot(self, ssm_state, state_indices_tensor_p)
""")

patch(mm,
"""            hidden_states_d, B_d, C_d = self.split_hidden_states_B_C_fn(
                hidden_states_B_C_d
            )
""",
"""            if self.v4R is not None:  # [nested_ssm] rotate B/C per group
                from vllm.model_executor.layers.mamba import v4_nested as _v4n
                B_d = _v4n.rot_flat(B_d, self.v4R)
                C_d = _v4n.rot_flat(C_d, self.v4R)
            _v4_dt_raw = dt_d  # [nested_ssm] pre-softplus (b,H), for cold decay log
""")

patch(mm,
"""                cu_seqlens=query_start_loc_d,
                is_blackwell=self.is_blackwell,
            )
""",
"""            if self.v4cfg is not None:  # [nested_ssm] tiered hot+cold readout
                assert self.num_spec == 0, "v4 tiering: spec decode unsupported"
                from vllm.model_executor.layers.mamba import v4_nested as _v4n
                _v4n.decode_readout(
                    self, ssm_state,
                    preallocated_ssm_out_d.view(num_decode_tokens, -1, self.head_dim),
                    hidden_states_d, _v4_dt_raw, C_d,
                    state_indices_tensor_d_output,
                )
""")

EOF

# nemotron_h return-line rewrite (replace, not append)
"$ENVROOT/bin/python3" - "$PKG" <<'EOF'
import sys
pkg = sys.argv[1]
nh = f"{pkg}/model_executor/models/nemotron_h.py"
src = open(nh).read()
old = """        loader = AutoWeightsLoader(self, skip_prefixes=["mtp"])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)"""
new = """        loader = AutoWeightsLoader(self, skip_prefixes=["mtp"])
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        from vllm.model_executor.layers.mamba import v4_nested as _v4n
        _v4n.apply_ckpt(self)  # [nested_ssm] no-op unless NESTED_SSM_CKPT is set
        return loaded"""
if new in src:
    print("  nemotron_h already patched")
elif old in src:
    open(nh, "w").write(src.replace(old, new, 1))
    print("  patched: nemotron_h.py")
else:
    raise AssertionError("nemotron_h anchor not found")
EOF

"$ENVROOT/bin/python3" -m py_compile \
  "$PKG/model_executor/layers/mamba/mamba_mixer2.py" \
  "$PKG/model_executor/layers/mamba/v4_nested.py" \
  "$PKG/model_executor/models/nemotron_h.py"
echo "INSTALL OK"
