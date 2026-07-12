"""v4 tiered execution in the NATIVE NemotronH engine's DECODE path (cache).

Motivation: nemo9b_eval.naive_mixer_forward reproduces teacher-forced ppl but
DEGENERATES in autoregressive generation, crushing the recall-intensive suite
(fda 0.013 vs native 0.31).  The native torch_forward generates coherently.  So
we patch v4 into the native decode branch — which is also the real deployment
path (prefill fresh, decode tiered == our warmup=prompt_len semantics).

Design (deployment-honest):
  * prefill (or any non-decode call): native torch_forward UNCHANGED; afterwards
    stash a cold snapshot of the recurrent state on the module.
  * decode (has_previous_state & seq_len==1): mirror the native decode branch
    exactly, EXCEPT the readout: y = hot(fresh) + cold(stale snapshot,
    decay-compensated); refresh the snapshot every c decode steps.

Gate: pb=128 (all-hot) must reproduce native fresh scores exactly.
NOTE: the native decode branch applies self.act on a 2D tensor, so an ActRotMask
wrapper would silently SKIP rotation there (dim check) — for rotated ckpts the
act call below is made 3D explicitly so R applies in decode too.
Attach with install(model, pb=32, c=16, cold_bf16=1)."""
import torch
import torch.nn.functional as F
from transformers.models.nemotron_h import modeling_nemotron_h as M

_orig_forward = M.NemotronHMamba2Mixer.torch_forward


def _v4_decode(self, input_states, cache_params, attention_mask):
    """Mirror of native torch_forward decode branch; divergence marked (*)."""
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype
    projected_states = self.in_proj(input_states)
    d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size
             - 2 * self.n_groups * self.ssm_state_size - self.num_heads) // 2
    _, _, gate, hidden_states, dt = projected_states.split(
        [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1)
    hidden_states = hidden_states.transpose(1, 2)
    # conv (native decode path)
    conv_states = cache_params.update_conv_state(hidden_states, self.layer_idx)
    hidden_states = torch.sum(conv_states * self.conv1d.weight[:, 0, :], dim=-1)
    if self.use_conv_bias:
        hidden_states += self.conv1d.bias
    # (*) act called 3D so ActRotMask rotation applies in decode as well
    hidden_states = self.act(hidden_states[:, None, :]).to(dtype)
    hidden_states, B, C = torch.split(
        hidden_states.squeeze(1),
        [self.intermediate_size, self.n_groups * self.ssm_state_size,
         self.n_groups * self.ssm_state_size], dim=-1)
    A = -torch.exp(self.A_log.float())                       # [H]
    dt = dt[:, 0, :] if dt.ndim == 3 else dt
    dt = dt.transpose(0, 1)[None] if False else dt           # [b,H]
    dt = dt[:, :, None].expand(batch_size, self.num_heads, self.head_dim)
    dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)
    dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
    dt = torch.clamp(dt, self.time_step_min)                 # [b,H,P]
    A_e = A[..., None, None].expand(self.num_heads, self.head_dim,
                                    self.ssm_state_size).to(torch.float32)
    dA = torch.exp(dt[..., None] * A_e)                      # [b,H,P,N]
    B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
    B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups,
                 B.shape[-1]).contiguous().reshape(batch_size, -1, B.shape[-1])
    dB = dt[..., None] * B[..., None, :]                     # [b,H,P,N]
    hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
    dBx = dB * hidden_states[..., None]
    ssm_states = cache_params.layers[self.layer_idx].recurrent_states.clone()
    ssm_states = ssm_states * dA + dBx
    ssm_states = cache_params.update_recurrent_state(ssm_states, self.layer_idx)
    C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
    C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups,
                 C.shape[-1]).contiguous().reshape(batch_size, -1, C.shape[-1])
    # ---- (*) v4 readout: hot fresh + cold stale (decay-compensated) ----
    cfg = self.v4cfg
    pb, c = cfg["pb"], cfg["c"]
    N = self.ssm_state_size
    Sf = ssm_states.to(torch.float32)                        # [b,H,P,N]
    Cf = C.to(torch.float32)                                 # [b,H,N]
    if pb >= N:
        y = torch.einsum('bhpn,bhn->bhp', Sf, Cf)
    else:
        st = self._v4state
        if cfg.get("corr"):
            # rank-c EXACT correction (additive/mamba2 only — the update term
            # is state-independent, so the writes missed since flush are
            # exactly the buffered rank-1 terms):
            #   y_cold = e^{G_t}(C·snap) + Σ_j e^{G_t-G_j}(C_cold·B_j)·Δx_j
            # with G inclusive of the current token (exact decay anchoring —
            # note the stale path below keeps its trained one-token-lag
            # semantics; do not unify).  Buffers are ~KB vs GB of state.
            st["glog"] = st["glog"] + dt * A[None, :, None]   # G_t inclusive
            g1 = st["glog"][:, :, 0]                          # [b,H] (const over P)
            # buffer THIS token's write BEFORE readout: y_t reads S_t which
            # already contains dBx_t — the j=t term (weight e^0) must be in
            # the correction sum or every token misses one rank-1 write.
            st["bufB"].append(B[..., pb:].float().clone())            # [b,H,Nc]
            st["bufX"].append((dt * hidden_states).float().clone())   # [b,H,P]
            st["bufG"].append(g1.clone())                             # [b,H]
            y_cold = torch.einsum('bhpn,bhn->bhp', st["snap"], Cf[..., pb:]) \
                * torch.exp(st["glog"])
            Bs = torch.stack(st["bufB"])                      # [j,b,H,Nc]
            Xs = torch.stack(st["bufX"])                      # [j,b,H,P]
            Gs = torch.stack(st["bufG"])                      # [j,b,H]
            s = torch.einsum('jbhn,bhn->jbh', Bs, Cf[..., pb:])
            w = torch.exp(g1[None] - Gs) * s                  # [j,b,H]
            y_cold = y_cold + torch.einsum('jbh,jbhp->bhp', w, Xs)
            y = torch.einsum('bhpn,bhn->bhp', Sf[..., :pb], Cf[..., :pb]) + y_cold
            st["t"] += 1
            if st["t"] % c == 0:                              # snapshot refresh
                snap = Sf[..., pb:]
                st["snap"] = (snap.to(torch.bfloat16).float()
                              if cfg.get("cold_bf16") else snap.clone())
                st["glog"] = torch.zeros_like(st["glog"])
                st["bufB"], st["bufX"], st["bufG"] = [], [], []
            D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
            y = (y + hidden_states * D).to(dtype)
            y = y.reshape(batch_size, -1)[:, None, ...]
            scan_output = self.norm(y, gate)
            return self.out_proj(scan_output.to(dtype))
        y = torch.einsum('bhpn,bhn->bhp', Sf[..., :pb], Cf[..., :pb]) \
            + torch.einsum('bhpn,bhn->bhp', st["snap"], Cf[..., pb:]) \
            * torch.exp(st["glog"])          # glog is per (b,H,P) here — no new axis
        st["t"] += 1
        if st["t"] % c == 0:                                 # snapshot refresh
            snap = Sf[..., pb:]
            st["snap"] = (snap.to(torch.bfloat16).float()
                          if cfg.get("cold_bf16") else snap.clone())
            st["glog"] = torch.zeros_like(st["glog"])
        else:                                                # decay since snapshot
            st["glog"] = st["glog"] + dt * A[None, :, None]  # [b,H,P]
    D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
    y = (y + hidden_states * D).to(dtype)
    y = y.reshape(batch_size, -1)[:, None, ...]
    scan_output = self.norm(y, gate)
    return self.out_proj(scan_output.to(dtype))


def _lean_prefill(self, input_states, cache_params, attention_mask):
    """Prefill via fused causal_conv1d + Triton chunk scan, R applied manually.

    HF torch_forward materializes O(T*H*P*N) broadcast intermediates (7.5GB+ on
    a 2k-token minerva prompt) — unusable on a GPU shared with another tenant.
    The Triton scan is memory-lean and exact; ActRotMask cannot fire on this
    path (silu is fused into the conv kernel), so B/C rotation is applied here
    with the same rotmask the wrapper uses."""
    from causal_conv1d import causal_conv1d_fn
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    batch_size, seq_len, _ = input_states.shape
    if attention_mask is not None and not torch.all(attention_mask == 1):
        input_states = (input_states * attention_mask[:, :, None]).to(input_states.dtype)
    projected_states = self.in_proj(input_states)
    d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size
             - 2 * self.n_groups * self.ssm_state_size - self.num_heads) // 2
    _, _, gate, hidden_states_B_C, time_step = projected_states.split(
        [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1)
    hidden_states_B_C = hidden_states_B_C.transpose(1, 2)
    if cache_params is not None:
        new_conv_state = F.pad(
            hidden_states_B_C, (self.conv_kernel_size - hidden_states_B_C.shape[-1], 0))
        cache_params.update_conv_state(new_conv_state, self.layer_idx)
    hidden_states_B_C = causal_conv1d_fn(
        x=hidden_states_B_C, weight=self.conv1d.weight.squeeze(1),
        bias=self.conv1d.bias, activation=self.activation).transpose(1, 2)
    gn = self.n_groups * self.ssm_state_size
    hidden_states, B, C = torch.split(
        hidden_states_B_C, [self.intermediate_size, gn, gn], dim=-1)
    if hasattr(self.act, "R"):                       # rotated ckpt
        B, C = self.act.rotmask(B), self.act.rotmask(C)
    A = -torch.exp(self.A_log.float())
    dt_limit_kwargs = ({} if getattr(self, "time_step_limit", None) is None
                       else {"dt_limit": self.time_step_limit})
    scan_output, ssm_state = mamba_chunk_scan_combined(
        hidden_states.view(batch_size, seq_len, -1, self.head_dim),
        time_step, A,
        B.view(batch_size, seq_len, self.n_groups, -1),
        C.view(batch_size, seq_len, self.n_groups, -1),
        chunk_size=self.chunk_size, D=self.D, z=None, seq_idx=None,
        return_final_states=True, dt_bias=self.dt_bias, dt_softplus=True,
        initial_states=None, **dt_limit_kwargs)
    if ssm_state is not None and cache_params is not None:
        cache_params.update_recurrent_state(ssm_state, self.layer_idx)
    scan_output = self.norm(scan_output.view(batch_size, seq_len, -1), gate)
    return self.out_proj(scan_output.to(self.out_proj.weight.dtype))


def _dispatch(self, input_states, cache_params=None, attention_mask=None):
    seq_len = input_states.shape[1]
    if (cache_params is not None and cache_params.has_previous_state(self.layer_idx)
            and seq_len == 1 and self._v4state is not None):
        return _v4_decode(self, input_states, cache_params, attention_mask)
    if (seq_len > 1 and cache_params is not None
            and not cache_params.has_previous_state(self.layer_idx)
            and self.v4cfg.get("lean_prefill", 1)):
        out = _lean_prefill(self, input_states, cache_params, attention_mask)
    else:
        out = _orig_forward(self, input_states, cache_params, attention_mask)
    if cache_params is not None:                             # end of prefill:
        ssm = cache_params.layers[self.layer_idx].recurrent_states.to(torch.float32)
        pb = self.v4cfg["pb"]
        snap = ssm[..., pb:]
        self._v4state = {
            "snap": (snap.to(torch.bfloat16).float()
                     if self.v4cfg.get("cold_bf16") else snap.clone()),
            "glog": torch.zeros(ssm.shape[0], ssm.shape[1], ssm.shape[2],
                                device=ssm.device),
            "t": 0, "bufB": [], "bufX": [], "bufG": []}
    return out


def install(model, pb=32, c=16, cold_bf16=1, corr=0):
    n = 0
    for m in model.modules():
        if type(m).__name__ == "NemotronHMamba2Mixer":
            m.v4cfg = dict(pb=pb, c=c, cold_bf16=cold_bf16, corr=corr)
            m._v4state = None
            m.torch_forward = _dispatch.__get__(m)
            n += 1
    return n
