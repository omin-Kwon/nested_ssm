"""v4 tiered decode on the FUSED kernel path — paper-grade speed measurement.

Baseline engine = HF fast path (causal_conv1d_update + selective_state_update),
i.e. how people actually run this model. Our v4 on the same engine:
  hot  : selective_state_update on a CONTIGUOUS hot-slice state (B,H,P,pb)
         — the same official fused kernel, just narrower
  cold : quantized snapshot readout — bf16 via cuBLAS bmm, fp8 via our verified
         Triton dequant-matvec (bench_v4_decode._fp8_matvec) — + exp(glog) comp
  flush: every c tokens, exact suffix-decay replay into the fp32 cache master
         (cuBLAS einsum, amortized /c), then re-quantize the snapshot
Prefill / multi-token: routed through torch_forward when the model is rotated
(ActRotMask applies R there — the validated accuracy path); raw/unrotated models
keep the native fused prefill. Decode applies R explicitly post-conv (the fused
conv kernel bypasses ActRotMask).
Gate: install(pb=128) must match raw fast-path semantics (all-hot, no cold).
ACCURACY-GRADE: with a rotated ckpt loaded (ActRotMask + R), this path is exact
v4 semantics — usable for both speed and accuracy runs. No rebuild ever needed
for changes here (Triton JIT-compiles; the pip build is 3rd-party kernels only).
"""
import torch
import torch.nn.functional as F
from transformers.models.nemotron_h import modeling_nemotron_h as M
from bench_v4_decode import fp8_cold_readout

_orig_forward = M.NemotronHMamba2Mixer.forward


def _v4_fused_decode(self, hidden_states, cache_params, attention_mask):
    from causal_conv1d import causal_conv1d_update
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    B_, _, _ = hidden_states.shape
    H, P, N, G = self.num_heads, self.head_dim, self.ssm_state_size, self.n_groups
    cfg = self.v4cfg; pb = cfg["pb"]; c = cfg["c"]; Nc = N - pb
    st = self._v4state
    conv_state = cache_params.layers[self.layer_idx].conv_states

    proj = self.in_proj(hidden_states.squeeze(1))
    d_mlp = (proj.shape[-1] - 2 * self.intermediate_size
             - 2 * G * N - H) // 2
    _, _, gate, hBC, dt = torch.split(
        proj, [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, H], dim=-1)
    hBC = causal_conv1d_update(hBC, conv_state, self.conv1d.weight.squeeze(1),
                               self.conv1d.bias, self.activation)
    x, Bv, Cv = torch.split(hBC, [self.intermediate_size, G * N, G * N], dim=-1)
    R = getattr(self, "_v4R", None)
    if R is not None:                       # rotated ckpt: apply R post-conv
        Bv = torch.einsum('bgn,gmn->bgm', Bv.view(B_, G, N).float(), R).reshape(B_, G * N).to(hBC.dtype)
        Cv = torch.einsum('bgn,gmn->bgm', Cv.view(B_, G, N).float(), R).reshape(B_, G * N).to(hBC.dtype)
    A = -torch.exp(self.A_log.float())                       # (H,)
    Bg = Bv.view(B_, G, N); Cg = Cv.view(B_, G, N)
    xh = x.view(B_, H, P)
    dt_e = dt[:, :, None].expand(-1, -1, P)
    dtb_e = self.dt_bias[:, None].expand(-1, P)
    D_e = self.D[:, None].expand(-1, P)
    # ---- hot tier: official fused kernel on the narrow slice ----
    A_hot = A[:, None, None].expand(H, P, pb).to(torch.float32)
    y = selective_state_update(st["hot"], xh, dt_e, A_hot,
                               Bg[..., :pb].contiguous(), Cg[..., :pb].contiguous(),
                               D_e, z=None, dt_bias=dtb_e, dt_softplus=True)
    y = y.view(B_, H, P).float()
    if pb < N:
        # ---- cold tier: quantized stale readout + decay compensation ----
        qc = Cg[..., pb:].repeat_interleave(H // G, dim=1)   # (B,H,Nc)
        comp = torch.exp(st["glog"])[..., None]              # (B,H,1)
        if cfg["cold"] == "fp8":
            fp8_cold_readout(qc.float().reshape(B_ * H, Nc).contiguous(),
                             st["snap8"], st["scale8"], st["y8"])
            y = y + st["y8"].view(B_, H, P) * comp
        else:                                                # bf16 snapshot, cuBLAS
            yc = torch.bmm(qc.reshape(B_ * H, 1, Nc).to(torch.bfloat16),
                           st["snapH"]).view(B_, H, P).float()
            y = y + yc * comp
        # ---- staleness bookkeeping + amortized exact flush ----
        dta = (F.softplus(dt.float() + self.dt_bias.float()) * A)   # (B,H) log-decay
        t = st["t"]
        st["bufB"][t % c].copy_(Bg[..., pb:].repeat_interleave(H // G, dim=1))
        st["bufX"][t % c].copy_(xh * F.softplus(dt.float() + self.dt_bias.float())[..., None])
        st["bufA"][t % c].copy_(dta)
        st["t"] = t + 1
        if (t + 1) % c == 0:
            cold = cache_params.layers[self.layer_idx].recurrent_states[..., pb:]
            suf = torch.flip(torch.cumsum(torch.flip(st["bufA"], [0]), 0), [0])  # (c,B,H)
            suf = suf - st["bufA"]                            # decay AFTER token j
            Wx = st["bufX"] * torch.exp(suf)[..., None]       # (c,B,H,P)
            cold.mul_(torch.exp(st["bufA"].sum(0))[..., None, None])
            cold.add_(torch.einsum('cbhp,cbhn->bhpn', Wx, st["bufB"]))
            st["glog"].zero_()
            if cfg["cold"] == "fp8":
                sc = cold.abs().amax(-2).clamp(min=1e-6).div(448.)       # (B,H,Nc)
                st["scale8"].copy_(sc.reshape(B_ * H, Nc))
                st["snap8"].copy_((cold.permute(0, 1, 3, 2) / sc[..., None])
                                  .reshape(B_ * H, Nc, P).to(torch.float8_e4m3fn))
            else:
                st["snapH"].copy_(cold.permute(0, 1, 3, 2)
                                  .reshape(B_ * H, Nc, P).to(torch.bfloat16))
        else:
            st["glog"] = st["glog"] + dta
    y = self.norm(y.view(B_, H * P).to(hidden_states.dtype), gate)
    return self.out_proj(y.to(self.out_proj.weight.dtype))[:, None, ...]


def _dispatch(self, hidden_states, cache_params=None, attention_mask=None, **kw):
    seq_len = hidden_states.shape[1]
    if (cache_params is not None and cache_params.has_previous_state(self.layer_idx)
            and seq_len == 1 and self._v4state is not None):
        return _v4_fused_decode(self, hidden_states, cache_params, attention_mask)
    if getattr(self, "_v4R", None) is not None:
        # rotated ckpt: prefill via torch_forward so ActRotMask applies R
        # (the fused prefill kernels bypass the act module entirely)
        out = self.torch_forward(hidden_states, cache_params, attention_mask)
    else:
        out = _orig_forward(self, hidden_states, cache_params=cache_params,
                            attention_mask=attention_mask)
    if cache_params is not None:                              # prefill done: capture
        B_ = hidden_states.shape[0]
        H, P, N, G = (self.num_heads, self.head_dim, self.ssm_state_size,
                      self.n_groups)
        cfg = self.v4cfg; pb = cfg["pb"]; Nc = N - pb; c = cfg["c"]
        rec = cache_params.layers[self.layer_idx].recurrent_states
        dev = rec.device
        st = {"hot": rec[..., :pb].contiguous().float(),
              "glog": torch.zeros(B_, H, device=dev),
              "t": 0,
              "bufB": torch.zeros(c, B_, H, Nc, device=dev),
              "bufX": torch.zeros(c, B_, H, P, device=dev),
              "bufA": torch.zeros(c, B_, H, device=dev)}
        cold = rec[..., pb:].float()
        if cfg["cold"] == "fp8":
            sc = cold.abs().amax(-2).clamp(min=1e-6).div(448.)
            st["scale8"] = sc.reshape(B_ * H, Nc).contiguous()
            st["snap8"] = ((cold.permute(0, 1, 3, 2) / sc[..., None])
                           .reshape(B_ * H, Nc, P).to(torch.float8_e4m3fn).contiguous())
            st["y8"] = torch.empty(B_ * H, P, device=dev)
        else:
            st["snapH"] = (cold.permute(0, 1, 3, 2).reshape(B_ * H, Nc, P)
                           .to(torch.bfloat16).contiguous())
        self._v4state = st
    return out


def install(model, pb=32, c=16, cold="bf16"):
    """Auto-detects rotation: if mixers carry ActRotMask (m.act.R), decode applies
    R explicitly and prefill routes through torch_forward -> accuracy-grade."""
    n = 0
    for m in model.modules():
        if type(m).__name__ == "NemotronHMamba2Mixer":
            m.v4cfg = dict(pb=pb, c=c, cold=cold)
            m._v4state = None
            m._v4R = (m.act.R.detach().float()
                      if hasattr(m.act, "R") else None)
            m.forward = _dispatch.__get__(m)
            n += 1
    return n
