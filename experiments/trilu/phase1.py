"""
TriLU Phase 1 quality experiment.

Compares activation functions in a tiny GPT trained on FineWeb-10B tokens:
  - ReLU squared (baseline)
  - GELU (control)
  - Symmetric TriLU with GELU-fit minimax init
  - Asymmetric TriLU with GELU-fit minimax init

Self-contained: vanilla PyTorch, no Muon/FP8/FA3/distributed/custom kernels.
The point is to measure steps-to-target on val loss at matched compute, not
to reproduce modded-nanogpt wallclock. Speed work happens in Phase 2.

Hardware:
  - Auto-detects CUDA / MPS / CPU.
  - --tiny config runs comfortably on M2 Air in a few minutes per seed.
  - Default config wants a real GPU (5060 Ti, A10, H100, etc).

Data:
  Expects modded-nanogpt's FineWeb-10B .bin files in data/fineweb10B/.
  Download with: python data/cached_fineweb10B.py 1

Usage:
  python experiments/trilu/phase1.py --tiny           # sanity check, ~5 min on M2
  python experiments/trilu/phase1.py                  # default, ~1 hr/seed on 5060 Ti
  python experiments/trilu/phase1.py --seeds 5        # more seeds for variance
  python experiments/trilu/phase1.py --activations relu2,trilu_asym   # subset
"""

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def _nullctx():
    yield


# -----------------------------------------------------------------------------
# Activations

class IQU(nn.Module):
    """Inverse-and-Quadratic Unit (IQU): C2 piecewise activation with no dead zone.
        x >= 0:  x + x^2     (quadratic branch; slope 1 and curvature 2 at origin)
        x <  0:  x/(1-x)     (inverse branch; slope 1 at 0, saturates to -1; no exp)
    C2 at the origin (value, slope, and curvature all match). No zeros -> dense
    activations; bounded below; cheap (one reciprocal, no transcendental)."""
    def forward(self, x):
        pos = F.relu(x)
        neg = x - pos                      # = min(x, 0) <= 0, so 1 - neg >= 1 (no div-by-zero)
        return pos + pos * pos + neg / (1.0 - neg)


class SNIQU(nn.Module):
    """Self-Normalizing Inverse-and-Quadratic Unit (SNIQU): IQU rescaled so that
    for x ~ N(0,1) the output is zero-mean and unit-variance. Same self-normalization
    recipe as SELU (solve moment equations), applied to the IQU shape. Constants from
    Gauss-Hermite quadrature: E[h]=0.70638, Var[h]=3.18830.
        sniqu(x) = LAM * (iqu(x) - BETA),  BETA=0.70638, LAM=1/sqrt(3.18830)=0.56004
    Keeps IQU's curvature (still leptokurtic, excess kurtosis ~10); only fixes
    scale+mean so the activation stops inflating the residual stream ~1.8x."""
    BETA = 0.70638
    LAM = 0.56004
    def forward(self, x):
        pos = F.relu(x)
        neg = x - pos
        h = pos + pos * pos + neg / (1.0 - neg)
        return self.LAM * (h - self.BETA)


class ReLU2(nn.Module):
    """Squared ReLU, optionally shifted and given a nonzero initial slope:

        f(x) = ReLU(x - t)**2 + s * ReLU(x - t)

    s=0, t=0 recovers plain ReLU² (max(0,x)²), whose derivative 2x -> 0 as
    x -> 0+, so small positive pre-activations are gradient-starved. s=1 puts a
    slope-1 floor at the threshold (derivative 2(x-t)+s >= 1 in the active
    region), fixing that without changing the cost. t shifts the activation
    threshold right (larger t -> more sparsity; post-LayerNorm input is ~N(0,1),
    so t=0 is ~50% dead, t=1 ~84% dead). Still transcendental-free and fusable.
    """

    def __init__(self, slope: float = 0.0, shift: float = 0.0, curv: float = 1.0):
        super().__init__()
        self.slope = slope
        self.shift = shift
        self.curv = curv

    def forward(self, x):
        z = F.relu(x - self.shift) if self.shift else F.relu(x)
        out = z.square()
        if self.curv != 1.0:
            out = self.curv * out
        if self.slope:
            out = out + self.slope * z
        return out


class XAbsX(nn.Module):
    """f(x) = x*|x| + s*x = sign(x)*x² + s*x

    Odd function (f(-x) = -f(x)). In a gated MLP with a linear gate, mean and skew
    of the output are automatically 0 (the gate's symmetry handles those, same as
    for SwiGLU/GeGLU/TriGLU); the distinguishing hypothesis for x|x| is using the
    *negative* half of the input distribution rather than zeroing it out like
    ReLU/ReLU² do. E[(z|z|)²] = E[z⁴] = 3 for z ~ N(0,1), so the natural init
    scaling for the gate branch is Var[W₂x] = 1/3 if you want output variance 1.

    Plain x|x| has derivative 2|x|, which is 0 at the origin (slow-start from both
    sides). The slope term s gives derivative c*2|x|+s, i.e. slope s at x=0, fixing
    the gradient-starved flat spot while keeping the function odd. s=1 is the
    two-sided analog of the sloped ReLU² (its positive branch c*x²+x matches exactly).
    The curvature c scales the x|x| term: f(x) = c*x|x| + s*x. With s=1, gating
    (half the outputs zeroed) and c=0.253, a standard-normal input gives unit-variance
    output (the variance-preserving, slope-1, monotonic member of the family).

    With s=1 this family (x + c*x|x|) is "QuadLU" -- the value branch of QuadGLU.
    """
    def __init__(self, slope: float = 0.0, curv: float = 1.0):
        super().__init__()
        self.slope = slope
        self.curv = curv

    def forward(self, x):
        out = x * x.abs()
        if self.curv != 1.0:
            out = self.curv * out
        if self.slope:
            out = out + self.slope * x
        return out


def _inv_softplus(y: float) -> float:
    """Numerically stable inverse of softplus: returns theta s.t. softplus(theta) = y, y > 0."""
    return math.log(math.expm1(y)) if y < 20 else y


# GELU-fit init values (computed earlier; see competition_strategy.html).
_INIT_VALUES = {
    "gelu_minimax": dict(L=1.910, R=1.910, alpha=0.247),   # recommended
    "gelu_rms":     dict(L=1.749, R=1.749, alpha=0.275),
    "c1":           dict(L=1.000, R=1.000, alpha=0.250),   # smooth-ReLU (no underbite)
}


class TriLU(nn.Module):
    """
    Three-piece Linear Unit:
        f(x) = 0                                          for x <= L
        f(x) = alpha*x**2 + beta*x + gamma                for L < x < R
        f(x) = x                                          for x >= R

    with continuity constraints q(L) = 0 and q(R) = R determining beta, gamma:
        beta  = R/(R-L) - alpha*(L+R)
        gamma = alpha*L*R - L*R/(R-L)

    L, R, alpha are learnable per-instance scalars (one TriLU per MLP layer in
    this harness). Positivity is enforced via softplus:
        L     = -softplus(theta_L)    (< 0)
        R     =  softplus(theta_R)    (> 0)
        alpha =  softplus(theta_alpha) (>= 0)

    If asymmetric=False, L and R share theta_a, so L = -a, R = +a always.
    """

    def __init__(self, asymmetric: bool = True, init: str = "gelu_minimax"):
        super().__init__()
        self.asymmetric = asymmetric
        v = _INIT_VALUES[init]

        if asymmetric:
            self.theta_L = nn.Parameter(torch.tensor(_inv_softplus(v["L"])))
            self.theta_R = nn.Parameter(torch.tensor(_inv_softplus(v["R"])))
        else:
            assert v["L"] == v["R"], "symmetric init requires L == R"
            self.theta_a = nn.Parameter(torch.tensor(_inv_softplus(v["L"])))

        self.theta_alpha = nn.Parameter(torch.tensor(_inv_softplus(v["alpha"])))

    def get_params(self):
        if self.asymmetric:
            L = -F.softplus(self.theta_L)
            R = F.softplus(self.theta_R)
        else:
            a = F.softplus(self.theta_a)
            L, R = -a, a
        alpha = F.softplus(self.theta_alpha)
        return L, R, alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, R, alpha = self.get_params()
        # Quadratic middle with C^0 continuity to flat and linear pieces.
        denom = (R - L).clamp(min=1e-6)
        beta = R / denom - alpha * (L + R)
        gamma = alpha * L * R - L * R / denom
        q = alpha * x * x + beta * x + gamma
        # Use full piecewise where to keep gradients clean.
        below = x <= L
        above = x >= R
        out = torch.where(below, torch.zeros_like(x), torch.where(above, x, q))
        return out


def make_activation(name: str, init: str = "gelu_minimax",
                    act_slope: float = 0.0, act_shift: float = 0.0, act_curv: float = 1.0):
    """Factory that returns a fresh activation module each call.

    For gated variants, returns the *inner* activation; the MLP class handles gating.
    act_slope affects the ReLU² family (relu2, reglu) and the x|x| family (xabsx, xglu);
    act_shift affects the ReLU² family only.
    """
    if name in ("relu2", "reglu"):
        return ReLU2(slope=act_slope, shift=act_shift, curv=act_curv)
    if name in ("gelu", "geglu"):
        return nn.GELU()
    if name == "swiglu":
        return nn.SiLU()
    if name in ("trilu_sym",):
        return TriLU(asymmetric=False, init=init)
    if name in ("trilu_asym", "triglu"):
        return TriLU(asymmetric=True, init=init)
    if name in ("xabsx", "xglu"):
        return XAbsX(slope=act_slope, curv=act_curv)
    if name == "bilinear":
        # No activation: gated MLP becomes (W1 x) * (W2 x) -- Shazeer's Bilinear.
        # The nonlinearity is purely the multiplicative gate. Cheapest gated variant.
        return nn.Identity()
    if name == "iqu":
        return IQU()
    if name in ("sniqu", "iqu_sn", "eluquad_sn"):   # sniqu = Self-Normalizing IQU
        return SNIQU()
    if name == "selu":
        return nn.SELU()
    raise ValueError(f"Unknown activation: {name}")


# Activations that use gated MLP (3 matmuls, hidden dim = round(8/3 * model_dim))
GATED_ACTIVATIONS = {"swiglu", "geglu", "triglu", "xglu", "reglu", "bilinear"}


# -----------------------------------------------------------------------------
# Model

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        # QK-norm: normalize Q and K per head to unit RMS before attention.
        # Motivated by conditioning: a non-unit-scale Q/K degrades Muon's
        # gradient orthogonalization the same way a non-zero activation mean does.
        self.q_norm = nn.RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


_DEAD_THRESHOLD = 1e-4  # |output| < this counts as "dead" for diagnostic purposes


class MLP(nn.Module):
    """Standard transformer MLP: W2 . act(W1 . x). Used by relu2, gelu, trilu_*."""

    def __init__(self, dim: int, mlp_dim: int, activation: nn.Module):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim, bias=False)
        self.fc2 = nn.Linear(mlp_dim, dim, bias=False)
        self.act = activation
        self._capture_dead = False
        self._last_dead_frac: float | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        if self._capture_dead:
            with torch.no_grad():
                self._last_dead_frac = (h.abs() < _DEAD_THRESHOLD).float().mean().item()
        return self.fc2(h)


class ChannelPool(nn.Module):
    """Compress the channel dim by factor k via fixed (non-learned) pooling.

    Groups the `dim` input channels into `dim // k` contiguous groups of `k` and
    pools within each group. This is the Squeeze-and-Excitation-style compression
    (Hu et al. 2018): the gate sees a k-fold summary of *all* channels rather than
    a fraction of them, at near-zero cost. Fixed (not learned) on purpose — a
    learned d -> d/k projection would just move the FLOPs we are trying to remove.
    """

    def __init__(self, dim: int, k: int, pool_type: str = "mean"):
        super().__init__()
        assert dim % k == 0, f"gate-pool factor {k} must divide model dim {dim}"
        self.dim, self.k, self.pool_type = dim, k, pool_type
        self.out_dim = dim // k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *lead, d = x.shape
        g = x.view(*lead, self.out_dim, self.k)
        if self.pool_type == "max":
            return g.amax(dim=-1)
        return g.mean(dim=-1)


def apply_gate_act(g: torch.Tensor, mode: str) -> torch.Tensor:
    """Nonlinearity on the gate branch (the 'whether-on' control).

    linear        : g                         (default; unbounded, signed -- SwiGLU-style)
    fast_sigmoid  : 0.5*g/(1+|g|) + 0.5        (cheap rational 0-1 valve, no transcendental;
                                                gentle polynomial-tail saturation)
    sigmoid       : 1/(1+e^-g)                 (true 0-1 valve; transcendental)

    The bounded gates implement the QuadGLU "division of labor" (transistor intuition):
    the gate decides *whether* a unit is on (0-1), the value branch (QuadLU = x+c*x|x|)
    decides the output *when* on. The linear gate (the modern SwiGLU default) instead
    lets the gate also amplify and flip sign -- which is why it tends to win.
    """
    if mode == "fast_sigmoid":
        return 0.5 * g / (1.0 + g.abs()) + 0.5
    if mode == "sigmoid":
        return torch.sigmoid(g)
    return g  # linear


class GatedMLP(nn.Module):
    """Gated MLP (SwiGLU / GeGLU / TriGLU): W2 . ( act(W1.x) * gate_act(V.x) ).

    With gate_pool > 1, the gate branch V sees a channel-pooled summary of the
    input (C7): x is compressed dim -> dim/k before the gate matmul, so V is
    (dim/k) x mlp_dim instead of dim x mlp_dim. The activated branch W1 still
    sees the full input. Tests whether the gate needs fine-grained per-channel
    info or only a compressed summary.

    gate_act applies a nonlinearity to the gate branch: 'linear' (default) is the
    standard SwiGLU signed/unbounded gate; 'fast_sigmoid'/'sigmoid' make it a
    bounded 0-1 'whether-on' valve. Paired with a QuadLU (x+c*x|x|) value branch,
    the bounded-gate form is "QuadGLU with a valve gate" (transistor intuition).
    """

    def __init__(self, dim: int, mlp_dim: int, activation: nn.Module,
                 gate_pool: int = 1, gate_pool_type: str = "mean", gate_act: str = "linear",
                 gate_zero_init: bool = False, dual_act: bool = False):
        super().__init__()
        self.dual_act = dual_act
        self.fc1 = nn.Linear(dim, mlp_dim, bias=False)
        if gate_pool > 1:
            self.gate_pool = ChannelPool(dim, gate_pool, gate_pool_type)
            self.fc_gate = nn.Linear(self.gate_pool.out_dim, mlp_dim, bias=False)
        else:
            self.gate_pool = None
            self.fc_gate = nn.Linear(dim, mlp_dim, bias=False)
        if gate_zero_init:
            # Zero gate matrix -> gate preactivation 0 at init. For a bounded gate
            # (fast_sigmoid/sigmoid) that is gate=0.5, the max-derivative / most-agnostic
            # point: every valve starts half-open and learns its own direction. Symmetry
            # is broken by the random fc1, so units still differentiate. (For a linear
            # gate this instead zeroes the block output, ReZero-style.)
            nn.init.zeros_(self.fc_gate.weight)
        self.fc2 = nn.Linear(mlp_dim, dim, bias=False)
        self.act = activation
        self.gate_act = gate_act
        self._capture_dead = False
        self._last_dead_frac: float | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_in = self.gate_pool(x) if self.gate_pool is not None else x
        g = self.fc_gate(gate_in)
        # dual_act: apply the SAME value activation to the gate branch too -- e.g.
        # double-QuadLU = QuadLU(W1 x) * QuadLU(W2 x). Symmetric in the two branches, so
        # the quadratic is available on whichever branch specializes as the value (tests
        # the role/arbitrariness hypothesis vs xglu's one-fixed-branch quadratic).
        gate = self.act(g) if self.dual_act else apply_gate_act(g, self.gate_act)
        h = self.act(self.fc1(x)) * gate
        if self._capture_dead:
            with torch.no_grad():
                self._last_dead_frac = (h.abs() < _DEAD_THRESHOLD).float().mean().item()
        return self.fc2(h)


def make_mlp(dim: int, model_dim: int, act_name: str, act_init: str,
             gate_pool: int = 1, gate_pool_type: str = "mean",
             act_slope: float = 0.0, act_shift: float = 0.0, act_curv: float = 1.0,
             gate_act: str = "linear", gate_zero_init: bool = False, dual_act: bool = False) -> nn.Module:
    """Build an MLP block. Hidden dim chosen to match params: 4d for standard, (8/3)d for gated."""
    act = make_activation(act_name, act_init, act_slope=act_slope, act_shift=act_shift, act_curv=act_curv)
    if act_name in GATED_ACTIVATIONS:
        # 8/3 * d, rounded to nearest multiple of 64 for hardware-friendliness
        hidden = max(64, round(8 * model_dim / 3 / 64) * 64)
        return GatedMLP(dim, hidden, act, gate_pool=gate_pool, gate_pool_type=gate_pool_type,
                        gate_act=gate_act, gate_zero_init=gate_zero_init, dual_act=dual_act)
    else:
        hidden = 4 * model_dim
        return MLP(dim, hidden, act)


class Block(nn.Module):
    def __init__(self, dim, num_heads, model_dim, act_name, act_init,
                 gate_pool=1, gate_pool_type="mean", act_slope=0.0, act_shift=0.0, act_curv=1.0,
                 gate_act="linear", gate_zero_init=False, dual_act=False, qk_norm=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, qk_norm=qk_norm)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = make_mlp(dim, model_dim, act_name, act_init,
                            gate_pool=gate_pool, gate_pool_type=gate_pool_type,
                            act_slope=act_slope, act_shift=act_shift, act_curv=act_curv,
                            gate_act=gate_act, gate_zero_init=gate_zero_init, dual_act=dual_act)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


def _ce_chunk(a, w, b):
    return F.cross_entropy(F.linear(a, w), b, reduction="sum")


def chunked_cross_entropy(x, weight, targets, chunk):
    """Mean cross-entropy over tokens, computed in chunks so the full (N, vocab) logits
    tensor is never held at once. Each chunk's logits are recomputed in backward
    (checkpoint), so peak logits memory ~ (chunk, vocab) instead of (N, vocab)."""
    n = x.size(0)
    total = x.new_zeros(())
    for i in range(0, n, chunk):
        total = total + torch.utils.checkpoint.checkpoint(
            _ce_chunk, x[i:i + chunk], weight, targets[i:i + chunk], use_reentrant=False)
    return total / n


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, num_layers, dim, num_heads, seq_len, act_name, act_init,
                 gate_pool=1, gate_pool_type="mean", act_slope=0.0, act_shift=0.0, act_curv=1.0,
                 gate_act="linear", gate_zero_init=False, dual_act=False, qk_norm=False):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(seq_len, dim)
        # one MLP per layer; gated variants get their own hidden dim sized to match params
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, dim, act_name, act_init,
                  gate_pool=gate_pool, gate_pool_type=gate_pool_type,
                  act_slope=act_slope, act_shift=act_shift, act_curv=act_curv,
                  gate_act=gate_act, gate_zero_init=gate_zero_init, dual_act=dual_act,
                  qk_norm=qk_norm)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        # Tie embedding and output weights (standard for small GPTs).
        self.head.weight = self.token_embed.weight
        self.seq_len = seq_len
        self.ce_chunk = 0  # 0 = materialize full logits; >0 = chunked CE (memory saver)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.token_embed(idx) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        if targets is None:
            return self.head(x), None
        if self.ce_chunk and self.ce_chunk > 0:
            # Chunked cross-entropy: never materialize the full (B*T, vocab) logits tensor
            # (the memory bottleneck that caps batch size). Each chunk's logits are
            # recomputed in backward via checkpointing, so peak memory ~ one chunk.
            loss = chunked_cross_entropy(x.reshape(-1, x.size(-1)), self.head.weight,
                                         targets.reshape(-1), self.ce_chunk)
            return None, loss
        logits = self.head(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


# -----------------------------------------------------------------------------
# Data

def _load_data_shard(path: Path) -> torch.Tensor:
    """Load a modded-nanogpt .bin shard. 256 int32 header + uint16 token stream.

    Returns int64 (long) so embedding lookups don't need a runtime cast.
    Clones the tensor so it owns its memory — sharing numpy-owned memory
    with MPS has caused SIGBUS on Apple Silicon in some torch builds.
    """
    import numpy as np
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520, f"magic mismatch in {path}"
    assert header[1] == 1, f"unsupported version in {path}"
    num_tokens = int(header[2])
    with path.open("rb") as f:
        f.seek(256 * 4)
        tokens_np = np.fromfile(f, dtype=np.uint16, count=num_tokens)
    assert tokens_np.size == num_tokens, f"short read in {path}"
    # cast to int64 in numpy (cheap), then clone into a torch-owned buffer
    return torch.from_numpy(tokens_np.astype(np.int64)).clone()


def load_data(data_dir: str = "data/fineweb10B", train_cap: int = 0, val_cap: int = 0):
    data_dir = Path(data_dir)
    train_files = sorted(data_dir.glob("fineweb_train_*.bin"))
    val_files = sorted(data_dir.glob("fineweb_val_*.bin"))
    if not train_files or not val_files:
        print(f"\nERROR: no .bin files in {data_dir.resolve()}")
        print("Run from repo root:  python data/cached_fineweb10B.py 1")
        sys.exit(1)
    print(f"Loading train shard: {train_files[0].name}")
    train = _load_data_shard(train_files[0])  # already int32
    print(f"Loading val shard:   {val_files[0].name}")
    val = _load_data_shard(val_files[0])
    if train_cap:
        train = train[:train_cap]
    if val_cap:
        val = val[:val_cap]
    print(f"  train tokens: {len(train):,}")
    print(f"  val tokens:   {len(val):,}")
    return train, val


def sample_batch(data: torch.Tensor, batch_size: int, seq_len: int, device, generator: torch.Generator):
    """Sample random contiguous windows."""
    high = len(data) - seq_len - 1
    ix = torch.randint(0, high, (batch_size,), generator=generator).tolist()
    batch = torch.stack([data[i : i + seq_len + 1] for i in ix]).contiguous()
    return batch.to(device, non_blocking=True)


# -----------------------------------------------------------------------------
# Training

def get_lr(step, total_steps, max_lr, min_lr, warmup):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def lr_mult(step, total_steps, warmup, min_frac=0.1):
    """Schedule as a multiplier in [~0, 1] (warmup then cosine to min_frac)."""
    return get_lr(step, total_steps, 1.0, min_frac, warmup)


# -----------------------------------------------------------------------------
# Muon optimizer (single-GPU). The real modded-nanogpt optimizer: momentum SGD on
# the hidden weight matrices, with each update orthogonalized via Newton-Schulz.
# Embeddings / norms / scalar activation params use AdamW instead. We add it here
# because Muon "makes activation scale non-absorbable" -- so an activation/gate win
# under AdamW must be re-confirmed under Muon before it means anything for the record.

# Newton-Schulz uses a single fixed quintic; Polar Express (the real modded-nanogpt
# orthogonalizer, arXiv 2505.16932) uses a per-step coefficient schedule that converges
# to the orthogonal factor faster. Porting it closes the most mechanism-relevant gap
# (our conclusions hinge on the orthogonalizer); the math is hardware-agnostic.
_POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]
_NEWTON_SCHULZ_COEFFS = [(3.4445, -4.7750, 2.0315)] * 5


@torch.no_grad()
def orthogonalize(G, method="polar_express"):
    """Orthogonalize G (2D) via a quintic iteration. method='polar_express' uses the
    real modded-nanogpt coefficient schedule; 'newton_schulz' the classic fixed quintic."""
    assert G.ndim == 2
    coeffs = _POLAR_EXPRESS_COEFFS if method == "polar_express" else _NEWTON_SCHULZ_COEFFS
    X = G.bfloat16()
    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T
    # spectral-norm safety factor (matches the real code's 1+2e-2)
    X = X / (X.norm() * (1 + 2e-2) + 1e-7)
    for a, b, c in coeffs:
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X.to(G.dtype)


@torch.no_grad()
def normuon_variance_reduce(v, buf, beta2):
    """NorMuon's neuron-wise variance normalization of an orthogonalized update v (2D).
    Ported from the record's _apply_normuon_variance_reduction. Maintains a per-neuron
    second-moment EMA `buf`, divides each neuron's update by sqrt of it, then globally
    rescales so the Frobenius norm is unchanged -- so it only *redistributes* magnitude
    across neurons (the larger dim), leaving the overall step size (and LR meaning) fixed.
    arXiv 2510.05491. buf shape = v with the reduced dim collapsed to 1."""
    red_dim = -1 if v.size(-2) >= v.size(-1) else -2   # average over the shorter dim
    n = v.size(red_dim)
    v_mean = v.float().square().mean(dim=red_dim, keepdim=True)      # per-neuron mean-square
    v_norm = (v_mean.sum() * n).sqrt()                              # ||v||_F
    buf.lerp_(v_mean.to(buf.dtype), 1 - beta2)                      # second-moment EMA
    step = buf.clamp_min(1e-10).rsqrt()                            # per-neuron 1/sqrt(v2)
    v_norm_new = ((v_mean * n) * step.float().square()).sum().sqrt().clamp_min(1e-10)
    return v.mul_((step * (v_norm / v_norm_new)).type_as(v))        # normalize, preserve ||·||_F


class Muon(torch.optim.Optimizer):
    """Muon (momentum SGD + orthogonalized update). With beta2 set, becomes NorMuon:
    adds the neuron-wise variance normalization the real record uses."""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ortho="polar_express",
                 beta2=None):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ortho=ortho, beta2=beta2))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum, beta2 = group["lr"], group["momentum"], group["beta2"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                g = orthogonalize(g, method=group["ortho"])
                if beta2 is not None:           # NorMuon: neuron-wise variance normalization
                    if "v2_buffer" not in state:
                        shape = list(g.shape)
                        shape[-1 if g.size(-2) >= g.size(-1) else -2] = 1
                        state["v2_buffer"] = torch.zeros(shape, device=g.device, dtype=torch.float32)
                    g = normuon_variance_reduce(g, state["v2_buffer"], beta2)
                    p.add_(g, alpha=-lr)        # variance-reduce preserves ||·||_F; lr applied directly
                else:
                    # shape-aware scale so the RMS update is comparable across matrices
                    p.add_(g, alpha=-lr * max(1.0, p.size(0) / p.size(1)) ** 0.5)


def build_optimizers(model, config):
    """Return a list of optimizers. Each param group carries a 'base_lr' that the
    training loop scales by the shared schedule multiplier.

    AdamW path (default): one AdamW over everything, activation params at 0.1x.
    Muon path: Muon over the hidden weight matrices (2D params inside blocks),
    AdamW over embeddings / norms / activation params."""
    freeze_trilu = config.get("freeze_trilu", False)
    body, act, other = [], [], []
    for name, p in model.named_parameters():
        if "theta_" in name:
            if freeze_trilu:
                p.requires_grad_(False)
            else:
                act.append(p)
        elif p.ndim == 2 and name.startswith("blocks"):
            body.append(p)
        else:
            other.append(p)

    lr = config["lr"]
    adam_groups = []

    def add_adam(params, base):
        if params:
            adam_groups.append({"params": params, "base_lr": base, "lr": base})

    opts = []
    opt_name = config.get("optimizer", "adamw")
    if opt_name in ("muon", "normuon"):
        mlr = config.get("muon_lr", 0.02)
        beta2 = config.get("muon_beta2", 0.9) if opt_name == "normuon" else None
        opts.append(Muon([{"params": body, "base_lr": mlr, "lr": mlr}], lr=mlr,
                         ortho=config.get("muon_ortho", "polar_express"), beta2=beta2))
        add_adam(other, lr)
    else:
        add_adam(body + other, lr)
    add_adam(act, lr * 0.1)
    if adam_groups:
        opts.append(torch.optim.AdamW(adam_groups, betas=(0.9, 0.95), weight_decay=0.1))
    return opts


def collect_trilu_params(model: nn.Module):
    snap = []
    for name, mod in model.named_modules():
        if isinstance(mod, TriLU):
            L, R, alpha = mod.get_params()
            snap.append(dict(layer=name, L=L.item(), R=R.item(), alpha=alpha.item()))
    return snap


@torch.no_grad()
def evaluate(model, val_data, batch_size, seq_len, device, gen, num_batches=20):
    model.eval()
    losses = []
    amp_ctx_factory = (lambda: torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)) if device == "cuda" else _nullctx
    for _ in range(num_batches):
        batch = sample_batch(val_data, batch_size, seq_len, device, gen)
        x, y = batch[:, :-1], batch[:, 1:]
        with amp_ctx_factory():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def measure_dead_zones(model, val_data, batch_size, seq_len, device, gen):
    """Run a single eval batch with dead-zone capture enabled; return per-layer fractions.

    A neuron is 'dead' for this measurement if its activation output magnitude is < 1e-4.
    For ReLU²/TriLU this corresponds to pre-activation in the zero region.
    For GELU/SiLU this is approximately never (smooth tails are nonzero).
    """
    mlp_modules = []
    for name, m in model.named_modules():
        if isinstance(m, (MLP, GatedMLP)):
            mlp_modules.append((name, m))
            m._capture_dead = True
    model.eval()
    batch = sample_batch(val_data, batch_size, seq_len, device, gen)
    x, y = batch[:, :-1], batch[:, 1:]
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else _nullctx()
    with amp_ctx:
        model(x, y)
    model.train()
    for _, m in mlp_modules:
        m._capture_dead = False
    return [(name, m._last_dead_frac) for name, m in mlp_modules]


def train_one(config, train_data, val_data, seed, device):
    torch.manual_seed(seed)
    gen_train = torch.Generator().manual_seed(seed)
    gen_val = torch.Generator().manual_seed(seed + 10_000)

    model = TinyGPT(
        vocab_size=config["vocab_size"],
        num_layers=config["num_layers"],
        dim=config["dim"],
        num_heads=config["num_heads"],
        seq_len=config["seq_len"],
        act_name=config["activation"],
        act_init=config.get("init", "gelu_minimax"),
        gate_pool=config.get("gate_pool", 1),
        gate_pool_type=config.get("gate_pool_type", "mean"),
        act_slope=config.get("act_slope", 0.0),
        act_curv=config.get("act_curv", 1.0),
        gate_act=config.get("gate_act", "linear"),
        gate_zero_init=config.get("gate_zero_init", False),
        dual_act=config.get("dual_act", False),
        act_shift=config.get("act_shift", 0.0),
        qk_norm=config.get("qk_norm", False),
    ).to(device)
    model.ce_chunk = config.get("ce_chunk", 0)
    # Compiled view for the train/eval forward (big speedup via kernel fusion); the
    # eager `model` is used for dead-zone capture (its in-forward .item() would force
    # recompiles) and shares parameters with the compiled view.
    cmodel = torch.compile(model) if config.get("compile") else model

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}  optimizer={config.get('optimizer','adamw')}"
          f"{'  compile' if config.get('compile') else ''}"
          f"{f'  ce_chunk={model.ce_chunk}' if model.ce_chunk else ''}")

    freeze_trilu = config.get("freeze_trilu", False)
    optimizers = build_optimizers(model, config)
    if freeze_trilu and config["activation"] in ("trilu_sym", "trilu_asym", "triglu"):
        print(f"  TriLU activation params FROZEN at init (0 learnable activation params)")

    has_trilu = config["activation"] in ("trilu_sym", "trilu_asym", "triglu")
    total_steps, warmup = config["total_steps"], config["warmup"]
    log = {"val_loss": [], "trilu_params": [], "dead_zones": [], "wallclock": [], "config": config, "seed": seed}

    t0 = time.time()
    for step in range(total_steps):
        mult = lr_mult(step, total_steps, warmup)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * mult

        batch = sample_batch(train_data, config["batch_size"], config["seq_len"], device, gen_train)
        x, y = batch[:, :-1], batch[:, 1:]
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else _nullctx()
        with amp_ctx:
            _, loss = cmodel(x, y)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()

        if step % config["eval_every"] == 0 or step == total_steps - 1:
            val_loss = evaluate(cmodel, val_data, config["batch_size"], config["seq_len"], device, gen_val)
            dead = measure_dead_zones(model, val_data, config["batch_size"], config["seq_len"], device, gen_val)
            mean_dead = sum(f for _, f in dead) / max(len(dead), 1)
            elapsed = time.time() - t0
            log["val_loss"].append([step, val_loss])
            log["wallclock"].append([step, elapsed])
            log["dead_zones"].append([step, dead])
            if has_trilu:
                log["trilu_params"].append([step, collect_trilu_params(model)])
            print(f"  step {step:5d}  lr={optimizers[0].param_groups[0]['lr']:.5f}  train={loss.item():.4f}  "
                  f"val={val_loss:.4f}  dead={mean_dead:.3f}  t={elapsed:.1f}s")

    # Convergence metrics (less noisy / better aligned with steps-to-target than the
    # single final eval): smoothed final loss, area under the val curve, steps-to-target.
    vals = [v for _, v in log["val_loss"]]
    k = min(5, len(vals))
    log["final_loss"] = vals[-1]
    log["final_loss_smoothed"] = sum(vals[-k:]) / k
    log["auc"] = sum(vals) / len(vals)
    target = config.get("target_loss")
    log["steps_to_target"] = next((s for s, v in log["val_loss"] if v <= target), None) if target else None
    print(f"  -> final={log['final_loss']:.4f}  smoothed(last{k})={log['final_loss_smoothed']:.4f}  "
          f"auc={log['auc']:.4f}  steps_to_target={log['steps_to_target']}")
    return log


# -----------------------------------------------------------------------------
# Main

def pick_device(force: str = None):
    if force:
        return force
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


CONFIGS = {
    "tiny": dict(  # fits on M2 Air in a few minutes; sanity check only
        num_layers=3, dim=128, num_heads=4,
        seq_len=256, batch_size=8,
        total_steps=300, warmup=30, eval_every=25,
        lr=3e-4, vocab_size=50304,
    ),
    "default": dict(  # comfortable on 5060 Ti / A10, ~40 min per seed
        num_layers=6, dim=384, num_heads=6,
        seq_len=1024, batch_size=16,
        total_steps=1000, warmup=80, eval_every=50,
        lr=3e-4, vocab_size=50304,
    ),
    "medium": dict(  # bigger, for H100; ~3hr per seed
        num_layers=8, dim=512, num_heads=8,
        seq_len=1024, batch_size=32,
        total_steps=3000, warmup=200, eval_every=100,
        lr=3e-4, vocab_size=50304,
    ),
    "small": dict(  # matches the modded-nanogpt small-track architecture (~124M):
                    # 12 layers, model_dim 768, 6 heads (head_dim 128). Removes the
                    # scale confound for the activation-under-Muon study. Use with
                    # --compile and --ce-chunk; tune --batch-size to fit 16 GB.
        num_layers=12, dim=768, num_heads=6,
        seq_len=1024, batch_size=16,
        total_steps=4000, warmup=256, eval_every=100,
        lr=3e-4, vocab_size=50304,
    ),
}


ALL_ACTIVATIONS = ["relu2", "gelu", "trilu_sym", "trilu_asym", "swiglu", "geglu", "triglu",
                   "xabsx", "xglu", "reglu", "bilinear", "iqu", "sniqu", "selu"]

# Default 7-activation sweep covering the standard/gated x ReLU-like/GELU-like/TriLU comparison,
# plus xglu (gated x|x|) which tests using the negative half of the input distribution.
# At 1000 steps, 3 seeds, ~40 min/run on 5060 Ti, total ~14 hr.
DEFAULT_ACTIVATIONS = ["relu2", "gelu", "trilu_asym", "swiglu", "geglu", "triglu", "xglu"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default="default")
    ap.add_argument("--tiny", action="store_true", help="alias for --config tiny")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=0,
                    help="first seed index (run indices seed_start .. seed_start+seeds-1). Lets a "
                         "sweep loop seeds OUTER / variants INNER for early tentative comparison.")
    ap.add_argument("--steps", type=int, default=None, help="override total_steps")
    ap.add_argument("--batch-size", type=int, default=None, help="override batch_size")
    ap.add_argument("--activations", type=str, default=",".join(DEFAULT_ACTIVATIONS),
                    help="comma-separated subset of: " + ",".join(ALL_ACTIVATIONS))
    ap.add_argument("--init", type=str, default="gelu_minimax",
                    choices=list(_INIT_VALUES))
    ap.add_argument("--data-dir", type=str, default="data/fineweb10B")
    ap.add_argument("--out", type=str, default="experiments/trilu/results.json")
    ap.add_argument("--device", type=str, default=None,
                    choices=[None, "cpu", "mps", "cuda"],
                    help="override device auto-detection (try --device cpu if MPS crashes)")
    ap.add_argument("--freeze-trilu", action="store_true",
                    help="freeze TriLU/TriGLU activation params at their init values (cleanest "
                         "hypothesis test — like-for-like with fixed ReLU²/GELU/etc.)")
    ap.add_argument("--gate-pool", type=int, default=1,
                    help="compress gate input by this factor via channel pooling (C7); "
                         "1 = no pooling. Only affects gated activations.")
    ap.add_argument("--gate-pool-type", type=str, default="mean", choices=["mean", "max"],
                    help="pooling op for --gate-pool")
    ap.add_argument("--act-slope", type=float, default=0.0,
                    help="initial slope s at the origin. ReLU² family: ReLU(x-t)²+s·ReLU(x-t); "
                         "x|x| family: x|x|+s·x. s=0 is the plain activation; s=1 gives slope 1 "
                         "at x=0. Affects relu2/reglu and xabsx/xglu.")
    ap.add_argument("--act-shift", type=float, default=0.0,
                    help="threshold shift t of the ReLU² family (larger t -> more sparsity). "
                         "Affects relu2/reglu only.")
    ap.add_argument("--act-curv", type=float, default=1.0,
                    help="curvature c, the coefficient on the quadratic term: ReLU² family "
                         "c·ReLU(x-t)²+s·ReLU(x-t); x|x| family c·x|x|+s·x. c=1 is the plain "
                         "activation. With --act-slope 1 and a gated MLP, c=0.253 makes a "
                         "standard-normal input give unit-variance output. Affects relu2/reglu "
                         "and xabsx/xglu.")
    ap.add_argument("--gate-act", type=str, default="linear",
                    choices=["linear", "fast_sigmoid", "sigmoid"],
                    help="nonlinearity on the gate branch of a gated MLP. linear = standard "
                         "SwiGLU signed/unbounded gate; fast_sigmoid = cheap rational 0-1 valve "
                         "(0.5*g/(1+|g|)+0.5); sigmoid = true 0-1 valve. Only affects gated activations.")
    ap.add_argument("--dual-act", action="store_true",
                    help="apply the value activation to the gate branch too (e.g. double-QuadLU = "
                         "QuadLU(W1 x)*QuadLU(W2 x)). Symmetric; quadratic available on both branches. Gated only.")
    ap.add_argument("--gate-zero-init", action="store_true",
                    help="zero-init the gate matrix. With a bounded gate this starts every valve "
                         "at 0.5 (max-derivative, agnostic), learning its direction from there; "
                         "with a linear gate it zeroes the block output (ReZero-style). Gated only.")
    ap.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon", "normuon"],
                    help="adamw (default); muon (orthogonalized momentum on the hidden matrices, "
                         "AdamW on embeddings/norms); normuon (muon + neuron-wise variance "
                         "normalization -- the real record's optimizer, arXiv 2510.05491).")
    ap.add_argument("--lr", type=float, default=None, help="AdamW base learning rate override (default: use config value, 3e-4).")
    ap.add_argument("--muon-lr", type=float, default=0.02, help="base LR for the Muon/NorMuon group.")
    ap.add_argument("--muon-beta2", type=float, default=0.9,
                    help="NorMuon second-moment EMA decay (only used with --optimizer normuon).")
    ap.add_argument("--muon-ortho", type=str, default="polar_express",
                    choices=["polar_express", "newton_schulz"],
                    help="orthogonalizer for Muon. polar_express = the real modded-nanogpt schedule "
                         "(default); newton_schulz = classic fixed quintic (what the earlier runs used).")
    ap.add_argument("--qk-norm", action="store_true",
                    help="apply RMSNorm to Q and K per head before attention (QK-norm). "
                         "Motivated by the same conditioning argument as sniqu: normalizing "
                         "Q/K prevents non-unit scale from degrading Muon's gradient "
                         "orthogonalization.")
    ap.add_argument("--target-loss", type=float, default=None,
                    help="if set, log steps-to-reach this val loss (benchmark-aligned metric).")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the train/eval forward (kernel fusion; big speedup, "
                         "one-time compile cost). Dead-zone capture stays eager.")
    ap.add_argument("--ce-chunk", type=int, default=0,
                    help="chunked cross-entropy chunk size (tokens). 0 = full logits. >0 avoids "
                         "materializing the (B*T, vocab) logits tensor, freeing memory for bigger batch.")
    args = ap.parse_args()

    if args.tiny:
        args.config = "tiny"

    cfg_base = dict(CONFIGS[args.config])
    if args.steps is not None:
        cfg_base["total_steps"] = args.steps
    if args.batch_size is not None:
        cfg_base["batch_size"] = args.batch_size
    cfg_base["gate_pool"] = args.gate_pool
    cfg_base["gate_pool_type"] = args.gate_pool_type
    cfg_base["act_slope"] = args.act_slope
    cfg_base["act_shift"] = args.act_shift
    cfg_base["act_curv"] = args.act_curv
    cfg_base["gate_act"] = args.gate_act
    cfg_base["gate_zero_init"] = args.gate_zero_init
    cfg_base["dual_act"] = args.dual_act
    cfg_base["qk_norm"] = args.qk_norm
    cfg_base["compile"] = args.compile
    cfg_base["ce_chunk"] = args.ce_chunk
    cfg_base["optimizer"] = args.optimizer
    if args.lr is not None:
        cfg_base["lr"] = args.lr
    cfg_base["muon_lr"] = args.muon_lr
    cfg_base["muon_beta2"] = args.muon_beta2
    cfg_base["muon_ortho"] = args.muon_ortho
    cfg_base["target_loss"] = args.target_loss

    device = pick_device(args.device)
    print(f"Device: {device}")
    print(f"Config: {args.config}  steps={cfg_base['total_steps']}  seeds={args.seeds}")

    activations = args.activations.split(",")
    for a in activations:
        if a not in ALL_ACTIVATIONS:
            print(f"Unknown activation: {a}  (known: {', '.join(ALL_ACTIVATIONS)})")
            sys.exit(1)

    # Cap data for tiny config to avoid loading 100M tokens for no reason.
    train_cap = 5_000_000 if args.config == "tiny" else 0
    val_cap = 1_000_000 if args.config == "tiny" else 0
    train_data, val_data = load_data(args.data_dir, train_cap=train_cap, val_cap=val_cap)

    all_results = {}
    for act in activations:
        cfg = dict(cfg_base, activation=act, init=args.init, freeze_trilu=args.freeze_trilu)
        # Self-describing key so pooled / shaped runs don't collide in the same file.
        key = act
        if args.gate_pool != 1:
            key += f"_pool{args.gate_pool}{args.gate_pool_type[0]}"
        if args.act_slope or args.act_shift:
            key += f"_s{args.act_slope:g}t{args.act_shift:g}"
        if args.act_curv != 1.0:
            key += f"_c{args.act_curv:g}"
        if args.gate_act != "linear":
            key += f"_g{args.gate_act}"
        if args.gate_zero_init:
            key += "_zg"
        if args.dual_act:
            key += "_dual"
        if args.optimizer != "adamw":
            key += f"_{args.optimizer}"
        all_results[key] = []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            print(f"\n=== activation={act}  gate_pool={args.gate_pool}  seed={seed} ===")
            log = train_one(cfg, train_data, val_data, seed, device)
            all_results[key].append(log)
            # Save incrementally so a crash doesn't lose everything.
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nSaved results to {args.out}")
    print("Plot with experiments/trilu/plot.py (or your own script).")


if __name__ == "__main__":
    main()
