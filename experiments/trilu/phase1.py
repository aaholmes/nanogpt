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
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Activations

class ReLU2(nn.Module):
    def forward(self, x):
        return F.relu(x).square()


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


def make_activation(name: str, init: str = "gelu_minimax"):
    """Factory that returns a fresh activation module each call.

    For gated variants, returns the *inner* activation; the MLP class handles gating.
    """
    if name in ("relu2",):
        return ReLU2()
    if name in ("gelu", "geglu"):
        return nn.GELU()
    if name == "swiglu":
        return nn.SiLU()
    if name in ("trilu_sym",):
        return TriLU(asymmetric=False, init=init)
    if name in ("trilu_asym", "triglu"):
        return TriLU(asymmetric=True, init=init)
    raise ValueError(f"Unknown activation: {name}")


# Activations that use gated MLP (3 matmuls, hidden dim = round(8/3 * model_dim))
GATED_ACTIVATIONS = {"swiglu", "geglu", "triglu"}


# -----------------------------------------------------------------------------
# Model

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class MLP(nn.Module):
    """Standard transformer MLP: W2 . act(W1 . x). Used by relu2, gelu, trilu_*."""

    def __init__(self, dim: int, mlp_dim: int, activation: nn.Module):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim, bias=False)
        self.fc2 = nn.Linear(mlp_dim, dim, bias=False)
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class GatedMLP(nn.Module):
    """Gated MLP (SwiGLU / GeGLU / TriGLU): W2 . ( act(W1.x) * (V.x) )."""

    def __init__(self, dim: int, mlp_dim: int, activation: nn.Module):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim, bias=False)
        self.fc_gate = nn.Linear(dim, mlp_dim, bias=False)
        self.fc2 = nn.Linear(mlp_dim, dim, bias=False)
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)) * self.fc_gate(x))


def make_mlp(dim: int, model_dim: int, act_name: str, act_init: str) -> nn.Module:
    """Build an MLP block. Hidden dim chosen to match params: 4d for standard, (8/3)d for gated."""
    act = make_activation(act_name, act_init)
    if act_name in GATED_ACTIVATIONS:
        # 8/3 * d, rounded to nearest multiple of 64 for hardware-friendliness
        hidden = max(64, round(8 * model_dim / 3 / 64) * 64)
        return GatedMLP(dim, hidden, act)
    else:
        hidden = 4 * model_dim
        return MLP(dim, hidden, act)


class Block(nn.Module):
    def __init__(self, dim, num_heads, model_dim, act_name, act_init):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = make_mlp(dim, model_dim, act_name, act_init)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, num_layers, dim, num_heads, seq_len, act_name, act_init):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(seq_len, dim)
        # one MLP per layer; gated variants get their own hidden dim sized to match params
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, dim, act_name, act_init)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        # Tie embedding and output weights (standard for small GPTs).
        self.head.weight = self.token_embed.weight
        self.seq_len = seq_len

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.token_embed(idx) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


# -----------------------------------------------------------------------------
# Data

def _load_data_shard(path: Path) -> torch.Tensor:
    """Load a modded-nanogpt .bin shard. 256 int32 header + uint16 token stream."""
    header = torch.from_file(str(path), False, 256, dtype=torch.int32)
    assert header[0].item() == 20240520, f"magic mismatch in {path}"
    assert header[1].item() == 1, f"unsupported version in {path}"
    num_tokens = int(header[2].item())
    with path.open("rb") as f:
        f.seek(256 * 4)
        tokens = torch.empty(num_tokens, dtype=torch.uint16)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, f"short read in {path}"
    return tokens


def load_data(data_dir: str = "data/fineweb10B", train_cap: int = 0, val_cap: int = 0):
    data_dir = Path(data_dir)
    train_files = sorted(data_dir.glob("fineweb_train_*.bin"))
    val_files = sorted(data_dir.glob("fineweb_val_*.bin"))
    if not train_files or not val_files:
        print(f"\nERROR: no .bin files in {data_dir.resolve()}")
        print("Run from repo root:  python data/cached_fineweb10B.py 1")
        sys.exit(1)
    print(f"Loading train shard: {train_files[0].name}")
    train = _load_data_shard(train_files[0]).int()  # int32 to be safe; tokens fit easily
    print(f"Loading val shard:   {val_files[0].name}")
    val = _load_data_shard(val_files[0]).int()
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
    ix = torch.randint(0, high, (batch_size,), generator=generator)
    batch = torch.stack([data[i : i + seq_len + 1].long() for i in ix])
    return batch.to(device, non_blocking=True)


# -----------------------------------------------------------------------------
# Training

def get_lr(step, total_steps, max_lr, min_lr, warmup):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


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
    for _ in range(num_batches):
        batch = sample_batch(val_data, batch_size, seq_len, device, gen)
        x, y = batch[:, :-1], batch[:, 1:]
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


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
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    # Separate param groups: main weights at full LR, TriLU activation params at 0.1x.
    main_params, act_params = [], []
    for name, p in model.named_parameters():
        if "theta_" in name:
            act_params.append(p)
        else:
            main_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": main_params, "lr": config["lr"]},
            {"params": act_params, "lr": config["lr"] * 0.1},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    has_trilu = config["activation"] in ("trilu_sym", "trilu_asym", "triglu")
    log = {"val_loss": [], "trilu_params": [], "wallclock": [], "config": config, "seed": seed}

    t0 = time.time()
    for step in range(config["total_steps"]):
        cur_lr = get_lr(step, config["total_steps"], config["lr"], config["lr"] * 0.1, config["warmup"])
        optimizer.param_groups[0]["lr"] = cur_lr
        optimizer.param_groups[1]["lr"] = cur_lr * 0.1

        batch = sample_batch(train_data, config["batch_size"], config["seq_len"], device, gen_train)
        x, y = batch[:, :-1], batch[:, 1:]
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % config["eval_every"] == 0 or step == config["total_steps"] - 1:
            val_loss = evaluate(model, val_data, config["batch_size"], config["seq_len"], device, gen_val)
            elapsed = time.time() - t0
            log["val_loss"].append([step, val_loss])
            log["wallclock"].append([step, elapsed])
            if has_trilu:
                log["trilu_params"].append([step, collect_trilu_params(model)])
            print(f"  step {step:5d}  lr={cur_lr:.5f}  train={loss.item():.4f}  val={val_loss:.4f}  t={elapsed:.1f}s")

    return log


# -----------------------------------------------------------------------------
# Main

def pick_device():
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
}


ALL_ACTIVATIONS = ["relu2", "gelu", "trilu_sym", "trilu_asym", "swiglu", "geglu", "triglu"]

# Default 6-activation sweep covering the standard/gated x ReLU-like/GELU-like/TriLU comparison.
# At 1000 steps, 3 seeds, ~40 min/run on 5060 Ti, total ~12 hr.
DEFAULT_ACTIVATIONS = ["relu2", "gelu", "trilu_asym", "swiglu", "geglu", "triglu"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default="default")
    ap.add_argument("--tiny", action="store_true", help="alias for --config tiny")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=None, help="override total_steps")
    ap.add_argument("--activations", type=str, default=",".join(DEFAULT_ACTIVATIONS),
                    help="comma-separated subset of: " + ",".join(ALL_ACTIVATIONS))
    ap.add_argument("--init", type=str, default="gelu_minimax",
                    choices=list(_INIT_VALUES))
    ap.add_argument("--data-dir", type=str, default="data/fineweb10B")
    ap.add_argument("--out", type=str, default="experiments/trilu/results.json")
    args = ap.parse_args()

    if args.tiny:
        args.config = "tiny"

    cfg_base = dict(CONFIGS[args.config])
    if args.steps is not None:
        cfg_base["total_steps"] = args.steps

    device = pick_device()
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
        cfg = dict(cfg_base, activation=act, init=args.init)
        all_results[act] = []
        for seed in range(args.seeds):
            print(f"\n=== activation={act}  seed={seed} ===")
            log = train_one(cfg, train_data, val_data, seed, device)
            all_results[act].append(log)
            # Save incrementally so a crash doesn't lose everything.
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nSaved results to {args.out}")
    print("Plot with experiments/trilu/plot.py (or your own script).")


if __name__ == "__main__":
    main()
