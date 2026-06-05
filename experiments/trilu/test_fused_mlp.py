"""Correctness test for the relu2_s1 fused-MLP kernel edit.

The real stack (train_gpt.py + triton_kernels.py) computes the MLP with a fused
Triton kernel `linear_relu_square_kernel` that hardcodes  post = relu(pre)^2  and,
in its backward pass,  dpre = grad * 2*relu(pre).  Switching the incumbent to
relu2_s1  ( act(z) = relu(z)^2 + relu(z) )  is a localized edit:

    forward :  post = relu(pre)^2 + relu(pre)
    backward:  dpre = grad * ( 2*relu(pre) + 1[pre>0] )

This file validates that edit two ways:

  Part A (CPU, fp64, default): a pure-torch autograd.Function that implements the
      SAME manual backward algebra as the kernel (dW2, dpre, dW1, dx), checked with
      torch.autograd.gradcheck and against autograd through an eager reference. This
      proves the math going into the kernel -- no GPU, cannot perturb the grid.

  Part B (GPU, --gpu): the actual modified Triton kernel (ACT constexpr flag, relu2
      vs relu2_s1), forward + backward vs an fp32 eager reference. This proves the
      real kernel that will run on the H100. Run when the GPU is free.

Usage:
    python experiments/trilu/test_fused_mlp.py          # Part A only (CPU)
    python experiments/trilu/test_fused_mlp.py --gpu     # Part A + Part B (Triton)
"""
import sys
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Activation algebra (single source of truth for the references)
#   relu2    : f(z) = relu(z)^2            f'(z) = 2*relu(z)
#   relu2_s1 : f(z) = relu(z)^2 + relu(z)  f'(z) = 2*relu(z) + 1[z>0]
def act_fwd(z, s1: bool):
    r = F.relu(z)
    return r * r + r if s1 else r * r

def act_dfwd(z, s1: bool):           # f'(z)
    r = F.relu(z)
    g = 2 * r
    if s1:
        g = g + (z > 0).to(z.dtype)
    return g


# -----------------------------------------------------------------------------
# Part A: pure-torch Function with the kernel's MANUAL backward algebra.
# MLP shapes match the kernel:  x (M,K)  W1 (N,K)  W2 (N,K)
#   pre  = x @ W1.T          (M,N)
#   post = act(pre)          (M,N)
#   out  = post @ W2         (M,K)
class ManualMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W1, W2, s1: bool):
        pre = x @ W1.T
        post = act_fwd(pre, s1)
        out = post @ W2
        ctx.save_for_backward(x, W1, W2, pre, post)
        ctx.s1 = s1
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, W1, W2, pre, post = ctx.saved_tensors
        s1 = ctx.s1
        dpost = grad_out @ W2.T               # (M,N)
        dpre = dpost * act_dfwd(pre, s1)      # (M,N)  <-- the kernel's elementwise step
        dW2 = post.T @ grad_out               # (N,K)
        dW1 = dpre.T @ x                      # (N,K)
        dx = dpre @ W1                        # (M,K)
        return dx, dW1, dW2, None


def eager_mlp(x, W1, W2, s1: bool):
    """Reference: plain autograd, no manual backward."""
    return act_fwd(x @ W1.T, s1) @ W2


def part_a():
    torch.manual_seed(0)
    M, K, N = 4, 6, 8     # tiny; gradcheck is O(n^2) in input size
    ok = True
    for s1 in (False, True):
        name = "relu2_s1" if s1 else "relu2"
        x = torch.randn(M, K, dtype=torch.float64, requires_grad=True)
        W1 = torch.randn(N, K, dtype=torch.float64, requires_grad=True)
        W2 = torch.randn(N, K, dtype=torch.float64, requires_grad=True)

        # (1) gradcheck the manual backward
        gc = torch.autograd.gradcheck(
            lambda a, b, c: ManualMLP.apply(a, b, c, s1), (x, W1, W2),
            eps=1e-6, atol=1e-6, rtol=1e-4)

        # (2) manual grads vs autograd-through-eager (independent reference)
        xs = [t.detach().clone().requires_grad_(True) for t in (x, W1, W2)]
        ManualMLP.apply(*xs, s1).sum().backward()
        ys = [t.detach().clone().requires_grad_(True) for t in (x, W1, W2)]
        eager_mlp(*ys, s1).sum().backward()
        gerr = max((a.grad - b.grad).abs().max().item() for a, b in zip(xs, ys))
        ferr = (ManualMLP.apply(*xs, s1) - eager_mlp(*ys, s1)).abs().max().item()

        passed = gc and gerr < 1e-9 and ferr < 1e-9
        ok = ok and passed
        print(f"  [A] {name:9s} gradcheck={gc}  fwd_err={ferr:.2e}  grad_err={gerr:.2e}  "
              f"-> {'PASS' if passed else 'FAIL'}")
    return ok


# -----------------------------------------------------------------------------
# Part B: the actual modified Triton kernel (ACT constexpr).  Imports triton only
# when --gpu is requested so Part A stays dependency-free.
def part_b():
    import triton
    import triton.language as tl
    from triton.tools.tensor_descriptor import TensorDescriptor

    @triton.jit
    def linear_act_square_kernel(a_desc, b_desc, c_desc, aux_desc,
                                 M, N, K,
                                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                 BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                                 NUM_SMS: tl.constexpr, FORWARD: tl.constexpr,
                                 ACT: tl.constexpr):           # 0=relu2, 1=relu2_s1
        dtype = tl.bfloat16
        start_pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
        num_tiles = num_pid_m * num_pid_n
        tile_id_c = start_pid - NUM_SMS

        for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for ki in range(k_tiles):
                offs_k = ki * BLOCK_SIZE_K
                a = a_desc.load([offs_am, offs_k])
                b = b_desc.load([offs_bn, offs_k])
                accumulator = tl.dot(a, b.T, accumulator)

            tile_id_c += NUM_SMS
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
            offs_am_c = pid_m * BLOCK_SIZE_M
            offs_bn_c = pid_n * BLOCK_SIZE_N
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)

            c0 = acc0.to(dtype)
            if not FORWARD:
                c0_pre = aux_desc.load([offs_am_c, offs_bn_c])
                r0 = tl.where(c0_pre > 0, c0_pre, 0)
                if ACT == 0:
                    c0 = 2 * c0 * r0
                else:
                    c0 = c0 * tl.where(c0_pre > 0, 2 * c0_pre + 1, 0.0)
            c_desc.store([offs_am_c, offs_bn_c], c0)
            if FORWARD:
                p0 = tl.maximum(c0, 0)
                c0_post = p0 * p0 + p0 if ACT == 1 else p0 * p0
                aux_desc.store([offs_am_c, offs_bn_c], c0_post)

            c1 = acc1.to(dtype)
            if not FORWARD:
                c1_pre = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
                if ACT == 0:
                    c1 = 2 * c1 * tl.where(c1_pre > 0, c1_pre, 0)
                else:
                    c1 = c1 * tl.where(c1_pre > 0, 2 * c1_pre + 1, 0.0)
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
            if FORWARD:
                p1 = tl.maximum(c1, 0)
                c1_post = p1 * p1 + p1 if ACT == 1 else p1 * p1
                aux_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1_post)

    def linear_act_square(a, b, act, aux=None):
        M, K = a.shape
        N, K = b.shape
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        FORWARD = aux is None
        if FORWARD:
            aux = torch.empty((M, N), device=a.device, dtype=a.dtype)
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        # Production (H100) tiles are BM=128, BN=256, BK=64, num_stages=4 (~180KB smem).
        # Consumer cards (sm_120, ~101KB smem) need smaller tiles; the relu2_s1 math is
        # tile-size independent, so this still validates the kernel edit. Override via env.
        import os
        BM = int(os.environ.get("TEST_BM", 64))
        BN = int(os.environ.get("TEST_BN", 128))
        BK = int(os.environ.get("TEST_BK", 64))
        STAGES = int(os.environ.get("TEST_STAGES", 2))
        a_desc = TensorDescriptor.from_tensor(a, [BM, BK])
        b_desc = TensorDescriptor.from_tensor(b, [BN, BK])
        c_desc = TensorDescriptor.from_tensor(c, [BM, BN // 2])
        aux_desc = TensorDescriptor.from_tensor(aux, [BM, BN // 2])
        grid = lambda META: (min(NUM_SMS, triton.cdiv(M, BM) * triton.cdiv(N, BN)),)
        linear_act_square_kernel[grid](a_desc, b_desc, c_desc, aux_desc, M, N, K,
                                       BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
                                       GROUP_SIZE_M=1, NUM_SMS=NUM_SMS, FORWARD=FORWARD,
                                       ACT=act, num_stages=STAGES, num_warps=8)
        return (c, aux) if FORWARD else c

    class FusedActMLP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, W1, W2, act):
            pre, post = linear_act_square(x.view(-1, x.shape[-1]), W1, act)
            out = post @ W2
            ctx.save_for_backward(x, W1, W2, pre, post)
            ctx.act = act
            return out.view(x.shape[:-1] + (W2.shape[-1],))
        @staticmethod
        def backward(ctx, grad_out):
            x, W1, W2, pre, post = ctx.saved_tensors
            dW2 = post.T @ grad_out.view(-1, grad_out.shape[-1])
            dpre = linear_act_square(grad_out.view(-1, grad_out.shape[-1]), W2, ctx.act, aux=pre)
            dW1 = dpre.T @ x.view(-1, x.shape[-1])
            dx = dpre @ W1
            return dx.view(x.shape), dW1, dW2, None

    torch.manual_seed(0)
    dev = "cuda"
    M, K, N = 512, 768, 2048          # realistic-ish MLP shape
    ok = True
    for act, s1 in ((0, False), (1, True)):
        name = "relu2_s1" if s1 else "relu2"
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16).requires_grad_(True)
        W1 = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) / K**0.5).requires_grad_(True)
        W2 = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) / N**0.5).requires_grad_(True)
        out = FusedActMLP.apply(x, W1, W2, act)
        g = torch.randn_like(out)
        out.backward(g)
        fused = (out.float(), x.grad.float(), W1.grad.float(), W2.grad.float())

        # fp32 eager reference
        xr = x.detach().float().requires_grad_(True)
        W1r = W1.detach().float().requires_grad_(True)
        W2r = W2.detach().float().requires_grad_(True)
        outr = act_fwd(xr @ W1r.T, s1) @ W2r
        outr.backward(g.float())
        ref = (outr, xr.grad, W1r.grad, W2r.grad)

        def relerr(a, b):
            return ((a - b).norm() / (b.norm() + 1e-6)).item()
        errs = [relerr(a, b) for a, b in zip(fused, ref)]
        passed = max(errs) < 3e-2     # bf16 fused vs fp32 ref
        ok = ok and passed
        print(f"  [B] {name:9s} relerr out/dx/dW1/dW2 = "
              f"{errs[0]:.2e}/{errs[1]:.2e}/{errs[2]:.2e}/{errs[3]:.2e} -> {'PASS' if passed else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("Part A (CPU, fp64): manual-backward algebra vs gradcheck + eager autograd")
    a_ok = part_a()
    b_ok = True
    if "--gpu" in sys.argv:
        print("Part B (GPU, Triton): modified fused kernel vs fp32 eager reference")
        b_ok = part_b()
    else:
        print("Part B (GPU) skipped -- pass --gpu to run the Triton kernel test.")
    print("RESULT:", "ALL PASS" if (a_ok and b_ok) else "FAILURE")
    sys.exit(0 if (a_ok and b_ok) else 1)
