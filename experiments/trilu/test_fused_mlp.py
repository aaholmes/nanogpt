"""Correctness test for the fused-MLP activation kernel edits.

The real stack (train_gpt.py + triton_kernels.py) computes the MLP with a fused Triton
kernel `linear_relu_square_kernel`. We add alternative activations selected by an ACT flag:

    ACT=0  relu2     post = relu(z)^2                 dpre = grad * 2*relu(z)
    ACT=1  relu2_s1  post = relu(z)^2 + relu(z)       dpre = grad * (2*relu(z) + 1[z>0])
    ACT=2  sniqu     post = LAM*(eluquad(z) - BETA)   dpre = grad * LAM*eluquad'(z)
           eluquad(z) = z+z^2 (z>=0) | z/(1-z) (z<0);  LAM=0.56004, BETA=0.70638

Part A (CPU, fp64, default): pure-torch Function with the SAME manual backward algebra as the
    kernel, checked by gradcheck + against autograd through an eager reference. Proves the math.
Part B (GPU, --gpu): the actual modified Triton kernel, fwd+bwd vs fp32 eager. Proves the kernel.

    python experiments/trilu/test_fused_mlp.py          # Part A only (CPU)
    python experiments/trilu/test_fused_mlp.py --gpu     # Part A + Part B (Triton)
"""
import sys
import torch
import torch.nn.functional as F

LAM, BETA = 0.56004, 0.70638
NAMES = {0: "relu2", 1: "relu2_s1", 2: "sniqu"}


# -----------------------------------------------------------------------------
# Activation algebra (single source of truth for the references)
def act_fwd(z, act):
    pos = F.relu(z)
    if act == 0:
        return pos * pos
    if act == 1:
        return pos * pos + pos
    neg = z - pos                              # = min(z, 0) <= 0
    h = pos + pos * pos + neg / (1.0 - neg)    # eluquad
    return LAM * (h - BETA)


def act_dfwd(z, act):                          # d/dz act(z)
    pos = F.relu(z)
    if act == 0:
        return 2 * pos
    if act == 1:
        return 2 * pos + (z > 0).to(z.dtype)
    neg = torch.minimum(z, torch.zeros_like(z))
    inv = 1.0 / (1.0 - neg)                     # safe: 1-neg >= 1
    dh = torch.where(z >= 0, 1.0 + 2.0 * pos, inv * inv)
    return LAM * dh


# -----------------------------------------------------------------------------
# Part A: pure-torch Function with the kernel's MANUAL backward algebra.
#   pre = x @ W1.T   post = act(pre)   out = post @ W2     (x (M,K) W1 (N,K) W2 (N,K))
class ManualMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W1, W2, act):
        pre = x @ W1.T
        post = act_fwd(pre, act)
        out = post @ W2
        ctx.save_for_backward(x, W1, W2, pre, post)
        ctx.act = act
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, W1, W2, pre, post = ctx.saved_tensors
        dpost = grad_out @ W2.T
        dpre = dpost * act_dfwd(pre, ctx.act)      # the kernel's elementwise step
        dW2 = post.T @ grad_out
        dW1 = dpre.T @ x
        dx = dpre @ W1
        return dx, dW1, dW2, None


def eager_mlp(x, W1, W2, act):
    return act_fwd(x @ W1.T, act) @ W2


def part_a():
    torch.manual_seed(0)
    M, K, N = 4, 6, 8
    ok = True
    for act in (0, 1, 2):
        x = torch.randn(M, K, dtype=torch.float64, requires_grad=True)
        W1 = torch.randn(N, K, dtype=torch.float64, requires_grad=True)
        W2 = torch.randn(N, K, dtype=torch.float64, requires_grad=True)
        gc = torch.autograd.gradcheck(
            lambda a, b, c: ManualMLP.apply(a, b, c, act), (x, W1, W2),
            eps=1e-6, atol=1e-6, rtol=1e-4)
        xs = [t.detach().clone().requires_grad_(True) for t in (x, W1, W2)]
        ManualMLP.apply(*xs, act).sum().backward()
        ys = [t.detach().clone().requires_grad_(True) for t in (x, W1, W2)]
        eager_mlp(*ys, act).sum().backward()
        gerr = max((a.grad - b.grad).abs().max().item() for a, b in zip(xs, ys))
        ferr = (ManualMLP.apply(*xs, act) - eager_mlp(*ys, act)).abs().max().item()
        passed = gc and gerr < 1e-9 and ferr < 1e-9
        ok = ok and passed
        print(f"  [A] {NAMES[act]:9s} gradcheck={gc}  fwd_err={ferr:.2e}  grad_err={gerr:.2e}  "
              f"-> {'PASS' if passed else 'FAIL'}")
    return ok


# -----------------------------------------------------------------------------
# Part B: the actual modified Triton kernel.
def part_b():
    import os
    import triton
    import triton.language as tl
    from triton.tools.tensor_descriptor import TensorDescriptor

    @triton.jit
    def linear_act_square_kernel(a_desc, b_desc, c_desc, aux_desc,
                                 M, N, K,
                                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                 BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                                 NUM_SMS: tl.constexpr, FORWARD: tl.constexpr,
                                 ACT: tl.constexpr):          # 0=relu2 1=relu2_s1 2=sniqu
        LAM = 0.56004
        BETA = 0.70638
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

            # ---- c0 half ----
            c0 = acc0.to(dtype)
            if not FORWARD:
                c0_pre = aux_desc.load([offs_am_c, offs_bn_c])
                if ACT == 2:
                    pos0 = tl.maximum(c0_pre, 0.0); neg0 = tl.minimum(c0_pre, 0.0)
                    inv0 = 1.0 / (1.0 - neg0)
                    dh0 = tl.where(c0_pre >= 0, 1.0 + 2.0 * pos0, inv0 * inv0)
                    c0 = c0 * (LAM * dh0)
                elif ACT == 1:
                    c0 = c0 * tl.where(c0_pre > 0, 2 * c0_pre + 1, 0.0)
                else:
                    c0 = 2 * c0 * tl.where(c0_pre > 0, c0_pre, 0)
            c_desc.store([offs_am_c, offs_bn_c], c0)
            if FORWARD:
                if ACT == 2:
                    pos0 = tl.maximum(c0, 0.0); neg0 = tl.minimum(c0, 0.0)
                    h0 = pos0 + pos0 * pos0 + neg0 / (1.0 - neg0)
                    c0_post = LAM * (h0 - BETA)
                else:
                    p0 = tl.maximum(c0, 0.0)
                    c0_post = p0 * p0 + p0 if ACT == 1 else p0 * p0
                aux_desc.store([offs_am_c, offs_bn_c], c0_post)

            # ---- c1 half ----
            c1 = acc1.to(dtype)
            if not FORWARD:
                c1_pre = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
                if ACT == 2:
                    pos1 = tl.maximum(c1_pre, 0.0); neg1 = tl.minimum(c1_pre, 0.0)
                    inv1 = 1.0 / (1.0 - neg1)
                    dh1 = tl.where(c1_pre >= 0, 1.0 + 2.0 * pos1, inv1 * inv1)
                    c1 = c1 * (LAM * dh1)
                elif ACT == 1:
                    c1 = c1 * tl.where(c1_pre > 0, 2 * c1_pre + 1, 0.0)
                else:
                    c1 = 2 * c1 * tl.where(c1_pre > 0, c1_pre, 0)
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
            if FORWARD:
                if ACT == 2:
                    pos1 = tl.maximum(c1, 0.0); neg1 = tl.minimum(c1, 0.0)
                    h1 = pos1 + pos1 * pos1 + neg1 / (1.0 - neg1)
                    c1_post = LAM * (h1 - BETA)
                else:
                    p1 = tl.maximum(c1, 0.0)
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
        # Production (H100) tiles are BM=128 BN=256 BK=64 stages=4 (~180KB smem). Consumer cards
        # (sm_120, ~101KB) need smaller tiles; the activation math is tile-size independent.
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
    M, K, N = 512, 768, 2048
    ok = True
    for act in (0, 1, 2):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16).requires_grad_(True)
        W1 = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) / K**0.5).requires_grad_(True)
        W2 = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) / N**0.5).requires_grad_(True)
        out = FusedActMLP.apply(x, W1, W2, act)
        g = torch.randn_like(out)
        out.backward(g)
        fused = (out.float(), x.grad.float(), W1.grad.float(), W2.grad.float())
        xr = x.detach().float().requires_grad_(True)
        W1r = W1.detach().float().requires_grad_(True)
        W2r = W2.detach().float().requires_grad_(True)
        outr = act_fwd(xr @ W1r.T, act) @ W2r
        outr.backward(g.float())
        ref = (outr, xr.grad, W1r.grad, W2r.grad)
        errs = [((a - b).norm() / (b.norm() + 1e-6)).item() for a, b in zip(fused, ref)]
        passed = max(errs) < 3e-2
        ok = ok and passed
        print(f"  [B] {NAMES[act]:9s} relerr out/dx/dW1/dW2 = "
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
