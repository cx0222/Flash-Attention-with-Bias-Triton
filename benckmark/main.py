"""
Testing correctness and benchmarking

Refactor from the original project from [https://github.com/pengzhangzhi]

@author: Zhangzhi Peng, Xuan Chen
"""

import math

import torch
import triton

from config import benchmark_configs
from attention_func import (
    attention_triton,
    attention_torch,
    attention_torch_compile,
    attention_sdpa,
    attention_xformers
)


def test_attention_correctness():
    """
    Test the correctness of the Triton FlashAttention implementation against PyTorch.
    Reports only the max absolute error for the forward pass and gradients.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")

    # Fixed sequence length
    batch, seqlen, nheads, d = 1, 2 ** 10, 2, 32
    softmax_scale = 1.0 / math.sqrt(d)
    causal = False

    # Initialize input tensors
    q, k, v = [
        torch.randn(batch, seqlen, nheads, d, device=device, dtype=torch.float16, requires_grad=True)
        for _ in range(3)
    ]
    bias = torch.randn(batch, nheads, seqlen, seqlen, device=device, dtype=torch.float16, requires_grad=True)

    # Compute outputs
    o_flash = attention_triton(q, k, v, bias, causal, softmax_scale)
    o_torch = attention_torch(q, k, v, softmax_scale, bias)
    o_torch_compile = attention_torch_compile(q, k, v, softmax_scale, bias)
    o_xformers = attention_xformers(q, k, v, softmax_scale, bias)
    o_sdpa = attention_sdpa(q, k, v, softmax_scale, bias)

    # Compute max forward errors
    max_forward_error_torch = torch.abs(o_flash - o_torch).max().item()
    max_forward_error_torch_compile = torch.abs(o_flash - o_torch_compile).max().item()
    max_forward_error_xformers = torch.abs(o_flash - o_xformers).max().item()
    max_forward_error_sdpa = torch.abs(o_flash - o_sdpa).max().item()

    # Compute loss and backpropagate for flash attention
    loss_flash = (0 - o_flash.sum())
    loss_flash.backward()
    dq_flash, dk_flash, dv_flash, dbias_flash = [x.grad.clone() for x in [q, k, v, bias]]

    # Reset gradients
    for x in [q, k, v, bias]:
        x.grad = None

    # Compute loss and backpropagate for torch attention
    loss_torch = (0 - o_torch.sum())
    loss_torch.backward()
    dq_torch, dk_torch, dv_torch, dbias_torch = [x.grad.clone() for x in [q, k, v, bias]]

    # Reset gradients
    for x in [q, k, v, bias]:
        x.grad = None

    # Compute loss and backpropagate for torch compile attention
    loss_torch_compile = (0 - o_torch_compile.sum())
    loss_torch_compile.backward()
    dq_torch_compile, dk_torch_compile, dv_torch_compile, dbias_torch_compile \
        = [x.grad.clone() for x in [q, k, v, bias]]

    # Reset gradients
    for x in [q, k, v, bias]:
        x.grad = None

    # Compute loss and backpropagate for xformers attention
    loss_xformers = (0 - o_xformers.sum())
    loss_xformers.backward()
    dq_torch_xformers, dk_torch_xformers, dv_torch_xformers, dbias_torch_xformers \
        = [x.grad.clone() for x in [q, k, v, bias]]

    # Reset gradients
    for x in [q, k, v, bias]:
        x.grad = None

    # Compute loss and backpropagate for SDPA
    loss_sdpa = (0 - o_sdpa.sum())
    loss_sdpa.backward()
    dq_sdpa, dk_sdpa, dv_sdpa, dbias_sdpa = [x.grad.clone() for x in [q, k, v, bias]]

    # Compute max gradient errors against torch implementation
    max_grad_errors_torch = {
        "q": torch.abs(dq_flash - dq_torch).max().item(),
        "k": torch.abs(dk_flash - dk_torch).max().item(),
        "v": torch.abs(dv_flash - dv_torch).max().item(),
        "bias": torch.abs(dbias_flash - dbias_torch).max().item(),
    }

    # Compute max gradient errors against torch compile implementation
    max_grad_errors_torch_compile = {
        "q": torch.abs(dq_flash - dq_torch_compile).max().item(),
        "k": torch.abs(dk_flash - dk_torch_compile).max().item(),
        "v": torch.abs(dv_flash - dv_torch_compile).max().item(),
        "bias": torch.abs(dbias_flash - dbias_torch_compile).max().item(),
    }

    # Compute max gradient errors against xformers implementation
    max_grad_errors_xformers = {
        "q": torch.abs(dq_flash - dq_torch_xformers).max().item(),
        "k": torch.abs(dk_flash - dk_torch_xformers).max().item(),
        "v": torch.abs(dv_flash - dv_torch_xformers).max().item(),
        "bias": torch.abs(dbias_flash - dbias_torch_xformers).max().item(),
    }

    # Compute max gradient errors against SDPA implementation
    max_grad_errors_sdpa = {
        "q": torch.abs(dq_flash - dq_sdpa).max().item(),
        "k": torch.abs(dk_flash - dk_sdpa).max().item(),
        "v": torch.abs(dv_flash - dv_sdpa).max().item(),
        "bias": torch.abs(dbias_flash - dbias_sdpa).max().item(),
    }

    # Print max errors
    print(f"✅ Max Forward Error vs Torch: {max_forward_error_torch:.6e}")
    print(f"✅ Max Forward Error vs Torch Compile: {max_forward_error_torch_compile:.6e}")
    print(f"✅ Max Forward Error vs xFormers: {max_forward_error_xformers:.6e}")
    print(f"✅ Max Forward Error vs SDPA: {max_forward_error_sdpa:.6e}")

    print(f"✅ Max Gradient Errors vs Torch:")
    for param, err in max_grad_errors_torch.items():
        print(f"   {param.upper()} : {err:.6e}")

    print(f"✅ Max Gradient Errors vs Torch Compile:")
    for param, err in max_grad_errors_torch_compile.items():
        print(f"   {param.upper()} : {err:.6e}")

    print(f"✅ Max Gradient Errors vs xFormers:")
    for param, err in max_grad_errors_xformers.items():
        print(f"   {param.upper()} : {err:.6e}")

    print(f"✅ Max Gradient Errors vs SDPA:")
    for param, err in max_grad_errors_sdpa.items():
        print(f"   {param.upper()} : {err:.6e}")


@triton.testing.perf_report(benchmark_configs)
def bench_attention_comparison(BATCH, H, len, HEAD_DIM, mode, provider, device="cuda"):
    dtype = torch.float16

    # Initialize tensors with same shape ordering as test_attention_correctness
    q = torch.randn((BATCH, len, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, len, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, len, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    bias = torch.randn((BATCH, H, len, len), dtype=dtype, device=device, requires_grad=True)

    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    def measure_max_error():
        # For forward pass
        if mode == "fwd":
            out_triton = attention_triton(q, k, v, bias, False, sm_scale)
            out_torch = attention_torch(q, k, v, sm_scale, bias)
            return torch.abs(out_triton - out_torch).max().item()
        # For backward pass
        else:
            # Run triton backward
            out_triton = attention_triton(q, k, v, bias, False, sm_scale)
            loss_triton = (0 - out_triton.sum())
            loss_triton.backward()
            dbias_triton = bias.grad.clone()
            bias.grad = None

            # Run torch backward
            out_torch = attention_torch(q, k, v, sm_scale, bias)
            loss_torch = (0 - out_torch.sum())
            loss_torch.backward()
            dbias_torch = bias.grad.clone()
            bias.grad = None

            # Compare gradients
            return torch.abs(dbias_triton - dbias_torch).max().item()

    def forward_backward_triton():
        out = attention_triton(q, k, v, bias, False, sm_scale)
        loss = (0 - out.sum())
        loss.backward()
        torch.cuda.synchronize()
        # Reset gradients for next iteration
        q.grad = None
        k.grad = None
        v.grad = None
        bias.grad = None

    def forward_backward_torch():
        out = attention_torch(q, k, v, sm_scale, bias)
        loss = (0 - out.sum())
        loss.backward()
        torch.cuda.synchronize()
        # Reset gradients for next iteration
        q.grad = None
        k.grad = None
        v.grad = None
        bias.grad = None

    def forward_backward_torch_compile():
        out = attention_torch_compile(q, k, v, sm_scale, bias)
        loss = (0 - out.sum())
        loss.backward()
        torch.cuda.synchronize()
        # Reset gradients for next iteration
        q.grad = None
        k.grad = None
        v.grad = None
        bias.grad = None

    def forward_backward_xformers():
        out = attention_xformers(q, k, v, sm_scale, bias)
        loss = (0 - out.sum())
        loss.backward()
        torch.cuda.synchronize()
        # Reset gradients for next iteration
        q.grad = None
        k.grad = None
        v.grad = None
        bias.grad = None

    def forward_backward_sdpa():
        out = attention_sdpa(q, k, v, sm_scale, bias)
        loss = (0 - out.sum())
        loss.backward()
        torch.cuda.synchronize()
        # Reset gradients for next iteration
        q.grad = None
        k.grad = None
        v.grad = None
        bias.grad = None

    if mode == "fwd":
        if provider == "triton-attn-bias":
            fn = lambda: attention_triton(q, k, v, bias, False, sm_scale)
        elif provider == "torch-attn-bias":
            fn = lambda: attention_torch(q, k, v, sm_scale, bias)
        elif provider == "torch-compile-attn-bias":
            fn = lambda: attention_torch_compile(q, k, v, sm_scale, bias)
        elif provider == "xformers-attn-bias":
            fn = lambda: attention_xformers(q, k, v, sm_scale, bias)
        elif provider == "torch-sdpa":
            fn = lambda: attention_sdpa(q, k, v, sm_scale, bias)
        else:
            raise Exception("Invalid provider \"{}\"".format(provider))
    elif mode == "bwd":
        if provider == "triton-attn-bias":
            fn = forward_backward_triton
        elif provider == "torch-attn-bias":
            fn = forward_backward_torch
        elif provider == "torch-compile-attn-bias":
            fn = forward_backward_torch_compile
        elif provider == "xformers-attn-bias":
            fn = forward_backward_xformers
        elif provider == "torch-sdpa":
            fn = forward_backward_sdpa
        else:
            raise Exception("Invalid provider \"{}\"".format(provider))
    else:
        raise Exception("Invalid mode \"{}\"".format(mode))

    # Run benchmark
    ms = triton.testing.do_bench(fn)

    # Calculate FLOPS
    # Forward pass:
    # FLOPS = 2 * QK + 2 * PV multiplications + bias addition
    # QK matmul: 2 * B * H * N * N * D
    # PV matmul: 2 * B * H * N * N * D
    # Bias addition: B * H * N * N
    flops_qk = 2.0 * BATCH * H * len * len * HEAD_DIM  # QK matmul
    flops_pv = 2.0 * BATCH * H * len * len * HEAD_DIM  # PV matmul
    flops_bias = BATCH * H * len * len  # Bias addition
    total_flops_fwd = flops_qk + flops_pv + flops_bias

    # Backward pass has roughly 2x the FLOPS of forward pass
    total_flops = total_flops_fwd * (3 if mode == "bwd" else 1)

    # Calculate TFLOPS
    tflops = total_flops * 1e-12 / (ms * 1e-3)

    # Measure max error (only for triton implementation)
    max_error = measure_max_error() if provider == "triton-attn-bias" else 0.0

    # Return TFLOPS for plotting
    return tflops


if __name__ == "__main__":
    print("Running correctness tests...")
    print("-" * 50)
    test_attention_correctness()

    print("\nRunning performance benchmarks...")
    print("-" * 50)
    df = bench_attention_comparison.run(save_path=".", print_data=True, return_df=True)
