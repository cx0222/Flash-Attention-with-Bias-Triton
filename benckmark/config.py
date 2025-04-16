"""
Configs for triton

Refactor from the original project from [https://github.com/pengzhangzhi]

@author: Zhangzhi Peng, Xuan Chen
"""

import triton

# Benchmarking configuration
BATCH, N_HEADS, HEAD_DIM = 2, 4, 32  # Aligned with test_attention_correctness

benchmark_configs = [
    triton.testing.Benchmark(
        x_names=["len"],
        x_vals=list(2 ** i for i in range(8, 15)),
        line_arg="provider",
        line_vals=[
            "triton-attn-bias",
            "torch-attn-bias",
            "torch-compile-attn-bias",
            "xformers-attn-bias",
            "torch-sdpa"
        ],
        line_names=[
            "Triton-Attn-Bias (FLOPS)",
            "PyTorch-Attn-Bias (FLOPS)",
            "PyTorch-Compile-Attn-Bias (FLOPS)",
            "xFormers-Attn-Bias (FLOPS)",
            "PyTorch-SDPA (FLOPS)"
        ],
        styles=[
            ("red", "-"),
            ("blue", "--"),
            ("cyan", "-."),
            ("orange", "-."),
            ("green", ":")
        ],
        ylabel="TFLOPS",  # Changed from Time (ms) to TFLOPS
        plot_name=f"attention-comparison-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}",
        args={
            "H": N_HEADS,
            "BATCH": BATCH,
            "HEAD_DIM": HEAD_DIM,
            "mode": mode,
        }
    )
    for mode in ["fwd", "bwd"]
]
