# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import importlib
import json
import time
from collections.abc import Callable

import torch

import vllm.envs as envs
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import per_block_cast_to_fp8


def measure_wall_time(
    fn: Callable[[], None], warmups: int, repetitions: int
) -> float:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repetitions):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / repetitions


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure repeated DeepGEMM dispatches in eager mode and CUDA graph "
            "replay. The difference exposes per-call host dispatch overhead."
        )
    )
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument("--calls-per-forward", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--skip-cudagraph", action="store_true")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not current_platform.support_deep_gemm():
        raise RuntimeError("This benchmark requires CUDA with DeepGEMM support")

    deep_gemm = importlib.import_module("vllm.third_party.deep_gemm")
    binding = importlib.import_module("vllm.third_party.deep_gemm._C")
    package_file = str(deep_gemm.__file__)
    binding_file = str(binding.__file__)
    print(f"DeepGEMM package: {package_file}")
    print(f"DeepGEMM binding: {binding_file}")

    if current_platform.is_arch_support_pdl():
        deep_gemm.set_pdl(True)

    if args.m <= 0 or args.n <= 0 or args.k <= 0:
        raise ValueError("M, N, and K must be positive")
    if args.n % 64 != 0 or args.k % 128 != 0:
        raise ValueError("DeepGEMM requires N divisible by 64 and K by 128")
    if args.calls_per_forward <= 0:
        raise ValueError("--calls-per-forward must be positive")

    use_ue8m0 = envs.VLLM_USE_DEEP_GEMM_E8M0
    a = torch.randn((args.m, args.k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((args.n, args.k), device="cuda", dtype=torch.bfloat16)
    a_fp8, a_scale = per_token_group_quant_fp8(
        a,
        group_size=128,
        column_major_scales=True,
        tma_aligned_scales=True,
        use_ue8m0=use_ue8m0,
    )
    b_fp8, b_scale = per_block_cast_to_fp8(
        b, block_size=[128, 128], use_ue8m0=use_ue8m0
    )
    output = torch.empty(
        (args.m, args.n), device="cuda", dtype=torch.bfloat16
    )

    def one_call() -> None:
        deep_gemm.fp8_fp4_gemm_tn(
            (a_fp8.t(), a_scale.t()),
            (b_fp8.t(), b_scale.t()),
            output,
            disable_ue8m0_cast=not use_ue8m0,
        )

    def repeated_calls() -> None:
        for _ in range(args.calls_per_forward):
            one_call()

    eager_ms = measure_wall_time(
        repeated_calls, args.warmups, args.repetitions
    )
    results: dict[str, object] = {
        "device": torch.cuda.get_device_name(),
        "binding": {
            "package_file": package_file,
            "binding_file": binding_file,
        },
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "calls_per_forward": args.calls_per_forward,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "eager_ms_per_forward": eager_ms,
        "eager_ms_per_call": eager_ms / args.calls_per_forward,
    }

    if not args.skip_cudagraph:
        graph = torch.cuda.CUDAGraph()
        repeated_calls()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            repeated_calls()

        graph_ms = measure_wall_time(
            graph.replay, args.warmups, args.repetitions
        )
        results.update(
            {
                "cudagraph_ms_per_forward": graph_ms,
                "cudagraph_ms_per_call": graph_ms / args.calls_per_forward,
                "eager_minus_cudagraph_us_per_call": (
                    eager_ms - graph_ms
                )
                * 1000
                / args.calls_per_forward,
            }
        )

    rendered = json.dumps(results, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")


if __name__ == "__main__":
    main()


# python benchmarks/kernels/benchmark_deepgemm_dispatch.py \
#   --m 128 --n 4096 --k 7168 \
#   --calls-per-forward 200 \
#   --repetitions 20 \
#   --output deepgemm_dispatch_results.json


# python benchmarks/kernels/benchmark_deepgemm_dispatch.py --output deepgemm_dispatch_results.json
