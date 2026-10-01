#!/usr/bin/env python3
"""Benchmark an explicit, editable list of DeepGEMM kernels.

Edit the configuration values in ``main`` and run:

    python scripts/benchmark_kernels.py

Inputs are created once and only the kernel call is timed. Results are printed
and written to a JSON file.
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"


@dataclass(frozen=True)
class BenchmarkCase:
    kernel: str
    m: int
    n: int
    k: int
    groups_or_batches: int = 1
    min_sm: int | None = None
    supported_sms: tuple[int, ...] | None = None


def unsupported_reason(case: BenchmarkCase, sm: int) -> str | None:
    if case.supported_sms is not None and sm not in case.supported_sms:
        supported = ", ".join(f"SM{value}0" for value in case.supported_sms)
        return f"requires one of: {supported}"
    if case.min_sm is not None and sm < case.min_sm:
        return f"requires SM{case.min_sm}0+"
    return None


def prioritize_repo_on_path() -> None:
    repo_root = str(REPO_ROOT)
    tests_dir = str(TESTS_DIR)
    if repo_root in sys.path:
        sys.path.remove(repo_root)
    if tests_dir in sys.path:
        sys.path.remove(tests_dir)
    sys.path[:0] = [repo_root, tests_dir]


def report_loaded_binding() -> tuple[bool, dict[str, object]]:
    """Print imported module paths and return binding metadata."""
    prioritize_repo_on_path()
    import deep_gemm
    from deep_gemm import _C

    binding_path = str(getattr(_C, "__file__", "<unknown>"))
    is_python_facade = binding_path.endswith(".py")
    package_path = str(getattr(deep_gemm, "__file__", "<unknown>"))
    print(f"Loaded DeepGEMM package: {deep_gemm.__file__}")
    print(f"Loaded binding module:   {binding_path}")
    native_modules = sorted(
        (name, str(module.__file__))
        for name, module in sys.modules.items()
        if name.startswith("deep_gemm")
        and isinstance(getattr(module, "__file__", None), str)
        and module.__file__.endswith((".so", ".pyd", ".dylib"))
    )
    if native_modules:
        for name, path in native_modules:
            print(f"Loaded native module:    {name} ({path})")
    else:
        print("Loaded native module:    none found")
    return is_python_facade, {
        "package_module": package_path,
        "binding_module": binding_path,
        "native_modules": dict(native_modules),
    }


def make_call(case: BenchmarkCase) -> Callable[[], None]:
    """Allocate inputs for one case and return only its kernel invocation."""
    prioritize_repo_on_path()

    import deep_gemm
    from deep_gemm import _C

    def get_kernel(name: str):
        kernel = getattr(deep_gemm, name, None)
        source = "deep_gemm" if kernel is not None else "deep_gemm._C"
        if kernel is None:
            kernel = getattr(_C, name)
        callable_module = getattr(kernel, "__module__", type(kernel).__module__)
        callable_name = getattr(kernel, "__name__", type(kernel).__name__)
        print(
            f"Selected binding: {name} via {source} "
            f"({callable_module}.{callable_name}; {type(kernel).__name__})"
        )
        return kernel

    m, n, k = case.m, case.n, case.k
    count = case.groups_or_batches
    if case.kernel.startswith(("bf16_gemm_", "cublaslt_gemm_")):
        layout = case.kernel.rsplit("_", 1)[-1]
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        inputs = {
            "nt": (a, b),
            "nn": (a, b.T.contiguous()),
            "tn": (a.T.contiguous(), b.T.contiguous()),
            "tt": (a.T.contiguous(), b),
        }[layout]
        kernel = get_kernel(case.kernel)
        return lambda: kernel(*inputs, d)

    if case.kernel.startswith("fp8_fp4_gemm_"):
        from deep_gemm.testing import get_arch_major
        from generators import KernelType, MajorTypeAB, generate_normal, get_ue8m0_usage

        layout = case.kernel.rsplit("_", 1)[-1]
        kernel_type = (
            KernelType.Kernel1D2D if get_arch_major() == 9 else KernelType.Kernel1D1D
        )
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        a, b, _, d, _ = generate_normal(
            m, n, k,
            MajorTypeAB.KMajor, MajorTypeAB.KMajor,
            False, torch.bfloat16, kernel_type,
            use_ue8m0=use_ue8m0,
        )
        if layout == "nn":
            b = (b[0].T, b[1].T)
        elif layout == "tn":
            a, b = (a[0].T, a[1].T), (b[0].T, b[1].T)
        elif layout == "tt":
            a = (a[0].T, a[1].T)
        kernel = get_kernel(case.kernel)
        return lambda: kernel(a, b, d, disable_ue8m0_cast=not use_ue8m0)

    from deep_gemm.testing import get_arch_major
    from generators import (
        KernelType,
        MajorTypeAB,
        QuantConfig,
        generate_k_grouped_contiguous,
        generate_m_grouped_contiguous,
        generate_m_grouped_masked,
        generate_normal,
        get_ue8m0_usage,
    )

    def set_mk_alignment(expected_m=None):
        alignment = get_kernel(
            "get_theoretical_mk_alignment_for_contiguous_layout"
        )(expected_m)
        get_kernel("set_mk_alignment_for_contiguous_layout")(alignment)

    if case.kernel.startswith("m_grouped_bf16_gemm_") and case.kernel.endswith("_contiguous"):
        layout = case.kernel.removeprefix("m_grouped_bf16_gemm_").removesuffix("_contiguous")
        set_mk_alignment()
        major_b = MajorTypeAB.KMajor if layout == "nt" else MajorTypeAB.MNMajor
        _, a, b, grouped_layout, d, _, _ = generate_m_grouped_contiguous(
            count, m, n, k, MajorTypeAB.KMajor, major_b, use_bf16=True
        )
        if not major_b.is_k_major():
            b = b.mT.contiguous()
        kernel = get_kernel(case.kernel)
        return lambda: kernel(a, b, d, grouped_layout)

    if case.kernel.startswith("m_grouped_fp8_fp4_gemm_") and case.kernel.endswith("_contiguous"):
        layout = case.kernel.removeprefix("m_grouped_fp8_fp4_gemm_").removesuffix("_contiguous")
        kernel_type = KernelType.Kernel1D2D if get_arch_major() == 9 else KernelType.Kernel1D1D
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        set_mk_alignment()
        _, a, b, grouped_layout, d, _, _ = generate_m_grouped_contiguous(
            count, m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
            use_ue8m0=use_ue8m0,
        )
        if layout == "nn":
            b = b[0].mT, b[1].mT
        kernel = get_kernel(case.kernel)
        return lambda: kernel(
            a, b, d, grouped_layout, disable_ue8m0_cast=not use_ue8m0
        )

    if case.kernel == "m_grouped_bf16_gemm_nt_masked":
        set_mk_alignment(int(m * 1.2))
        a, b, masked_m, d, _, _ = generate_m_grouped_masked(
            count, 2 * m, m, n, k, use_bf16=True
        )
        kernel = get_kernel(case.kernel)
        return lambda: kernel(a, b, d, masked_m, m)

    if case.kernel == "m_grouped_fp8_fp4_gemm_nt_masked":
        kernel_type = KernelType.Kernel1D2D if get_arch_major() == 9 else KernelType.Kernel1D1D
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        set_mk_alignment(int(m * 1.2))
        a, b, masked_m, d, _, _ = generate_m_grouped_masked(
            count, 2 * m, m, n, k, use_ue8m0=use_ue8m0
        )
        kernel = get_kernel(case.kernel)
        return lambda: kernel(
            a, b, d, masked_m, m, disable_ue8m0_cast=not use_ue8m0
        )

    if case.kernel == "k_grouped_bf16_gemm_tn_contiguous":
        _, a, b, c, d, _, grouped_layout, ks = generate_k_grouped_contiguous(
            count, m, n, MajorTypeAB.MNMajor, MajorTypeAB.MNMajor, [k] * count,
            use_bf16=True,
        )
        kernel = get_kernel(case.kernel)
        return lambda: kernel(a, b, d, ks, grouped_layout, c=c)

    if case.kernel in {
        "k_grouped_fp8_gemm_nt_contiguous",
        "k_grouped_fp8_gemm_tn_contiguous",
    }:
        layout = "nt" if "_nt_" in case.kernel else "tn"
        if layout == "tn" and get_arch_major() < 10:
            raise RuntimeError(f"{case.kernel} requires SM100+")
        use_ue8m0 = get_arch_major() >= 10
        _, a, b, c, d, _, grouped_layout, ks = generate_k_grouped_contiguous(
            count, m, n, MajorTypeAB.MNMajor, MajorTypeAB.MNMajor, [k] * count,
            use_ue8m0=use_ue8m0,
        )
        if layout == "nt":
            a = a[0].T.contiguous(), a[1].T.contiguous()
            b = b[0].T.contiguous(), b[1].T.contiguous()
        kernel = get_kernel(case.kernel)
        return lambda: kernel(a, b, d, ks, grouped_layout, c=c)

    if case.kernel == "k_grouped_fp4_gemm_nt_contiguous":
        if get_arch_major() < 10:
            raise RuntimeError(f"{case.kernel} requires SM100+")
        _, a, b, _, d, _, grouped_layout, ks = generate_k_grouped_contiguous(
            count, m, n, MajorTypeAB.KMajor, MajorTypeAB.KMajor, [k] * count,
            use_ue8m0=True, gran_k=32,
            quant_config=QuantConfig((32, 32, True, True)),
            k_alignment=256, accumulate=False, out_dtype=torch.bfloat16,
        )
        kernel = get_kernel(case.kernel)
        return lambda: kernel(a, b, d, ks, grouped_layout, recipe=(1, 1, 32))

    if case.kernel == "cublaslt_nvfp4_gemm_nt":
        if get_arch_major() < 10:
            raise RuntimeError(f"{case.kernel} requires SM100+")
        from utils import to_cublaslt_vec16_sf_layout
        a, b, _, d, _ = generate_normal(
            m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
            False, torch.bfloat16, KernelType.Kernel1D1D, use_ue8m0=True,
            quant_config=QuantConfig((32, 32, True, True)),
        )
        a = a[0], to_cublaslt_vec16_sf_layout(a[1])
        b = b[0], to_cublaslt_vec16_sf_layout(b[1])
        kernel = get_kernel(case.kernel)
        return lambda: kernel(a, b, d)

    if case.kernel == "batched_syrk":
        a = torch.randn(count, m, k, device="cuda", dtype=torch.bfloat16)
        d = torch.empty(count, m, m, device="cuda", dtype=a.dtype)
        kernel = get_kernel(case.kernel)
        return lambda: kernel(a, d)

    if case.kernel == "batched_symm":
        a = torch.randn(count, m, m, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(count, m, n, device="cuda", dtype=torch.bfloat16)
        d = torch.empty(count, m, n, device="cuda", dtype=a.dtype)
        kernel = get_kernel(case.kernel)
        return lambda: kernel(a, b, d)

    raise ValueError(f"no input builder for kernel: {case.kernel}")


def benchmark(call: Callable[[], None], warmups: int, repetitions: int) -> float:
    for _ in range(warmups):
        call()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repetitions):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3


def verify_runtime_dispatch(call: Callable[[], None], case: BenchmarkCase,
                            require_torch_dispatch: bool) -> str:
    """Profile one untimed call to see whether this API reached torch.ops."""
    op_name = f"deep_gemm::{case.kernel}"
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        call()
    torch.cuda.synchronize()
    recorded = {event.key for event in profile.key_averages()}
    if op_name in recorded:
        print(f"Runtime path check: confirmed dispatcher call {op_name}")
        return "torch dispatcher confirmed"

    if require_torch_dispatch:
        raise RuntimeError(
            f"{case.kernel} is exported through the Python _C facade, but the "
            f"profiler did not observe the expected dispatcher call {op_name}. "
            "Check the Python wrapper and registered operator name."
        )
    print(
        f"Runtime path check: no {op_name} dispatcher event observed; "
        "this is consistent with a direct native-extension binding."
    )
    return "native extension; dispatcher event not observed"


def main() -> None:
    random.seed(0)

    # Edit these values to change the benchmark. To add another case, copy one
    # line and change the kernel name and/or dimensions.
    dense_m, dense_n, dense_k = 128, 4096, 7168
    m_grouped_groups = 32
    m_grouped_m, m_grouped_n, m_grouped_k = 192, 6144, 7168
    k_grouped_groups = 8
    k_grouped_m, k_grouped_n, k_grouped_k = 768, 2048, 256
    nvfp4_m, nvfp4_n, nvfp4_k = 128, 4096, 7168
    syrk_batch, syrk_m, syrk_k = 16, 576, 5120
    symm_batch, symm_m, symm_n = 16, 576, 5120
    cases = [
        BenchmarkCase("bf16_gemm_nt", dense_m, dense_n, dense_k),
        BenchmarkCase("bf16_gemm_nn", dense_m, dense_n, dense_k),
        BenchmarkCase("bf16_gemm_tn", dense_m, dense_n, dense_k),
        BenchmarkCase("bf16_gemm_tt", dense_m, dense_n, dense_k),
        BenchmarkCase("cublaslt_gemm_nt", dense_m, dense_n, dense_k),
        BenchmarkCase("cublaslt_gemm_nn", dense_m, dense_n, dense_k),
        BenchmarkCase("cublaslt_gemm_tn", dense_m, dense_n, dense_k),
        BenchmarkCase("cublaslt_gemm_tt", dense_m, dense_n, dense_k),
        BenchmarkCase("fp8_fp4_gemm_nt", dense_m, dense_n, dense_k),
        BenchmarkCase("fp8_fp4_gemm_nn", dense_m, dense_n, dense_k),
        BenchmarkCase("fp8_fp4_gemm_tn", dense_m, dense_n, dense_k),
        BenchmarkCase("fp8_fp4_gemm_tt", dense_m, dense_n, dense_k),
        BenchmarkCase("m_grouped_fp8_fp4_gemm_nt_contiguous", m_grouped_m, m_grouped_n, m_grouped_k, m_grouped_groups),
        BenchmarkCase("m_grouped_fp8_fp4_gemm_nn_contiguous", m_grouped_m, m_grouped_n, m_grouped_k, m_grouped_groups),
        BenchmarkCase("m_grouped_fp8_fp4_gemm_nt_masked", m_grouped_m, m_grouped_n, m_grouped_k, m_grouped_groups),
        BenchmarkCase(
            "k_grouped_fp8_gemm_nt_contiguous", k_grouped_m, k_grouped_n, k_grouped_k, k_grouped_groups,
            supported_sms=(9, 12),
        ),
        BenchmarkCase(
            "k_grouped_fp8_gemm_tn_contiguous", k_grouped_m, k_grouped_n, k_grouped_k, k_grouped_groups,
            supported_sms=(10, 12),
        ),
        BenchmarkCase(
            "k_grouped_fp4_gemm_nt_contiguous", k_grouped_m, k_grouped_n, k_grouped_k, k_grouped_groups,
            supported_sms=(10,),
        ),
        BenchmarkCase("m_grouped_bf16_gemm_nt_contiguous", m_grouped_m, m_grouped_n, m_grouped_k, m_grouped_groups),
        BenchmarkCase("m_grouped_bf16_gemm_nn_contiguous", m_grouped_m, m_grouped_n, m_grouped_k, m_grouped_groups),
        BenchmarkCase("m_grouped_bf16_gemm_nt_masked", m_grouped_m, m_grouped_n, m_grouped_k, m_grouped_groups),
        BenchmarkCase("k_grouped_bf16_gemm_tn_contiguous", k_grouped_m, k_grouped_n, k_grouped_k, k_grouped_groups),
        BenchmarkCase("cublaslt_nvfp4_gemm_nt", nvfp4_m, nvfp4_n, nvfp4_k, min_sm=10),
        BenchmarkCase("batched_syrk", syrk_m, syrk_m, syrk_k, syrk_batch),
        BenchmarkCase("batched_symm", symm_m, symm_n, symm_m, symm_batch),
    ]
    warmups = 10
    repetitions = 1000
    FILE_INDEX = 1 
    output_file = REPO_ROOT / f"kernel_benchmark_results_torch.json"   # _no_gil{FILE_INDEX}.json"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to benchmark DeepGEMM")

    is_python_facade, binding_info = report_loaded_binding()
    sm = torch.cuda.get_device_capability()[0]
    kernel_results = []
    for case in cases:
        reason = unsupported_reason(case, sm)
        if reason is not None:
            result = {
                **asdict(case),
                "status": "skipped",
                "reason": reason,
            }
            kernel_results.append(result)
            print(f"skip {case.kernel:44} {reason} (device is SM{sm}0)")
            continue

        call = make_call(case)
        runtime_binding_path = verify_runtime_dispatch(call, case, is_python_facade)
        total_wall_milliseconds = benchmark(call, warmups, repetitions)
        wall_milliseconds_per_call = total_wall_milliseconds / repetitions
        tflops = (
            2 * case.m * case.n * case.k * case.groups_or_batches
        ) / (wall_milliseconds_per_call * 1e9)
        result = {
            **asdict(case),
            "status": "passed",
            "runtime binding path": runtime_binding_path,
            "total wall milliseconds": total_wall_milliseconds,
            "wall milliseconds per call": wall_milliseconds_per_call,
            "effective tflops": tflops,
        }
        kernel_results.append(result)
        print(
            f"{case.kernel:24} m={case.m:5} n={case.n:5} k={case.k:5}  "
            f"{wall_milliseconds_per_call:8.3f} ms/call  "
            f"{total_wall_milliseconds:8.3f} ms total  "
            f"{tflops:8.2f} effective TFLOP/s"
        )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "binding": binding_info,
        "warmups": warmups,
        "repetitions": repetitions,
        "results": kernel_results,
    }
    output_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved results to {output_file}")


if __name__ == "__main__":
    main()
