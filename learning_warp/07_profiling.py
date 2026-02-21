"""
Warp Profiling 工具库 —— NVTX 标记 / Nsight 集成 / 多次计时统计
================================================================

【这是一个工具库】可以在任何代码中 import 使用，不是只能单独运行。

【使用方式】

    # 方式1：import 后直接用
    from learning_warp.profiling import nvtx_range, gpu_timer, benchmark, NsightHelper

    # 方式2：直接运行看演示
    cd /home/hkang/newton_related/newton
    uv run python learning_warp/07_profiling.py

【Nsight Systems 集成】

    # 用 nsys 采集 NVTX 标记 + GPU 活动：
    nsys profile -t cuda,nvtx -o my_report uv run python my_script.py

    # 打开报告：
    nsys-ui my_report.nsys-rep

【Nsight Compute 集成】

    # 用 ncu 分析单个 kernel 的性能：
    ncu --set full -o my_kernel_report uv run python my_script.py

    # 打开报告：
    ncu-ui my_kernel_report.ncu-rep

【依赖】
    pip install nvtx    # NVTX 标记（可选，没装也能用，只是 NVTX 标记不生效）
"""

from __future__ import annotations

import contextlib
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np
import warp as wp

wp.init()

# ============================================================================
# 检测 nvtx 是否可用
# ============================================================================
try:
    import nvtx as _nvtx

    _NVTX_AVAILABLE = True
except ImportError:
    _nvtx = None
    _NVTX_AVAILABLE = False


# ============================================================================
# 核心工具1：NVTX Range 标记（with 语法）
# ============================================================================

class nvtx_range:
    """NVTX 范围标记，用于 Nsight Systems 中标记代码段。

    用法：
        with nvtx_range("my_section", color="green"):
            wp.launch(my_kernel, ...)

        # 嵌套使用
        with nvtx_range("frame"):
            with nvtx_range("physics", color="blue"):
                solver.step(...)
            with nvtx_range("render", color="red"):
                viewer.render(...)

    颜色支持：
        "red", "green", "blue", "yellow", "cyan", "magenta", "white"
        或 ARGB 整数值如 0xFF00FF00
    """

    def __init__(self, name: str, color: str | int = "green",
                 sync_before: bool = False, sync_after: bool = False):
        self.name = name
        self.color = color
        self.sync_before = sync_before
        self.sync_after = sync_after
        self._range_id = None

    def __enter__(self):
        if self.sync_before:
            wp.synchronize()
        if _NVTX_AVAILABLE:
            self._range_id = _nvtx.start_range(self.name, color=self.color)
        return self

    def __exit__(self, *args):
        if self.sync_after:
            wp.synchronize()
        if _NVTX_AVAILABLE and self._range_id is not None:
            _nvtx.end_range(self._range_id)


def nvtx_mark(message: str, color: str | int = "yellow"):
    """在 Nsight 时间线上打一个瞬时标记点。"""
    if _NVTX_AVAILABLE:
        _nvtx.mark(message, color=color)


# ============================================================================
# 核心工具2：GPU Timer（精确 GPU 计时）
# ============================================================================

class gpu_timer:
    """GPU 精确计时器（使用 CUDA Event 或 CPU wall clock）。

    用法：
        with gpu_timer("my_kernel") as t:
            wp.launch(kernel, dim=n, inputs=[...])
        print(f"耗时: {t.elapsed_ms:.3f} ms")

        # 同时打 NVTX 标记 + 计时
        with gpu_timer("solver.step", nvtx=True, color="blue") as t:
            solver.step(...)
        print(f"solver: {t.elapsed_ms:.3f} ms")
    """

    def __init__(self, name: str = "", nvtx: bool = False,
                 color: str | int = "green", sync: bool = True):
        self.name = name
        self.use_nvtx = nvtx
        self.color = color
        self.sync = sync
        self.elapsed_ms = 0.0
        self._nvtx_ctx = None

    def __enter__(self):
        if self.use_nvtx:
            self._nvtx_ctx = nvtx_range(self.name, color=self.color)
            self._nvtx_ctx.__enter__()
        if self.sync:
            wp.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.sync:
            wp.synchronize()
        self._end = time.perf_counter()
        self.elapsed_ms = (self._end - self._start) * 1000.0
        if self._nvtx_ctx is not None:
            self._nvtx_ctx.__exit__(*args)


# ============================================================================
# 核心工具3：Benchmark（多次运行统计）
# ============================================================================

class BenchmarkResult:
    """Benchmark 结果。"""

    def __init__(self, name: str, times_ms: list[float]):
        self.name = name
        self.times_ms = times_ms
        self.mean_ms = statistics.mean(times_ms)
        self.std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
        self.min_ms = min(times_ms)
        self.max_ms = max(times_ms)
        self.median_ms = statistics.median(times_ms)
        self.count = len(times_ms)

    def __repr__(self):
        return (f"{self.name}: {self.mean_ms:.3f} ± {self.std_ms:.3f} ms "
                f"(min={self.min_ms:.3f}, max={self.max_ms:.3f}, n={self.count})")


def benchmark(name: str, func: Callable, warmup: int = 5, repeats: int = 20,
              sync: bool = True, nvtx: bool = False) -> BenchmarkResult:
    """对一个函数做多次运行统计。

    用法：
        result = benchmark("spring_force", lambda: wp.launch(kernel, dim=n, inputs=[...]))
        print(result)

        # 对 Newton 的 solver.step 做 benchmark
        result = benchmark("solver.step",
            lambda: solver.step(state_0, state_1, control, contacts, dt),
            warmup=3, repeats=50)
    """
    for _ in range(warmup):
        func()
        if sync:
            wp.synchronize()

    times = []
    for i in range(repeats):
        if nvtx:
            ctx = nvtx_range(f"{name}_iter_{i}", color="cyan")
            ctx.__enter__()

        if sync:
            wp.synchronize()
        t0 = time.perf_counter()
        func()
        if sync:
            wp.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

        if nvtx:
            ctx.__exit__(None, None, None)

    return BenchmarkResult(name, times)


def benchmark_kernel(name: str, kernel, dim, inputs, outputs=None,
                     warmup: int = 5, repeats: int = 20,
                     device=None) -> BenchmarkResult:
    """对一个 Warp kernel 做 benchmark 的快捷方式。"""
    if outputs is None:
        outputs = []

    def run():
        wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, device=device)

    return benchmark(name, run, warmup=warmup, repeats=repeats)


def benchmark_compare(*results: BenchmarkResult):
    """对比多个 benchmark 结果，打印表格。"""
    if not results:
        return

    baseline = results[0].mean_ms

    print(f"\n  {'名称':30s} {'平均(ms)':>10s} {'标准差':>10s} {'最小':>10s} {'相对速度':>10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for r in results:
        ratio = baseline / r.mean_ms if r.mean_ms > 0 else float('inf')
        print(f"  {r.name:30s} {r.mean_ms:10.3f} {r.std_ms:10.3f} {r.min_ms:10.3f} {ratio:9.1f}x")
    print()


# ============================================================================
# 核心工具4：Warp 内置 CUDA 活动计时
# ============================================================================

@contextlib.contextmanager
def warp_cuda_timing(filter_flags=None, report: bool = True):
    """使用 Warp 内置的 CUDA 活动计时（kernel/memcpy/memset/graph）。

    用法：
        with warp_cuda_timing():
            solver.step(state_0, state_1, control, contacts, dt)
        # 自动打印每个 kernel 的执行时间

        # 只记录 kernel 时间
        with warp_cuda_timing(filter_flags=wp.TIMING_KERNEL):
            ...

        # 记录所有活动
        with warp_cuda_timing(filter_flags=wp.TIMING_ALL):
            ...
    """
    if filter_flags is None:
        filter_flags = wp.TIMING_ALL

    wp.synchronize()
    wp.timing_begin(filter_flags)
    try:
        yield
    finally:
        wp.synchronize()
        results = wp.timing_end()
        if report and results:
            wp.timing_print(results)


# ============================================================================
# 核心工具5：Nsight 集成辅助
# ============================================================================

class NsightHelper:
    """Nsight Systems 和 Nsight Compute 的辅助工具。

    用法：
        nsight = NsightHelper()

        # 生成 nsys 命令
        cmd = nsight.nsys_command("my_script.py", output="my_report")
        print(cmd)  # 复制粘贴到终端执行

        # 生成 ncu 命令
        cmd = nsight.ncu_command("my_script.py", output="my_kernel")
        print(cmd)

        # 解析 nsys 统计输出
        nsight.parse_nsys_stats("my_report.nsys-rep")
    """

    @staticmethod
    def nsys_command(script: str, output: str = "report",
                     trace: str = "cuda,nvtx,osrt",
                     extra_args: str = "") -> str:
        """生成 nsys profile 命令。"""
        return (f"nsys profile "
                f"--trace={trace} "
                f"--output={output} "
                f"--force-overwrite=true "
                f"{extra_args} "
                f"uv run python {script}")

    @staticmethod
    def ncu_command(script: str, output: str = "kernel_report",
                    metrics: str = "",
                    kernel_filter: str = "") -> str:
        """生成 ncu (Nsight Compute) 命令。"""
        cmd = f"ncu --set full --output {output}"
        if kernel_filter:
            cmd += f' --kernel-name "{kernel_filter}"'
        if metrics:
            cmd += f" --metrics {metrics}"
        cmd += f" uv run python {script}"
        return cmd

    @staticmethod
    def nsys_stats(report_path: str) -> str:
        """生成 nsys stats 命令（从 .nsys-rep 提取统计数据）。"""
        return f"nsys stats --report cuda_gpu_kern_sum {report_path}"

    @staticmethod
    def print_commands(script: str, output_prefix: str = "profile"):
        """打印常用的 profiling 命令。"""
        print(f"\n  === Nsight Systems (整体 timeline) ===")
        print(f"  {NsightHelper.nsys_command(script, output_prefix + '_nsys')}")
        print(f"\n  打开报告:")
        print(f"  nsys-ui {output_prefix}_nsys.nsys-rep")
        print(f"\n  提取 kernel 统计:")
        print(f"  {NsightHelper.nsys_stats(output_prefix + '_nsys.nsys-rep')}")
        print(f"\n  === Nsight Compute (单 kernel 详细分析) ===")
        print(f"  {NsightHelper.ncu_command(script, output_prefix + '_ncu')}")
        print(f"\n  打开报告:")
        print(f"  ncu-ui {output_prefix}_ncu.ncu-rep")
        print()


# ============================================================================
# 核心工具6：结果导出（JSON / CSV）
# ============================================================================

def export_results_json(results: list[BenchmarkResult], path: str):
    """将 benchmark 结果导出为 JSON。"""
    data = []
    for r in results:
        data.append({
            "name": r.name,
            "mean_ms": r.mean_ms,
            "std_ms": r.std_ms,
            "min_ms": r.min_ms,
            "max_ms": r.max_ms,
            "median_ms": r.median_ms,
            "count": r.count,
            "times_ms": r.times_ms,
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  结果已导出到 {path}")


def export_results_csv(results: list[BenchmarkResult], path: str):
    """将 benchmark 结果导出为 CSV。"""
    with open(path, "w") as f:
        f.write("name,mean_ms,std_ms,min_ms,max_ms,median_ms,count\n")
        for r in results:
            f.write(f"{r.name},{r.mean_ms:.4f},{r.std_ms:.4f},"
                    f"{r.min_ms:.4f},{r.max_ms:.4f},{r.median_ms:.4f},{r.count}\n")
    print(f"  结果已导出到 {path}")


# ============================================================================
# 演示：展示所有工具的用法
# ============================================================================

@wp.kernel
def _demo_kernel(data: wp.array(dtype=float)):
    tid = wp.tid()
    x = float(tid)
    for _ in range(50):
        x = wp.sin(x) + wp.cos(x)
    data[tid] = x


@wp.kernel
def _demo_spring(x: wp.array(dtype=wp.vec3),
                 f: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    if tid > 0:
        dx = x[tid] - x[tid - 1]
        length = wp.length(dx)
        if length > 1.0e-6:
            force = dx / length * (length - 1.0) * 1000.0
            wp.atomic_sub(f, tid, force)
            wp.atomic_add(f, tid - 1, force)


def demo_all():
    """运行所有工具的演示。"""

    print("=" * 60)
    print("工具1：nvtx_range —— NVTX 范围标记")
    print("=" * 60)
    print(f"  NVTX 可用: {_NVTX_AVAILABLE}")
    print(f"  （没装 nvtx 包也能用，只是标记不生效）")

    n = 100000
    data = wp.zeros(n, dtype=float)

    with nvtx_range("demo_section", color="green"):
        wp.launch(_demo_kernel, dim=n, inputs=[data])
    print(f"  nvtx_range 使用完成")
    print()

    print("=" * 60)
    print("工具2：gpu_timer —— GPU 计时")
    print("=" * 60)

    with gpu_timer("demo_kernel") as t:
        wp.launch(_demo_kernel, dim=n, inputs=[data])
    print(f"  {t.name}: {t.elapsed_ms:.3f} ms")

    with gpu_timer("demo_kernel_with_nvtx", nvtx=True, color="blue") as t:
        wp.launch(_demo_kernel, dim=n, inputs=[data])
    print(f"  {t.name}: {t.elapsed_ms:.3f} ms (同时打了 NVTX 标记)")
    print()

    print("=" * 60)
    print("工具3：benchmark —— 多次运行统计")
    print("=" * 60)

    r1 = benchmark("demo_kernel_100k",
                    lambda: wp.launch(_demo_kernel, dim=100000, inputs=[data]))
    print(f"  {r1}")

    x = wp.array(np.linspace(0, 10, 1000).reshape(-1, 1).repeat(3, axis=1).astype(np.float32),
                 dtype=wp.vec3)
    f = wp.zeros(1000, dtype=wp.vec3)

    r2 = benchmark_kernel("spring_force_1k", _demo_spring, dim=1000, inputs=[x, f])
    print(f"  {r2}")

    benchmark_compare(r1, r2)

    print("=" * 60)
    print("工具4：warp_cuda_timing —— Warp 内置 CUDA 活动计时")
    print("=" * 60)

    with warp_cuda_timing(filter_flags=wp.TIMING_KERNEL):
        wp.launch(_demo_kernel, dim=n, inputs=[data])
        wp.launch(_demo_kernel, dim=n, inputs=[data])
    print()

    print("=" * 60)
    print("工具5：NsightHelper —— 生成 Nsight 命令")
    print("=" * 60)

    NsightHelper.print_commands("learning_warp/07_profiling.py")

    print("=" * 60)
    print("工具6：导出结果")
    print("=" * 60)
    export_results_json([r1, r2], "/tmp/benchmark_results.json")
    export_results_csv([r1, r2], "/tmp/benchmark_results.csv")
    print()


if __name__ == "__main__":
    demo_all()
    print("全部完成！")
