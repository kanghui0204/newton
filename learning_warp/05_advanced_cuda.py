"""
Warp 学习 05：高级 CUDA 特性 —— 原子操作 / 归约 / CUDA 图 / ScopedDevice
========================================================================

【运行方式】
    cd /home/hkang/newton_related/newton
    uv run python learning_warp/05_advanced_cuda.py

【学习目标】
- 原子操作（atomic_add/atomic_min/atomic_max）及其在物理仿真中的用途
- 归约模式（全局求和、求最大值）
- CUDA 图捕获（ScopedCapture）消除 Python 开销
- 设备管理（ScopedDevice）
"""

import numpy as np
import warp as wp

wp.init()


# ============================================================================
# 第1节：原子操作 —— 多线程安全地写入同一位置
# ============================================================================
# 场景：1000 个线程同时往 result[0] 里加数 → 必须用 atomic_add
#
# Newton 中的典型用法：
#   - 多根弹簧同时给同一个粒子加力 → wp.atomic_add(f, particle_id, force)
#   - 碰撞检测中的计数器 → wp.atomic_add(count, 0, 1)

@wp.kernel
def wrong_sum(data: wp.array(dtype=float),
              result: wp.array(dtype=float)):
    """错误示范：多线程同时写 result[0] 会丢失数据。"""
    tid = wp.tid()
    result[0] = result[0] + data[tid]  # 竞争条件！


@wp.kernel
def correct_sum(data: wp.array(dtype=float),
                result: wp.array(dtype=float)):
    """正确做法：用 atomic_add。"""
    tid = wp.tid()
    wp.atomic_add(result, 0, data[tid])


@wp.kernel
def atomic_min_max(data: wp.array(dtype=float),
                   min_val: wp.array(dtype=float),
                   max_val: wp.array(dtype=float)):
    """用原子操作找最小/最大值。"""
    tid = wp.tid()
    wp.atomic_min(min_val, 0, data[tid])
    wp.atomic_max(max_val, 0, data[tid])


def demo_atomic():
    print("=" * 60)
    print("第1节：原子操作")
    print("=" * 60)

    n = 1000
    data = wp.array(np.ones(n, dtype=np.float32))
    expected = float(n)

    result_wrong = wp.zeros(1, dtype=float)
    wp.launch(wrong_sum, dim=n, inputs=[data, result_wrong])
    wrong_val = result_wrong.numpy()[0]

    result_correct = wp.zeros(1, dtype=float)
    wp.launch(correct_sum, dim=n, inputs=[data, result_correct])
    correct_val = result_correct.numpy()[0]

    print(f"  1000 个 1.0 求和，期望 = {expected}")
    print(f"  不用 atomic:  结果 = {wrong_val}  (可能不正确！)")
    print(f"  用 atomic_add: 结果 = {correct_val}  (正确!)")

    random_data = wp.array(np.random.randn(n).astype(np.float32))
    min_val = wp.array([1e10], dtype=float)
    max_val = wp.array([-1e10], dtype=float)
    wp.launch(atomic_min_max, dim=n, inputs=[random_data, min_val, max_val])

    np_data = random_data.numpy()
    print(f"\n  atomic_min: {min_val.numpy()[0]:.4f}  numpy min: {np_data.min():.4f}")
    print(f"  atomic_max: {max_val.numpy()[0]:.4f}  numpy max: {np_data.max():.4f}")
    print()


# ============================================================================
# 第2节：归约模式 —— 用 atomic 实现全局统计
# ============================================================================
# 物理仿真中的归约应用：
#   - 计算所有粒子的动能总和 → 监控能量守恒
#   - 找最大速度 → 用于 CFL 条件判断时间步长

@wp.kernel
def compute_kinetic_energy(
    v: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    total_ke: wp.array(dtype=float),
):
    """计算所有粒子的动能总和：KE = Σ 0.5 * m * v²"""
    tid = wp.tid()
    speed_sq = wp.dot(v[tid], v[tid])
    ke = 0.5 * mass[tid] * speed_sq
    wp.atomic_add(total_ke, 0, ke)


@wp.kernel
def find_max_speed(v: wp.array(dtype=wp.vec3),
                   max_speed: wp.array(dtype=float)):
    """找所有粒子中的最大速度。"""
    tid = wp.tid()
    speed = wp.length(v[tid])
    wp.atomic_max(max_speed, 0, speed)


def demo_reduction():
    print("=" * 60)
    print("第2节：归约模式 —— 全局统计量")
    print("=" * 60)

    n = 1000
    np.random.seed(42)
    v_np = np.random.randn(n, 3).astype(np.float32)
    m_np = np.ones(n, dtype=np.float32)

    v = wp.array(v_np, dtype=wp.vec3)
    mass = wp.array(m_np, dtype=float)

    total_ke = wp.zeros(1, dtype=float)
    wp.launch(compute_kinetic_energy, dim=n, inputs=[v, mass, total_ke])

    max_speed = wp.zeros(1, dtype=float)
    wp.launch(find_max_speed, dim=n, inputs=[v, max_speed])

    ke_gpu = total_ke.numpy()[0]
    ke_cpu = 0.5 * np.sum(m_np[:, None] * v_np ** 2)
    max_gpu = max_speed.numpy()[0]
    max_cpu = np.max(np.linalg.norm(v_np, axis=1))

    print(f"  动能 (GPU atomic): {ke_gpu:.4f}")
    print(f"  动能 (CPU numpy):  {ke_cpu:.4f}")
    print(f"  最大速度 (GPU):    {max_gpu:.4f}")
    print(f"  最大速度 (CPU):    {max_cpu:.4f}")
    print()


# ============================================================================
# 第3节：CUDA 图捕获 —— 消除 Python 开销
# ============================================================================
# 普通模式：每帧 Python 调用 wp.launch → GPU 执行 → 回到 Python → 下一个 launch
# CUDA 图模式：一次性录制所有 launch → 之后每帧直接在 GPU 上重放，Python 零开销
#
# Newton 中所有示例的 capture() 方法就是这个模式

@wp.kernel
def simple_step(x: wp.array(dtype=float),
                v: wp.array(dtype=float),
                dt: float):
    tid = wp.tid()
    v[tid] = v[tid] + (-9.81) * dt
    x[tid] = x[tid] + v[tid] * dt


def demo_cuda_graph():
    print("=" * 60)
    print("第3节：CUDA 图捕获（ScopedCapture）")
    print("=" * 60)

    if not wp.is_cuda_available():
        print("  跳过：需要 CUDA GPU")
        print()
        return

    n = 10000
    x = wp.zeros(n, dtype=float, device="cuda:0")
    v = wp.zeros(n, dtype=float, device="cuda:0")
    dt = 0.001

    # 普通模式计时
    import time
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(1000):
        for _ in range(10):
            wp.launch(simple_step, dim=n, inputs=[x, v, dt], device="cuda:0")
    wp.synchronize()
    normal_time = time.perf_counter() - t0

    # 重置
    x.zero_()
    v.zero_()

    # CUDA 图模式
    with wp.ScopedCapture(device="cuda:0") as capture:
        for _ in range(10):
            wp.launch(simple_step, dim=n, inputs=[x, v, dt], device="cuda:0")
    graph = capture.graph

    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(1000):
        wp.capture_launch(graph)
    wp.synchronize()
    graph_time = time.perf_counter() - t0

    speedup = normal_time / graph_time if graph_time > 0 else float('inf')
    print(f"  普通模式: {normal_time:.3f}s")
    print(f"  CUDA 图:  {graph_time:.3f}s")
    print(f"  加速比:   {speedup:.1f}x")
    print(f"  → CUDA 图消除了每次 wp.launch 的 Python→GPU 调度开销")
    print()


# ============================================================================
# 第4节：设备管理
# ============================================================================

def demo_device():
    print("=" * 60)
    print("第4节：设备管理")
    print("=" * 60)

    print(f"  默认设备: {wp.get_device()}")
    print(f"  CUDA 可用: {wp.is_cuda_available()}")

    # CPU 数组
    a_cpu = wp.zeros(5, dtype=float, device="cpu")
    print(f"  CPU 数组设备: {a_cpu.device}")

    if wp.is_cuda_available():
        # GPU 数组
        a_gpu = wp.zeros(5, dtype=float, device="cuda:0")
        print(f"  GPU 数组设备: {a_gpu.device}")

        # ScopedDevice：临时切换默认设备
        with wp.ScopedDevice("cuda:0"):
            b = wp.zeros(5, dtype=float)
            print(f"  ScopedDevice 内: {b.device}")

        # CPU ↔ GPU 拷贝
        wp.copy(a_gpu, a_cpu)
        print(f"  CPU→GPU 拷贝完成")

    print()


# ============================================================================
if __name__ == "__main__":
    demo_atomic()
    demo_reduction()
    demo_cuda_graph()
    demo_device()
    print("全部通过！")
