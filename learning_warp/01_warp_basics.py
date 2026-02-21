"""
Warp 学习 01：基础概念 —— kernel / func / array / launch / tid
================================================================

【运行方式】
    cd /home/hkang/newton_related/newton
    uv run python learning_warp/01_warp_basics.py

【学习目标】
- 理解 Warp 的核心编程模型：写 Python，跑 GPU
- 掌握 5 个核心概念：@wp.kernel, @wp.func, wp.array, wp.launch, wp.tid()
- 理解"一个线程处理一个元素"的并行思维

【Warp 是什么？（一句话）】
  Warp 让你用 Python 语法写 GPU 并行代码。
  你写的 Python 函数会被 JIT 编译成 CUDA 核函数，在 GPU 上并行执行。
"""

import numpy as np
import warp as wp

wp.init()


# ============================================================================
# 第1节：@wp.kernel —— GPU 并行函数
# ============================================================================
# @wp.kernel 装饰的函数不是普通 Python 函数！
# 它会被 Warp 编译成 CUDA 代码，在 GPU 上并行执行。
#
# 关键规则：
#   1. kernel 内不能 return 值（通过输出数组写结果）
#   2. kernel 内不能用 Python 的 list/dict/class
#   3. kernel 内只能用 Warp 提供的类型和函数
#   4. 每个线程用 wp.tid() 获取自己的编号

@wp.kernel
def add_arrays(a: wp.array(dtype=float),
               b: wp.array(dtype=float),
               c: wp.array(dtype=float)):
    """最简单的 kernel：c[i] = a[i] + b[i]"""
    tid = wp.tid()  # 获取当前线程的编号（0, 1, 2, ..., n-1）
    c[tid] = a[tid] + b[tid]


def demo_basic_kernel():
    """演示最基本的 kernel 用法。"""
    print("=" * 60)
    print("第1节：基本 kernel —— 数组相加")
    print("=" * 60)

    n = 8
    # wp.array 是 GPU 数组（类似 numpy 的 ndarray，但在 GPU 上）
    a = wp.array(np.arange(n, dtype=np.float32))       # [0,1,2,...,7]
    b = wp.array(np.ones(n, dtype=np.float32) * 10.0)   # [10,10,...,10]
    c = wp.zeros(n, dtype=float)                         # [0,0,...,0]

    # wp.launch 启动 kernel：
    #   dim=n → 启动 n 个线程（每个线程处理一个元素）
    #   inputs=[a, b] → 输入参数
    #   outputs=[c] → 输出参数（区分 inputs/outputs 是给自动微分用的）
    wp.launch(add_arrays, dim=n, inputs=[a, b], outputs=[c])

    # .numpy() 把 GPU 数组拷回 CPU，变成 numpy 数组
    print(f"  a = {a.numpy()}")
    print(f"  b = {b.numpy()}")
    print(f"  c = a + b = {c.numpy()}")
    print()


# ============================================================================
# 第2节：@wp.func —— 可复用的设备函数
# ============================================================================
# @wp.func 是在 kernel 内部调用的辅助函数。
# 它不能被 wp.launch 启动，只能被 @wp.kernel 或其他 @wp.func 调用。
#
# 类比：
#   @wp.kernel = GPU 上的 main 函数（入口）
#   @wp.func   = GPU 上的普通函数（被 main 调用的子函数）

@wp.func
def clamp(x: float, lo: float, hi: float) -> float:
    """将 x 限制在 [lo, hi] 范围内。"""
    return wp.max(lo, wp.min(hi, x))


@wp.func
def safe_normalize(v: wp.vec3) -> wp.vec3:
    """安全的向量归一化（避免除以零）。"""
    length = wp.length(v)
    if length > 1.0e-6:
        return v / length
    return wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def apply_clamp_kernel(data: wp.array(dtype=float),
                       lo: float,
                       hi: float,
                       result: wp.array(dtype=float)):
    """演示在 kernel 中调用 @wp.func。"""
    tid = wp.tid()
    result[tid] = clamp(data[tid], lo, hi)


def demo_wp_func():
    """演示 @wp.func 的用法。"""
    print("=" * 60)
    print("第2节：@wp.func —— 可复用的设备函数")
    print("=" * 60)

    data = wp.array(np.array([-5.0, -1.0, 0.5, 3.0, 7.0, 15.0], dtype=np.float32))
    result = wp.zeros(len(data), dtype=float)

    wp.launch(apply_clamp_kernel, dim=len(data), inputs=[data, 0.0, 10.0], outputs=[result])

    print(f"  原始数据:     {data.numpy()}")
    print(f"  clamp(0, 10): {result.numpy()}")
    print()


# ============================================================================
# 第3节：wp.array —— GPU 数组
# ============================================================================
# wp.array 是 Warp 的核心数据容器，类似 numpy.ndarray 但在 GPU 上。
#
# 创建方式：
#   wp.zeros(n, dtype=float)              全零数组
#   wp.ones(n, dtype=wp.vec3)             全一数组
#   wp.array(numpy_array)                 从 numpy 转换
#   wp.empty(n, dtype=float)              未初始化（最快但内容随机）
#
# 类型（dtype）：
#   float, int, wp.vec2, wp.vec3, wp.vec4,
#   wp.mat22, wp.mat33, wp.mat44,
#   wp.quat, wp.transform, wp.spatial_vector

def demo_arrays():
    """演示 wp.array 的各种用法。"""
    print("=" * 60)
    print("第3节：wp.array —— GPU 数组")
    print("=" * 60)

    # 标量数组
    a = wp.zeros(5, dtype=float)
    print(f"  zeros(5):     {a.numpy()}")

    # 向量数组（每个元素是 vec3）
    positions = wp.zeros(3, dtype=wp.vec3)
    print(f"  vec3 zeros:   {positions.numpy()}")

    # 从 numpy 转换
    np_data = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]], dtype=np.float32)
    gpu_data = wp.array(np_data, dtype=wp.vec3)
    print(f"  from numpy:   {gpu_data.numpy()}")

    # GPU → CPU
    back_to_numpy = gpu_data.numpy()
    print(f"  back to numpy: {back_to_numpy}")
    print(f"  shape: {back_to_numpy.shape}, dtype: {back_to_numpy.dtype}")

    # requires_grad=True 启用梯度追踪（用于自动微分）
    grad_array = wp.zeros(5, dtype=float, requires_grad=True)
    print(f"  requires_grad: {grad_array.requires_grad}")
    print()


# ============================================================================
# 第4节：wp.launch 的细节
# ============================================================================
# wp.launch(kernel, dim, inputs, outputs, device)
#
# dim 参数：
#   dim=n         → 启动 n 个线程，tid = 0..n-1（1D）
#   dim=(m, n)    → 启动 m×n 个线程，i, j = wp.tid()（2D）
#   dim=(l, m, n) → 启动 l×m×n 个线程，i, j, k = wp.tid()（3D）
#
# 2D/3D 的 tid 用法：
#   i, j = wp.tid()       # 2D
#   i, j, k = wp.tid()    # 3D

@wp.kernel
def fill_2d(grid: wp.array2d(dtype=float),
            rows: int,
            cols: int):
    """2D kernel：每个线程处理网格的一个格子。"""
    i, j = wp.tid()
    grid[i, j] = float(i * cols + j)


def demo_launch_2d():
    """演示 2D launch。"""
    print("=" * 60)
    print("第4节：2D launch —— 二维网格并行")
    print("=" * 60)

    rows, cols = 3, 4
    grid = wp.zeros((rows, cols), dtype=float)

    wp.launch(fill_2d, dim=(rows, cols), inputs=[grid, rows, cols])

    print(f"  3×4 grid:")
    result = grid.numpy()
    for i in range(rows):
        print(f"    {result[i]}")
    print()


# ============================================================================
# 第5节：完整示例 —— 粒子重力模拟（Newton 风格）
# ============================================================================
# 把上面的概念组合起来，写一个 Newton 求解器风格的粒子模拟。

@wp.func
def integrate_particle(pos: wp.vec3,
                       vel: wp.vec3,
                       gravity: wp.vec3,
                       dt: float):
    """半隐式欧拉积分（和 Newton 的 integrate_particles 一样的逻辑）。"""
    vel_new = vel + gravity * dt      # 先更新速度
    pos_new = pos + vel_new * dt      # 用新速度更新位置
    return pos_new, vel_new


@wp.kernel
def step_particles(positions: wp.array(dtype=wp.vec3),
                   velocities: wp.array(dtype=wp.vec3),
                   gravity: wp.vec3,
                   dt: float,
                   positions_out: wp.array(dtype=wp.vec3),
                   velocities_out: wp.array(dtype=wp.vec3)):
    """一个时间步：对所有粒子做积分。"""
    tid = wp.tid()
    pos_new, vel_new = integrate_particle(
        positions[tid], velocities[tid], gravity, dt
    )
    positions_out[tid] = pos_new
    velocities_out[tid] = vel_new


def demo_particle_sim():
    """演示一个简单的粒子重力模拟。"""
    print("=" * 60)
    print("第5节：粒子重力模拟（Newton 风格）")
    print("=" * 60)

    n_particles = 4
    dt = 0.01
    gravity = wp.vec3(0.0, 0.0, -9.81)

    positions = wp.array(
        [wp.vec3(float(i), 0.0, 10.0) for i in range(n_particles)],
        dtype=wp.vec3,
    )
    velocities = wp.zeros(n_particles, dtype=wp.vec3)
    positions_out = wp.zeros(n_particles, dtype=wp.vec3)
    velocities_out = wp.zeros(n_particles, dtype=wp.vec3)

    print(f"  初始位置: {positions.numpy()}")

    for step in range(100):
        wp.launch(
            step_particles,
            dim=n_particles,
            inputs=[positions, velocities, gravity, dt],
            outputs=[positions_out, velocities_out],
        )
        positions, positions_out = positions_out, positions
        velocities, velocities_out = velocities_out, velocities

    print(f"  100步后位置: {positions.numpy()}")
    print(f"  100步后速度: {velocities.numpy()}")
    z_values = positions.numpy()[:, 2]
    expected_z = 10.0 + 0.5 * (-9.81) * (100 * dt) ** 2
    print(f"  理论 z = 10 + 0.5*(-9.81)*1² = {expected_z:.3f}")
    print(f"  实际 z = {z_values[0]:.3f} （应该接近理论值）")
    print()


# ============================================================================
# 运行所有演示
# ============================================================================
if __name__ == "__main__":
    demo_basic_kernel()
    demo_wp_func()
    demo_arrays()
    demo_launch_2d()
    demo_particle_sim()
    print("全部通过！")
