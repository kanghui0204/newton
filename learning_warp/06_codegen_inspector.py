"""
Warp 学习 06：查看 Warp 生成的 CUDA/C++ 代码
=============================================

【运行方式】
    cd /home/hkang/newton_related/newton
    uv run python learning_warp/06_codegen_inspector.py

【学习目标】
- 理解 Warp 的 JIT 编译流程：Python → C++/CUDA → 二进制
- 学会查看生成的 CUDA 代码（在缓存目录中）
- 理解前向代码和反向（adjoint）代码的关系
"""

import os

import warp as wp

wp.init()


# ============================================================================
# 第1节：Warp 的编译流程
# ============================================================================
# Python 代码 → Warp AST 分析 → C++/CUDA 代码生成 → 编译为 .so/.dll
#
# 流程图：
#   @wp.kernel                    你写的 Python
#       ↓ (codegen.py)
#   .cu / .cpp 文件               生成的 CUDA/C++ 代码
#       ↓ (NVRTC / Clang)
#   .cubin / .o 二进制            编译后的 GPU/CPU 代码
#       ↓
#   wp.launch() 执行              在 GPU/CPU 上运行


# ============================================================================
# 第2节：查看缓存目录
# ============================================================================

def show_cache_info():
    print("=" * 60)
    print("第2节：Warp 内核缓存位置")
    print("=" * 60)

    cache_dir = wp.config.kernel_cache_dir
    print(f"  缓存目录: {cache_dir}")

    if cache_dir and os.path.exists(cache_dir):
        items = os.listdir(cache_dir)
        print(f"  文件/目录数: {len(items)}")

        cu_files = []
        cpp_files = []
        for root, dirs, files in os.walk(cache_dir):
            for f in files:
                if f.endswith(".cu"):
                    cu_files.append(os.path.join(root, f))
                elif f.endswith(".cpp"):
                    cpp_files.append(os.path.join(root, f))

        print(f"  .cu 文件数: {len(cu_files)}")
        print(f"  .cpp 文件数: {len(cpp_files)}")

        if cu_files:
            print(f"\n  最近的 .cu 文件:")
            sorted_files = sorted(cu_files, key=os.path.getmtime, reverse=True)
            for f in sorted_files[:3]:
                size = os.path.getsize(f)
                print(f"    {os.path.basename(f)} ({size} bytes)")
    print()


# ============================================================================
# 第3节：触发编译并查看生成的代码
# ============================================================================

@wp.kernel
def example_for_codegen(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=float),
    gravity: wp.vec3,
    dt: float,
    pos_out: wp.array(dtype=wp.vec3),
    vel_out: wp.array(dtype=wp.vec3),
):
    """这个 kernel 会被编译，我们查看生成的代码。"""
    tid = wp.tid()

    m = masses[tid]
    if m <= 0.0:
        pos_out[tid] = positions[tid]
        vel_out[tid] = velocities[tid]
        return

    inv_m = 1.0 / m
    v_new = velocities[tid] + (forces[tid] * inv_m + gravity) * dt
    x_new = positions[tid] + v_new * dt

    pos_out[tid] = x_new
    vel_out[tid] = v_new


def inspect_generated_code():
    print("=" * 60)
    print("第3节：查看生成的 CUDA 代码")
    print("=" * 60)

    n = 4
    x = wp.zeros(n, dtype=wp.vec3)
    v = wp.zeros(n, dtype=wp.vec3)
    f = wp.zeros(n, dtype=wp.vec3)
    m = wp.ones(n, dtype=float)
    x_out = wp.zeros(n, dtype=wp.vec3)
    v_out = wp.zeros(n, dtype=wp.vec3)

    wp.launch(example_for_codegen, dim=n,
              inputs=[x, v, f, m, wp.vec3(0.0, 0.0, -9.81), 0.01],
              outputs=[x_out, v_out])

    cache_dir = wp.config.kernel_cache_dir
    if not cache_dir or not os.path.exists(cache_dir):
        print("  缓存目录不存在，跳过")
        return

    cu_files = []
    for root, dirs, files in os.walk(cache_dir):
        for fname in files:
            if fname.endswith(".cu"):
                cu_files.append(os.path.join(root, fname))

    if not cu_files:
        print("  没有找到 .cu 文件")
        return

    latest = max(cu_files, key=os.path.getmtime)
    print(f"  查看最新的 .cu 文件: {os.path.basename(latest)}")
    print(f"  完整路径: {latest}")
    print()

    with open(latest, "r") as file:
        content = file.read()

    lines = content.split("\n")
    print(f"  总行数: {len(lines)}")
    print()

    # 查找前向核函数
    print("  【生成的 CUDA 代码片段（前向）】:")
    print("  " + "-" * 50)
    in_forward = False
    forward_lines = []
    for i, line in enumerate(lines):
        if "example_for_codegen" in line and "void" in line.lower() and "adj" not in line.lower():
            in_forward = True
        if in_forward:
            forward_lines.append(f"  {i+1:4d} | {line}")
            if len(forward_lines) > 30:
                forward_lines.append("  ... (truncated)")
                break
            if line.strip() == "}" and len(forward_lines) > 3:
                break

    if forward_lines:
        print("\n".join(forward_lines[:25]))
    else:
        print("  (未找到前向核函数，显示文件开头)")
        for i in range(min(20, len(lines))):
            print(f"  {i+1:4d} | {lines[i]}")
    print()

    # 查找反向核函数（adjoint）
    print("  【生成的 CUDA 代码片段（反向/adjoint）】:")
    print("  " + "-" * 50)
    adj_lines = []
    in_adjoint = False
    for i, line in enumerate(lines):
        if "adj_example_for_codegen" in line and "void" in line.lower():
            in_adjoint = True
        if in_adjoint:
            adj_lines.append(f"  {i+1:4d} | {line}")
            if len(adj_lines) > 30:
                adj_lines.append("  ... (truncated)")
                break
            if line.strip() == "}" and len(adj_lines) > 3:
                break

    if adj_lines:
        print("\n".join(adj_lines[:25]))
    else:
        print("  (反向代码可能在另一个文件中)")

    print()
    print("  【如何自己查看完整代码】:")
    print(f"  1. 打开缓存目录: {cache_dir}")
    print(f"  2. 找到 .cu 文件并用编辑器打开")
    print(f"  3. 搜索你的 kernel 函数名（如 'example_for_codegen'）")
    print(f"  4. 'adj_' 前缀的是反向传播代码")
    print()


# ============================================================================
# 第4节：Warp 如何生成反向代码
# ============================================================================

def explain_adjoint():
    print("=" * 60)
    print("第4节：前向代码 vs 反向代码（adjoint）")
    print("=" * 60)

    print("""
  Warp 对每个 kernel 自动生成两个版本：

  【前向代码】你写的逻辑：
    v_new = v + (f * inv_m + gravity) * dt
    x_new = x + v_new * dt

  【反向代码】自动生成的梯度计算（链式法则）：
    adj_v_new += adj_x_new * dt           # dx_new/dv_new = dt
    adj_v += adj_v_new                     # dv_new/dv = 1
    adj_f += adj_v_new * inv_m * dt        # dv_new/df = inv_m * dt
    adj_dt += dot(adj_v_new, f * inv_m + gravity)

  规则：
    y = a + b  → adj_a += adj_y, adj_b += adj_y
    y = a * b  → adj_a += adj_y * b, adj_b += adj_y * a
    y = f(a)   → adj_a += adj_y * f'(a)

  在缓存的 .cu 文件中：
    - 搜索函数名 → 前向代码
    - 搜索 adj_函数名 → 反向代码
    """)


# ============================================================================
if __name__ == "__main__":
    show_cache_info()
    inspect_generated_code()
    explain_adjoint()
    print("全部通过！")
