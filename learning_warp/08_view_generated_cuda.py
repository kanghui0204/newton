"""
Warp 学习 08：查看和分析 Warp 生成的 CUDA Kernel 源码
=====================================================

【运行方式】
    cd /home/hkang/newton_related/newton
    uv run python learning_warp/08_view_generated_cuda.py

【本文件目的】
手把手教你找到 Warp 生成的 CUDA 代码，理解前向/反向代码的对应关系。
以你这台电脑上 Newton 实际运行过的 kernel 为例。


================================================================================
一、Warp 的编译流程（全景图）
================================================================================

  你写的 Python:                     Warp 内部处理:                   最终产物:

  @wp.kernel                         codegen.py                      .cu 文件
  def integrate_particles(      →    AST 分析 + 代码生成         →   (CUDA C++ 源码)
      x, v, f, ...):                                                    ↓
      tid = wp.tid()                                               NVRTC 编译
      v1 = v + f * dt                                                  ↓
      x1 = x + v1 * dt                                             .ptx 文件
                                                                   (GPU 汇编码)
                                                                        ↓
                                                                    GPU 上执行

  Warp 自动生成两个版本：
    - forward kernel:  你写的逻辑
    - backward kernel: 自动推导的梯度计算（用于自动微分）


================================================================================
二、缓存目录在哪？（你这台电脑的实际路径）
================================================================================

  Warp 把生成的所有代码缓存在这里：

    /home/hkang/.cache/warp/1.12.0.dev20260127/

  怎么确认？运行：
    uv run python -c "import warp as wp; wp.init(); print(wp.config.kernel_cache_dir)"

  目录结构：
    /home/hkang/.cache/warp/1.12.0.dev20260127/
    ├── wp_newton._src.solvers.solver_9972fda/
    │   ├── ...solver_9972fda.cu        ← CUDA C++ 源码（1131 行）
    │   ├── ...solver_9972fda.sm89.ptx  ← PTX 汇编（编译后的 GPU 代码）
    │   └── ...solver_9972fda.meta      ← 元数据（编译选项等）
    │
    ├── wp_newton._src.solvers.xpbd.kernels_7a107a6/
    │   └── ...kernels_7a107a6.cu       ← XPBD 求解器（30900 行！最大的文件）
    │
    ├── wp_newton._src.geometry.kernels_17bce7a/
    │   └── ...kernels_17bce7a.cu       ← 碰撞检测核函数（9456 行）
    │
    └── ...（其他模块）


================================================================================
三、Newton 生成的所有 CUDA 文件一览（你这台电脑上的）
================================================================================

  文件名（模块名）                                行数      内容
  ─────────────────────────────────────────────── ──────── ───────────────
  wp_newton._src.solvers.xpbd.kernels              30900   XPBD 约束求解
  wp_newton._src.geometry.narrow_phase             18203   窄相碰撞检测(GJK/MPR)
  wp_newton._src.solvers.vbd.particle_vbd_kernels  12121   VBD 粒子求解
  wp_newton._src.geometry.kernels                   9456   碰撞几何 + SDF
  wp_newton._src.geometry.narrow_phase (另一版本)    9007   窄相碰撞检测
  wp_newton._src.geometry.raycast                   8405   光线投射
  wp_newton._src.solvers.vbd.rigid_vbd_kernels      6970   VBD 刚体求解
  wp_newton._src.solvers.semi_implicit.kernels      6374   半隐式接触力
  wp_newton._src.sim.collide                        4667   碰撞管线
  wp_newton._src.viewer.kernels                     4501   可视化渲染
  wp_newton._src.solvers.solver                     1131   基础积分(本文重点)
  wp_newton._src.utils.mesh                         1310   网格工具


================================================================================
四、怎么打开和阅读生成的代码（以 integrate_particles 为例）
================================================================================

  第1步：用编辑器打开 .cu 文件

    # 在 Cursor 中打开（直接能看 CUDA 语法高亮）
    # 文件路径：
    /home/hkang/.cache/warp/1.12.0.dev20260127/wp_newton._src.solvers.solver_9972fda/wp_newton._src.solvers.solver_9972fda.cu

    # 或者用命令行快速查看：
    # less /home/hkang/.cache/warp/1.12.0.dev20260127/wp_newton._src.solvers.solver_9972fda/*.cu

  第2步：搜索你关心的 kernel 名称

    文件中有 4 个 __global__ 函数：

    第 402 行: integrate_particles_..._cuda_kernel_forward   ← 粒子积分（前向）
    第 541 行: integrate_particles_..._cuda_kernel_backward  ← 粒子积分（反向/梯度）
    第 791 行: integrate_bodies_..._cuda_kernel_forward      ← 刚体积分（前向）
    第 914 行: integrate_bodies_..._cuda_kernel_backward     ← 刚体积分（反向/梯度）


================================================================================
五、Python 代码 vs 生成的 CUDA 代码 —— 逐行对照
================================================================================

  【你写的 Python（newton/_src/solvers/solver.py 第 50-56 行）】

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    v1 = v0 + (f0 * inv_mass + world_g * wp.step(-inv_mass)) * dt
    x1 = x0 + v1 * dt

  【Warp 生成的 CUDA（.cu 文件第 510-531 行）】

    // v1 = v0 + (f0 * inv_mass + world_g * wp.step(-inv_mass)) * dt    <L 51>
    var_27 = wp::mul(var_14, var_17);              // f0 * inv_mass
    var_28 = wp::neg(var_17);                      // -inv_mass
    var_29 = wp::step(var_28);                     // wp.step(-inv_mass)
    var_30 = wp::mul(var_25, var_29);              // world_g * step(...)
    var_31 = wp::add(var_27, var_30);              // f0*inv_mass + world_g*step
    var_32 = wp::mul(var_31, var_dt);              // (...) * dt
    var_33 = wp::add(var_11, var_32);              // v0 + (...)

    // x1 = x0 + v1 * dt                                                <L 56>
    var_40 = wp::mul(var_39, var_dt);              // v1 * dt
    var_41 = wp::add(var_2, var_40);               // x0 + v1 * dt

  可以看到：
    - 每个 Python 操作被拆成一个个原子操作（add, mul, neg 等）
    - 变量名变成了 var_0, var_1, var_2...（按顺序编号）
    - 注释 <L 51> 标注了对应 Python 源码的行号！


================================================================================
六、前向代码 vs 反向代码 —— 理解自动微分
================================================================================

  【前向代码（第 402 行开始）】
    函数名:  integrate_particles_..._cuda_kernel_forward
    参数:    x, v, f, w, ..., x_new, v_new
    做的事:  你写的物理逻辑（v1 = v0 + a*dt, x1 = x0 + v1*dt）

  【反向代码（第 541 行开始）】
    函数名:  integrate_particles_..._cuda_kernel_backward
    参数:    x, v, f, ..., adj_x, adj_v, adj_f, ...
             ↑ 前向参数              ↑ 对应的梯度（adjoint）
    做的事:  自动推导的链式法则

  反向代码的参数数量 = 前向参数数量 × 2：
    - 前半部分: 前向计算时的输入/输出（需要重新算一遍前向，获取中间变量）
    - 后半部分: 每个参数对应的梯度（adj_ 前缀）

  反向代码会：
    1. 重新执行前向计算（获取所有中间变量 var_0, var_1, ...）
    2. 从后往前，用链式法则传递梯度
    3. 把梯度写入 adj_x, adj_v, adj_f 等数组


================================================================================
七、wp:: 封装函数的 C++ 源码在哪？
================================================================================

  .cu 文件里调用的 wp::mul、wp::add、wp::dot 等不是黑盒！
  它们的 C++ 源码全部在 Warp 仓库的 native/ 目录下：

    /home/hkang/newton_related/warp/warp/native/
    ├── vec.h          ← wp::vec_t<3>、wp::dot、wp::length、wp::cross、wp::normalize
    │                     wp::add、wp::mul、wp::atomic_add、wp::atomic_sub
    │                     以及所有 adj_dot、adj_length 等反向函数（2333 行）
    │
    ├── mat.h          ← wp::mat_t<3,3>、wp::determinant、wp::inverse、wp::transpose
    │                     矩阵乘法、矩阵-向量乘法（5696 行）
    │
    ├── quat.h         ← wp::quat_t、wp::quat_rotate、wp::quat_rotate_inv
    │                     wp::quat_from_axis_angle、四元数乘法（1682 行）
    │
    ├── spatial.h      ← wp::transform_t、wp::spatial_vector
    │                     wp::transform_point、wp::transform_vector
    │                     wp::spatial_top、wp::spatial_bottom（1304 行）
    │
    ├── array.h        ← wp::array_t、wp::array_store、wp::address、wp::load
    │                     数组的 GPU 内存访问封装（2227 行）
    │
    ├── builtin.h      ← wp::sin、wp::cos、wp::sqrt、wp::abs、wp::step
    │                     wp::min、wp::max、wp::clamp 等数学函数（2173 行）
    │
    ├── tile.h         ← Tile API（共享内存 tile 操作）（6182 行，最大的文件）
    │
    ├── mesh.h         ← wp::mesh_query_point、wp::mesh_eval_position（2699 行）
    ├── volume.h       ← wp::volume_sample_grad_f（SDF 体积采样）（1074 行）
    └── bvh.h          ← wp::bvh_query_aabb（BVH 空间查询）（587 行）

  举例：.cu 文件里看到 wp::dot(var_a, var_b)
  对应源码在 warp/native/vec.h 第 532 行：

    template <typename Type>
    inline CUDA_CALLABLE Type dot(vec_t<3, Type> a, vec_t<3, Type> b)
    {
        return a.c[0]*b.c[0] + a.c[1]*b.c[1] + a.c[2]*b.c[2];
    }

  反向传播版本在同一文件第 1798 行：

    inline CUDA_CALLABLE void adj_dot(...)   ← Warp 自动微分调用的梯度函数

  每个 wp:: 函数都有对应的 adj_ 版本，这就是 Warp 能自动微分的秘密：
    前向:  result = wp::dot(a, b)        → 算 a·b
    反向:  wp::adj_dot(a, b, adj_result) → 把 adj_result 的梯度传回 a 和 b

  你可以在 Cursor 中直接打开这些 .h 文件阅读：
    Ctrl+P → 输入 warp/native/vec.h → 回车


================================================================================
八、PTX 文件 —— GPU 汇编码
================================================================================

  .ptx 文件是 NVIDIA GPU 的汇编语言，类似 CPU 的汇编码。
  一般不需要直接阅读，但在做极致性能优化时有用。

  位置: /home/hkang/.cache/warp/1.12.0.dev20260127/
        wp_newton._src.solvers.solver_9972fda/
        wp_newton._src.solvers.solver_9972fda.sm89.ptx

  sm89 表示编译目标是 SM 8.9 架构（你的 RTX 5000 Ada 是 SM 8.9）。

  查看方式:
    less ...solver_9972fda.sm89.ptx

  PTX 里你会看到类似：
    .reg .f32 %f<200>;          // 浮点寄存器
    .reg .pred %p<30>;          // 条件寄存器
    ld.global.v4.f32 {...};     // 从显存加载 4 个 float
    fma.rn.f32 %f10, %f5, %f3, %f2;  // 乘加运算


================================================================================
八、实用技巧
================================================================================

  【技巧1：快速找到某个 kernel 的生成代码】
    在 .cu 文件中搜索你的 kernel 函数名：
    grep "你的函数名" /home/hkang/.cache/warp/1.12.0.dev20260127/wp_*/*.cu

  【技巧2：强制重新编译】
    删除缓存目录后重新运行：
    rm -rf /home/hkang/.cache/warp/1.12.0.dev20260127/
    uv run python your_script.py

  【技巧3：在 Cursor 中打开 .cu 文件】
    Ctrl+P → 输入路径 → 直接打开
    Cursor 支持 CUDA 语法高亮

  【技巧4：查看一个 kernel 到底用了多少寄存器】
    grep "\.reg" ...solver_9972fda.sm89.ptx | head -10

  【技巧5：比较两个版本的生成代码】
    修改 Python kernel → 重新运行 → 缓存目录会生成新的 hash 后缀目录
    用 diff 比较两个 .cu 文件看差异
"""

import os
import warp as wp

wp.init()


def main():
    cache_dir = wp.config.kernel_cache_dir
    print("=" * 60)
    print("Warp 生成代码查看工具")
    print("=" * 60)
    print(f"\n  缓存目录: {cache_dir}\n")

    if not cache_dir or not os.path.exists(cache_dir):
        print("  缓存目录不存在，请先运行一个 Newton 示例生成缓存")
        return

    # 列出所有生成的 .cu 文件
    entries = []
    for name in sorted(os.listdir(cache_dir)):
        full = os.path.join(cache_dir, name)
        if not os.path.isdir(full):
            continue
        cu_file = os.path.join(full, name + ".cu")
        if os.path.exists(cu_file):
            lines = sum(1 for _ in open(cu_file))
            ptx_count = len([f for f in os.listdir(full) if f.endswith(".ptx")])
            entries.append((lines, name, cu_file, ptx_count))

    entries.sort(reverse=True)

    print(f"  {'行数':>8s}  {'PTX':>4s}  模块名")
    print(f"  {'-'*8}  {'-'*4}  {'-'*50}")
    for lines, name, cu_file, ptx_count in entries[:20]:
        short = name.replace("wp_newton._src.", "newton/").replace("wp___main___", "__main__/")
        print(f"  {lines:>8d}  {ptx_count:>4d}  {short}")

    # 显示最重要的文件的完整路径
    print(f"\n  === 你可以用 Cursor 直接打开以下文件查看生成的 CUDA 代码 ===\n")

    interesting = ["solver_", "xpbd", "geometry.kernels", "collide"]
    for lines, name, cu_file, _ in entries:
        if any(k in name for k in interesting):
            short = name.replace("wp_newton._src.", "")
            print(f"  {short} ({lines} 行):")
            print(f"    {cu_file}")
            print()

    # 显示一个示例 kernel 的结构
    solver_entries = [e for e in entries if "solvers.solver_" in e[1] and "xpbd" not in e[1]]
    if solver_entries:
        _, _, cu_file, _ = solver_entries[0]
        print(f"  === 示例：solver.py 中生成的 kernel 列表 ===\n")

        with open(cu_file) as f:
            for i, line in enumerate(f, 1):
                if '__global__' in line and 'void' in line:
                    func_name = line.split('void')[1].split('(')[0].strip()
                    kind = "前向" if "forward" in func_name else "反向"
                    short_name = func_name.split('_cuda_kernel')[0]
                    short_name = short_name.rsplit('_', 1)[0]
                    print(f"    第 {i:>5d} 行: [{kind}] {short_name}")

    print(f"\n  提示: 在 .cu 文件中搜索 '<L ' 可以找到每行 CUDA 代码对应的 Python 源码行号")
    print(f"  提示: 搜索 'adj_' 前缀的函数是反向传播（梯度计算）代码")


if __name__ == "__main__":
    main()
