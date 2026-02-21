# Warp GPU 编程学习指南

## 学习路线（从基础到高级）

| 文件 | 主题 | 核心内容 |
|------|------|----------|
| `01_warp_basics.py` | Warp 基础 | `@wp.kernel`, `@wp.func`, `wp.array`, `wp.launch`, `wp.tid()` |
| `02_warp_types.py` | 类型系统 | `vec3`, `mat33`, `quat`, `transform`, `spatial_vector` |
| `03_physics_kernels.py` | 物理算子实战 | 半隐式欧拉积分、弹簧力(atomic_add)、SDF碰撞、刚体积分 |
| `04_autodiff.py` | 自动微分 | `wp.Tape`, `backward()`, `requires_grad`, 梯度下降优化 |
| `05_advanced_cuda.py` | 高级 CUDA | 原子操作、归约、CUDA图(`ScopedCapture`)、设备管理 |
| `06_codegen_inspector.py` | 代码生成 | 查看 Warp 生成的 CUDA/C++ 代码、前向/反向代码对比 |
| `07_profiling.py` | Profiling 工具库 | **可 import 复用**：NVTX 标记、gpu_timer、benchmark、Nsight 集成、结果导出 |
| `08_view_generated_cuda.py` | 查看 CUDA 源码 | 列出所有生成的 .cu 文件、Python→CUDA 逐行对照、前向/反向代码解析 |

## 运行方式

```bash
cd /home/hkang/newton_related/newton

# 按顺序学习
uv run python learning_warp/01_warp_basics.py
uv run python learning_warp/02_warp_types.py
uv run python learning_warp/03_physics_kernels.py
uv run python learning_warp/04_autodiff.py
uv run python learning_warp/05_advanced_cuda.py
uv run python learning_warp/06_codegen_inspector.py
uv run python learning_warp/07_profiling.py

# 运行全部
for f in learning_warp/0*.py; do echo "=== $f ===" && uv run python "$f" && echo; done
```

## 和 Newton 的关系

这些算子都是 Newton 物理引擎中实际使用的算子的简化版：

| 本教程算子 | Newton 源码位置 |
|-----------|----------------|
| `integrate_particles` | `newton/_src/solvers/solver.py` 第 22-59 行 |
| `eval_springs` | `newton/_src/solvers/semi_implicit/kernels_particle.py` 第 21-65 行 |
| `sphere_sdf` / `ground_sdf` | `newton/_src/geometry/kernels.py` 第 695-836 行 |
| `integrate_rigid_bodies` | `newton/_src/solvers/solver.py` 第 62-159 行 |
| `atomic_add` 力累积 | Newton 所有求解器中的力计算 |
| `ScopedCapture` CUDA 图 | Newton 所有示例中的 `capture()` 方法 |
