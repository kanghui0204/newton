# Newton 物理引擎学习指南

## Newton 代码规模统计

### Newton 引擎（Python）—— 18 万行

| 模块 | 行数 | 说明 |
|------|------|------|
| `newton/_src/solvers/` | 33,051 | 物理求解器（7 种） |
| `newton/_src/geometry/` | 21,961 | 碰撞检测（宽相/窄相/SDF） |
| `newton/_src/sim/` | 17,633 | 核心仿真（Builder/Model/State） |
| `newton/_src/viewer/` | 11,063 | 可视化 |
| `newton/_src/utils/` | 9,677 | 工具函数 |
| `newton/_src/sensors/` | 4,692 | 传感器 |
| `newton/_src/core/` | 692 | 核心类型 |
| `newton/examples/` | 19,216 | 官方示例 |
| `newton/tests/` | 59,696 | 测试套件 |
| **合计** | **~181,000** | |

### 求解器详细

| 求解器 | 行数 | 用途 |
|--------|------|------|
| VBD | 8,515 | 布料/软体（隐式级精度） |
| MuJoCo | 6,892 | 关节机器人（桥接 mujoco-warp） |
| Implicit MPM | 5,668 | 颗粒/流体/沙 |
| Style3D | 3,920 | 布料（各向异性） |
| XPBD | 3,172 | 通用刚体+布料（位置级） |
| Featherstone | 2,395 | 关节动力学（广义坐标） |
| SemiImplicit | 2,058 | 可微仿真（最简单） |

### Warp 框架 —— 26 万行

| 语言 | 行数 | 说明 |
|------|------|------|
| Python | 190,000 | JIT 编译器、类型系统、自动微分 |
| C++/CUDA | 70,000 | GPU 运行时（vec.h/mat.h/quat.h 等） |
| **合计** | **~260,000** | |

---

## 环境安装

### 前置要求

| 项目 | 最低要求 | 说明 |
|------|----------|------|
| **操作系统** | Linux (Ubuntu 22.04+) | 也支持 Windows/macOS，但 GPU 功能仅限 Linux/Windows |
| **Python** | 3.10+ | uv 会自动管理 Python 版本 |
| **NVIDIA 驱动** | 535+ | 运行 `nvidia-smi` 确认驱动版本 |
| **CUDA** | 12.0+ | Warp 自带 CUDA 运行时，无需单独安装 CUDA Toolkit |
| **GPU 显存** | 4 GB+ | 基础示例所需；多世界复制需要更多 |

### 安装步骤

```bash
# 1. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 同步依赖
cd /home/hkang/newton_related/newton
uv sync --extra examples

# 3. 验证
uv run python -c "import newton; import warp as wp; wp.init(); print('OK')"
```

---

## learning 文件夹总览

### 可运行的示例（7 个）

这些文件包含 `class Example`，可以直接运行并看到可视化效果。

| 文件 | 行数 | 主题 | 运行命令 |
|------|------|------|---------|
| `01_basic_pendulum_annotated.py` | 370 | 双摆仿真 | `uv run python learning/01_basic_pendulum_annotated.py` |
| `02_basic_shapes_annotated.py` | 344 | 碰撞形状 | `uv run python learning/02_basic_shapes_annotated.py` |
| `03_basic_joints_annotated.py` | 306 | 关节类型 | `uv run python learning/03_basic_joints_annotated.py` |
| `04_robot_cartpole_annotated.py` | 257 | 机器人仿真 | `uv run python learning/04_robot_cartpole_annotated.py` |
| `05_diffsim_ball_annotated.py` | 634 | 可微仿真 | `uv run python learning/05_diffsim_ball_annotated.py` |
| `08_softbody_cloth_particles.py` | 602 | 布料落球 | `uv run python learning/08_softbody_cloth_particles.py` |

### 源码解析文档（8 个，只读不运行）

这些文件是对 Newton 内部源码的逐层解析，在编辑器中阅读即可。

| 文件 | 行数 | 主题 | 对应源码 |
|------|------|------|---------|
| `06_collision_pipeline_internals.py` | 313 | 碰撞检测管线 | `_src/sim/collide.py` |
| `07_solver_internals.py` | 334 | 求解器物理方程 | `_src/solvers/xpbd/`, `_src/solvers/semi_implicit/` |
| `09_builder_finalize_internals.py` | 271 | `finalize()` 流程 | `_src/sim/builder.py` |
| `10_warp_types_and_joints_explained.py` | 354 | Warp 类型与关节 | 参考手册 |
| `11_solver_deep_dive.py` | 368 | 求解器深层分析 | 各求解器实现 |
| `12_joints_and_articulation_reference.py` | 260 | 关节参考 | 参考手册 |
| `13_body_q_vs_shape_transform_and_warp_tiles.py` | 216 | 坐标变换 | 坐标系解析 |
| `14_broad_phase_explicit_explained.py` | 388 | 宽相碰撞 | `_src/geometry/broad_phase_*.py` |

### 知识文档（4 个，专题讲解）

这些文件不是源码解析，而是专题知识讲解，配有 ASCII 图解和对比表格。

| 文件 | 行数 | 主题 | 核心内容 |
|------|------|------|---------|
| `15_numerical_methods_explained.py` | 656 | 数值方法全解 | 显式/半隐式/隐式欧拉、FDM/FVM/FEM、牛顿-欧拉方程 |
| `16_viewer_ui_guide.py` | 295 | Viewer 界面指南 | 左侧面板、键盘鼠标操作、右上角信息 |
| `17_control_and_soft_contacts.py` | 231 | Control 与软接触 | Control 属性、Rigid vs Soft Contact、margin |
| `18_model_pipeline_explained.py` | 516 | Model 全流程 | Builder→finalize→Model→State→仿真循环，含源码行号 |

---

## 推荐学习顺序

### 阶段1：学会使用（运行示例）

1. `01_basic_pendulum_annotated.py` — Builder→Model→State→Solver 完整流程
2. `02_basic_shapes_annotated.py` — 所有碰撞几何体
3. `03_basic_joints_annotated.py` — 关节类型

### 阶段2：进阶应用

4. `04_robot_cartpole_annotated.py` — USD 导入、MuJoCo 求解器
5. `05_diffsim_ball_annotated.py` — 可微仿真、梯度优化
6. `08_softbody_cloth_particles.py` — 布料仿真

### 阶段3：理解架构（先读知识文档，再对照源码）

7. `18_model_pipeline_explained.py` — **最重要**，理解 Model 全流程
8. `15_numerical_methods_explained.py` — 理解求解器背后的数学
9. `17_control_and_soft_contacts.py` — 理解 Control 和碰撞接触
10. `16_viewer_ui_guide.py` — 理解可视化界面

### 阶段4：深入源码

11. `06_collision_pipeline_internals.py` — 碰撞管线源码
12. `07_solver_internals.py` — 求解器源码
13. `09_builder_finalize_internals.py` — finalize() 源码
14. `10`~`14` — 参考手册，需要时查阅

---

## Newton 核心架构

### 三个仓库的关系

```
┌─────────────────────────────────────────────────────────┐
│                    IsaacLab_newton                       │
│        (强化学习框架，集成 Newton 作为物理后端)            │
└─────────────────────────┬───────────────────────────────┘
                          │ 调用
┌─────────────────────────▼───────────────────────────────┐
│                      Newton (18万行 Python)              │
│         (GPU加速物理引擎，核心仿真逻辑)                    │
└─────────────────────────┬───────────────────────────────┘
                          │ 构建于
┌─────────────────────────▼───────────────────────────────┐
│                  Warp (19万 Python + 7万 C++/CUDA)       │
│        (GPU/CPU 并行计算框架，JIT编译Python到CUDA)         │
└─────────────────────────────────────────────────────────┘
```

### 核心数据流

```
ModelBuilder (CPU)          Model (GPU)           State (GPU)
───────────────           ──────────            ───────────
Python 列表               wp.array              wp.array
add_body()          →     body_mass       →     body_q (位姿)
add_joint()         →     joint_type      →     body_qd (速度)
add_shape()         →     shape_transform →     body_f (力)
add_particle()      →     particle_mass   →     particle_q
                                          →     particle_qd
     finalize()                           →     particle_f
     CPU→GPU                              →     joint_q
                                          →     joint_qd

仿真循环:
  state.clear_forces()   → 清零力
  pipeline.collide()     → 碰撞检测 → 填充 Contacts
  solver.step()          → state_in → state_out
  swap                   → 交换双缓冲
```

### 求解器选择指南

| 求解器 | 适用场景 | 时间积分 | 空间方法 |
|--------|----------|---------|---------|
| **SolverXPBD** | 通用刚体+布料 | 位置投影 | 弹簧约束 |
| **SolverSemiImplicit** | 可微仿真 | 半隐式欧拉 | 三角形 FEM |
| **SolverFeatherstone** | 关节机器人 | 半隐式欧拉 | 广义坐标 |
| **SolverMuJoCo** | 关节机器人 | MuJoCo 内部 | 关节坐标 |
| **SolverVBD** | 布料/软体 | VBD 迭代 | 四面体 FEM |
| **SolverStyle3D** | 布料 | 全隐式欧拉 | 投影动力学 |
| **SolverImplicitMPM** | 颗粒/流体/沙 | 全隐式 | MPM+FEM |

---

## 在源码中增加注释来深入理解

### 方法1：print 调试

```python
# 在 newton/_src/sim/collide.py 的 collide() 方法中
def collide(self, state, contacts):
    print(f"[DEBUG] num_shapes={self.model.shape_count}")
```

### 方法2：.numpy() 打印 GPU 数据

```python
print("body positions:", state_0.body_q.numpy())
solver.step(state_0, state_1, control, contacts, dt)
print("body positions after:", state_1.body_q.numpy())
```

### 方法3：Git 跟踪笔记

```bash
git diff                    # 查看你加的注释
git add -A && git commit -m "Add learning notes"
```

---

## 常见模式

### 标准仿真循环

```python
builder = newton.ModelBuilder()
# ... 构建场景 ...
model = builder.finalize()
solver = newton.solvers.SolverXPBD(model)
state_0, state_1 = model.state(), model.state()
control = model.control()
pipeline = newton.CollisionPipeline(model)
contacts = pipeline.contacts()

for substep in range(sim_substeps):
    state_0.clear_forces()
    pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, sim_dt)
    state_0, state_1 = state_1, state_0
```

### CUDA 图加速

```python
with wp.ScopedCapture() as capture:
    simulate()
graph = capture.graph
# 之后每帧：
wp.capture_launch(graph)  # 零 Python 开销
```

### 多世界复制

```python
single = newton.ModelBuilder()
# ... 构建单个机器人 ...
scene = newton.ModelBuilder()
scene.replicate(single, num_worlds=100)
model = scene.finalize()
```
