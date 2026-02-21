# Newton 物理引擎学习指南

## 项目概述

Newton 是一个基于 NVIDIA Warp 构建的 **GPU 加速物理引擎**，专为机器人仿真和研究设计。
它由 Disney Research、Google DeepMind 和 NVIDIA 联合发起，是 Linux Foundation 项目。

### 三个仓库的关系

```
┌─────────────────────────────────────────────────────────┐
│                    IsaacLab_newton                       │
│        (强化学习框架，集成 Newton 作为物理后端)            │
│   提供: 环境管理、资产加载、传感器、执行器、训练流程         │
└─────────────────────────┬───────────────────────────────┘
                          │ 调用
┌─────────────────────────▼───────────────────────────────┐
│                      Newton                              │
│         (GPU加速物理引擎，核心仿真逻辑)                    │
│   提供: 模型构建、碰撞检测、多种求解器、传感器、可视化      │
└─────────────────────────┬───────────────────────────────┘
                          │ 构建于
┌─────────────────────────▼───────────────────────────────┐
│                       Warp                               │
│        (GPU/CPU 并行计算框架，JIT编译Python到CUDA)         │
│   提供: @wp.kernel, wp.array, wp.launch, 自动微分         │
└─────────────────────────────────────────────────────────┘
```

## Newton 核心架构

### 目录结构

```
newton/
├── __init__.py          # 公共API入口（用户只从这里导入）
├── geometry.py          # 几何与碰撞检测 API
├── solvers.py           # 物理求解器 API
├── sensors.py           # 传感器 API
├── ik.py                # 逆运动学 API
├── math.py              # 数学工具 API
├── selection.py         # 选择工具 API (ArticulationView)
├── utils.py             # 实用工具 API
├── viewer.py            # 可视化 API
├── usd.py               # USD格式工具 API
├── _src/                # 【内部实现，用户代码不要导入】
│   ├── core/            # 核心类型（轴、空间数学）
│   ├── sim/             # 仿真核心（Model, State, Builder, Control, Contacts）
│   ├── geometry/        # 碰撞检测实现（宽相/窄相、SDF等）
│   ├── solvers/         # 求解器实现
│   │   ├── featherstone/  # Featherstone 关节求解器
│   │   ├── mujoco/        # MuJoCo 求解器
│   │   ├── semi_implicit/ # 半隐式求解器（支持可微仿真）
│   │   ├── xpbd/          # XPBD 求解器（位置级动力学）
│   │   ├── vbd/           # VBD 求解器
│   │   ├── style3d/       # Style3D 布料求解器
│   │   └── implicit_mpm/  # MPM 求解器（颗粒/流体）
│   ├── sensors/         # 传感器实现
│   ├── viewer/          # 可视化实现
│   └── utils/           # 工具函数实现
├── examples/            # 官方示例
└── tests/               # 测试套件
```

### 核心数据流

Newton 的仿真遵循以下核心流程：

```
             ┌──────────────────┐
             │   ModelBuilder   │  ← 第1步：在 Python 中构建场景
             │                  │     (CPU端，使用Python列表)
             │  add_body()      │
             │  add_joint()     │
             │  add_shape()     │
             │  add_particle()  │
             └────────┬─────────┘
                      │ finalize(device="cuda:0")
                      ▼
             ┌──────────────────┐
             │      Model       │  ← 第2步：生成GPU常驻模型
             │  (静态、不变的)     │     (wp.array 在 GPU 上)
             │                  │
             ├──→ state()      ─┼──→ State   (动态：位姿、速度、力)
             ├──→ control()    ─┼──→ Control (控制输入：目标位置/力)
             └──→ contacts()   ─┼──→ Contacts(碰撞结果缓冲区)
                                │
          ┌─────────────────────┼────────────────┐
          ▼                     ▼                ▼
    ┌──────────┐         ┌───────────┐    ┌──────────┐
    │  Solver  │         │ Collision │    │  Viewer  │
    │ (求解器)  │         │ Pipeline  │    │ (可视化)  │
    └──────────┘         └───────────┘    └──────────┘

每一仿真步：
  1. state.clear_forces()               # 清零外力
  2. collision_pipeline.collide(state)   # 碰撞检测
  3. solver.step(state_in → state_out)   # 物理积分
  4. swap(state_in, state_out)           # 交换状态缓冲区
```

### 核心类说明

| 类名 | 作用 | 关键属性 |
|------|------|----------|
| **ModelBuilder** | 场景构建器（CPU端） | `add_body()`, `add_joint_*()`, `add_shape_*()`, `finalize()` |
| **Model** | 静态仿真模型（GPU端） | `body_q`, `joint_q`, `shape_*`, `gravity` |
| **State** | 动态仿真状态 | `body_q`(位姿), `body_qd`(速度), `body_f`(力), `joint_q`, `joint_qd` |
| **Control** | 控制输入 | `joint_f`(力), `joint_target_pos`, `joint_target_vel` |
| **Contacts** | 碰撞结果 | `rigid_contact_*`, `soft_contact_*` |
| **CollisionPipeline** | 碰撞检测管线 | `collide(state, contacts)` |
| **SolverXPBD/MuJoCo/...** | 物理求解器 | `step(state_in, state_out, control, contacts, dt)` |

### 求解器选择指南

| 求解器 | 适用场景 | 特点 |
|--------|----------|------|
| **SolverXPBD** | 刚体、通用场景 | 默认选择，位置级动力学 |
| **SolverMuJoCo** | 关节机器人 | 精确关节求解，自带接触处理 |
| **SolverFeatherstone** | 关节动力学 | 递推牛顿-欧拉算法 |
| **SolverSemiImplicit** | 可微仿真 | 支持 `wp.Tape()` 反向传播 |
| **SolverVBD** | 布料/软体 | 需要 `builder.color()` 图着色 |
| **SolverStyle3D** | 布料仿真 | 各向异性刚度 |
| **SolverImplicitMPM** | 颗粒/流体/沙 | 物质点法 |

### Warp 基础概念（Newton 用户需要了解的）

| 概念 | 说明 | 示例 |
|------|------|------|
| `wp.vec3` | 3D向量 | `wp.vec3(1.0, 2.0, 3.0)` |
| `wp.quat` | 四元数(旋转) | `wp.quat_identity()`, `wp.quat_from_axis_angle(...)` |
| `wp.transform` | 位姿(位置+旋转) | `wp.transform(wp.vec3(...), wp.quat(...))` |
| `wp.spatial_vector` | 空间向量(线速度+角速度) | 6维：前3为角速度，后3为线速度 |
| `wp.array` | GPU数组 | `wp.zeros(n, dtype=wp.vec3)` |
| `@wp.kernel` | GPU并行核函数 | 使用 `wp.tid()` 获取线程ID |
| `wp.launch` | 启动核函数 | `wp.launch(kernel, dim=n, inputs=[...])` |
| `wp.Tape` | 自动微分磁带 | 记录操作用于反向传播 |

## 学习路线（推荐顺序）

### 阶段1：基础入门（先学会用）

1. **`01_basic_pendulum_annotated.py`** - 双摆仿真
   - 学习：ModelBuilder 构建场景、关节、状态双缓冲、仿真循环
   - 关键API：`add_link()`, `add_joint_revolute()`, `add_articulation()`

2. **`02_basic_shapes_annotated.py`** - 碰撞形状
   - 学习：所有碰撞几何体类型、碰撞管线
   - 关键API：`add_body()`, `add_shape_*()`, `CollisionPipeline`

3. **`03_basic_joints_annotated.py`** - 关节类型
   - 学习：REVOLUTE、PRISMATIC、BALL 关节、运动学体
   - 关键API：`add_joint_*()`, `ShapeConfig.density=0`

### 阶段2：进阶应用

4. **`04_robot_cartpole_annotated.py`** - 机器人仿真
   - 学习：USD导入、MuJoCo求解器、多世界复制
   - 关键API：`add_usd()`, `SolverMuJoCo`, `replicate()`

5. **`05_diffsim_ball_annotated.py`** - 可微仿真
   - 学习：可微物理、梯度优化、Warp Tape
   - 关键API：`finalize(requires_grad=True)`, `wp.Tape()`, `SolverSemiImplicit`

### 阶段3：源码级深入理解

6. **`06_collision_pipeline_internals.py`** - 碰撞检测管线源码解析
   - 学习：AABB计算、宽相(NXN/SAP/EXPLICIT)、窄相(GJK/MPR)、接触归约
   - 关键源码：`_src/sim/collide.py`, `_src/geometry/narrow_phase.py`

7. **`07_solver_internals.py`** - 求解器内部物理方程解析
   - 学习：XPBD位置投影、半隐式力累积、Neo-Hookean弹性、关节力
   - 关键源码：`_src/solvers/xpbd/`, `_src/solvers/semi_implicit/`

8. **`08_softbody_cloth_particles.py`** - 软体/布料/粒子系统解析 (可运行!)
   - 学习：三角面片FEM、四面体Neo-Hookean、弯曲约束、MPM方法
   - 关键源码：`_src/sim/builder.py`(add_cloth_grid), `_src/solvers/implicit_mpm/`

9. **`09_builder_finalize_internals.py`** - Builder.finalize() 完整流程解析
   - 学习：CPU→GPU转换、AABB缓存、SDF生成、碰撞对预计算、惯性验证
   - 关键源码：`_src/sim/builder.py`(finalize方法)

### 阶段4：进一步阅读源码

10. 阅读 `newton/_src/sim/model.py` - 理解 Model 的数据结构和世界分组系统
11. 阅读 `warp/_src/context.py` - 理解 Warp 的 JIT 编译和设备管理

### 阶段5：IsaacLab 集成

12. 阅读 `IsaacLab_newton/source/isaaclab_newton/` - 理解 Newton 如何作为 IsaacLab 的后端
13. 关注 `newton_manager.py` - NewtonManager 单例管理仿真生命周期
14. 理解 Isaac Lab 的资产系统如何映射到 Newton 的 ModelBuilder

## 运行示例

```bash
# 设置环境
cd /path/to/newton_my
uv sync --extra examples

# 运行官方示例
uv run -m newton.examples basic_pendulum
uv run -m newton.examples basic_shapes
uv run -m newton.examples basic_joints

# 运行学习版注释示例（独立脚本）
uv run python learning/01_basic_pendulum_annotated.py
uv run python learning/02_basic_shapes_annotated.py
```

## 常见模式总结

### 1. 标准仿真循环模式
```python
# 初始化
builder = newton.ModelBuilder()
# ... 构建场景 ...
model = builder.finalize()
solver = newton.solvers.SolverXPBD(model)
state_0, state_1 = model.state(), model.state()
control = model.control()
collision_pipeline = newton.CollisionPipeline(model)
contacts = collision_pipeline.contacts()

# 仿真循环
for substep in range(sim_substeps):
    state_0.clear_forces()
    collision_pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, sim_dt)
    state_0, state_1 = state_1, state_0  # 交换
```

### 2. CUDA 图捕获模式（性能优化）
```python
if wp.get_device().is_cuda:
    with wp.ScopedCapture() as capture:
        simulate()  # 捕获整个仿真循环
    graph = capture.graph
# 之后每帧：
wp.capture_launch(graph)  # 零Python开销重放
```

### 3. 多世界复制模式
```python
single_robot = newton.ModelBuilder()
# ... 构建单个机器人 ...
scene = newton.ModelBuilder()
scene.replicate(single_robot, num_worlds=100)
model = scene.finalize()
```
