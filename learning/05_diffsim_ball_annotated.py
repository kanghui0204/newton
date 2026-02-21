"""
Newton 学习示例 05：可微仿真 (Differentiable Simulation)
========================================================

【学习目标】
- 理解可微物理仿真的核心概念
- 学习 Warp 的 Tape（自动微分磁带）机制
- 理解为什么需要 requires_grad=True
- 学习粒子系统（particle）仿真
- 理解基于梯度的轨迹优化

【可微仿真到底有什么用？ — 白话解释】

  想象你在打台球：
  - 白球位置固定（初始位置）
  - 你要选择"往哪打、用多大力"（初始速度）
  - 目标：让白球弹墙后进袋

  传统方法：手动试几百种力度和方向 → 太慢
  可微仿真：
    1. 随便打一杆 v₀=(0,5,-5)，球落在 (0.5,-1,0.3)，离目标差3米
    2. 自动计算"调整方向"：∂loss/∂v₀ = (-0.2, 0.8, -1.5)
       → "y方向力稍微减、z方向力稍微加，球会更接近目标"
    3. 按梯度调整 v₀，再打一杆，离目标差1米
    4. 重复100次，球精准进袋！

  本质：和训练神经网络是同一个方法（梯度下降）！

  更实际的应用：
  - 机器人控制：优化每一步施加的关节力矩(joint_f)，让机器人走路
    （注意：不是优化kp/kd。kp/kd是控制器"脾气"，力矩是控制器"动作"）
  - 系统辨识：从实验数据反推材料参数（摩擦系数、弹簧刚度等）
  - 机构设计：优化连杆长度、关节位置
  - 强化学习：策略网络输出力矩→仿真→梯度反传到网络参数

  可微的参数有哪些？
    任何 requires_grad=True 的 wp.array 都可以被微分：
    - 初始速度/位置: particle_qd, particle_q
    - 控制力矩: control.joint_f, control.joint_target_pos
    - 物理参数: ke/kd/mu (需要手动设置 requires_grad)
    - 神经网络权重（PyTorch互操作时）
    本例中可微参数 = states[0].particle_qd（粒子初始速度）

  可微仿真 vs PPO 强化学习（效率差异）：
    PPO: 蒙着眼找路，随机尝试几百万次，靠"碰巧走对"来学习
    可微: 睁着眼找路，梯度告诉你"目标在左前方"，几百步到达
    差距取决于问题，简单问题10-100倍，复杂问题更大
    但可微仿真也有局限：长序列梯度爆炸/消失，不连续碰撞梯度不准

【为什么要改变初始状态？我不能固定初始速度吗？】

  当然可以固定！但如果你固定了初始速度 v₀=(0,5,-5)，
  球弹墙后落在 (0.5,-1,0.3)，不在目标 (0,-2,1.5) 上。
  你怎么让球打到目标？ → 只能改变初始速度！

  这个例子中"初始速度"就是要优化的"参数"。
  正是因为不知道该用什么速度，才需要优化来找到。

【SolverSemiImplicit 为什么能完全支持可微仿真？】

  半隐式求解器的计算图是"单向直线"，没有循环：

    x₀,v₀ → 算力F(x,v) → v₁=v₀+F/m×dt → x₁=x₀+v₁×dt → 算力 → ...

  每一步只依赖上一步，Warp Tape 可以完美记录这条链，
  backward() 沿链反向传播梯度。

  XPBD 的计算图有迭代循环（同一个变量被反复读写），
  梯度追踪更复杂。虽然也支持 requires_grad，但效率低。

  源码: newton/_src/solvers/semi_implicit/solver_semi_implicit.py
  力的计算: kernels_particle.py (弹簧/三角/tet)
           kernels_contact.py (接触力)

【碰撞检测怎么可微？— 详细解释】

  先理解 SDF (Signed Distance Field，有符号距离场)：
    对空间中任意一点，SDF 返回"这个点到物体表面的带符号距离"。

    SDF(点) > 0 → 点在物体外面，值 = 到表面的距离
    SDF(点) = 0 → 点恰好在表面上
    SDF(点) < 0 → 点在物体里面（穿透），值 = 穿透深度（负数）

    每种形状都有解析 SDF 公式：
      球体: SDF(p) = ||p - center|| - radius
      盒子: SDF(p) = max(|px|-hx, |py|-hy, |pz|-hz)
      平面: SDF(p) = dot(p - plane_point, normal)
      网格: SDF 预计算在 3D 体素网格中 (NanoVDB)

    SDF 源码: newton/_src/geometry/kernels.py
              → sphere_sdf(), box_sdf(), capsule_sdf(), plane_sdf() 等
              SDF 生成: newton/_src/geometry/sdf_mc.py

  为什么 SDF 让碰撞可微？

    传统碰撞（不可微）：
      if 穿透了: f = 推力       ← 阶跃函数，有"跳变"
      else:      f = 0            不连续，无法求导

    软接触（可微）：
      d = SDF(粒子位置) - 粒子半径   ← SDF 是连续函数
      f = ke × max(-d, 0)            ← max(x,0) 是连续函数

      当粒子从外面靠近表面：
        d = +0.5 → f = 0           (在外面，没力)
        d = +0.1 → f = 0           (快到了，还没力)
        d =  0.0 → f = 0           (刚好接触，力=0)
        d = -0.1 → f = ke × 0.1    (穿透了，有推力)
        d = -0.5 → f = ke × 0.5    (穿透更深，力更大)

      力随距离连续变化，没有跳变 → 可以对位置求导！

  完整梯度链（本例中）：
    v₀(初始速度) → 积分 → x(位置) → SDF(x) → d → f=ke×max(-d,0)
    → 速度改变(弹跳) → 继续积分 → xₙ(最终位置) → loss=||xₙ-target||²

    反向传播沿这条链计算 ∂loss/∂v₀
    其中 ∂f/∂x = ke × ∇SDF(x) = ke × 表面法线方向

  所以 requires_grad=True 的 CollisionPipeline：
    - 跳过 rigid contacts（不可微，enable_backward=False）
    - 只计算 soft contacts（通过 SDF + 惩罚力实现可微）
    - SDF 查询、力计算全部被 Tape 记录，支持反向传播

【为什么每个时间步需要独立的 State？】

  普通仿真：state_0 ↔ state_1 交替使用，旧数据被覆盖
  可微仿真：states[0] → states[1] → ... → states[N] 全部保留

  为什么？反向传播需要所有中间状态！
    前向: x₀ → x₁ → x₂ → ... → xₙ → loss
    反向: 算 ∂loss/∂xₙ₋₁ 时需要 xₙ₋₁ 的值
          如果 xₙ₋₁ 被覆盖了就没法算了！

【Warp Tape 机制】
wp.Tape() 类似 PyTorch 的 autograd：
- 记录所有 wp.launch() 操作
- tape.backward(loss) 自动执行反向传播
- 每个有 requires_grad=True 的 wp.array 都有 .grad 属性

【与普通仿真的关键区别】
1. finalize(requires_grad=True) - 启用梯度追踪
2. 每个时间步一个独立的 State - 不交换、不覆盖（给反向传播用）
3. 使用 SolverSemiImplicit - 计算图最简单，梯度追踪最高效
4. CollisionPipeline(requires_grad=True) - 只算软接触（可微），跳过刚体接触

【运行方式】
    uv run python learning/05_diffsim_ball_annotated.py
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.tests.unittest_utils import assert_np_equal
from newton.utils import bourke_color_map


# ===================================================================
# 【Warp 核函数】
# ===================================================================
# @wp.kernel 定义 GPU 并行核函数
# 这些函数在 GPU 上并行执行，每个线程处理一个元素

@wp.kernel
def loss_kernel(pos: wp.array(dtype=wp.vec3), target: wp.vec3, loss: wp.array(dtype=float)):
    """计算粒子到目标的距离平方作为损失。

    这个核函数在1个线程上执行（dim=1）
    loss = ||pos[0] - target||²

    注意：核函数不能 return 值，必须写入输出数组
    """
    delta = pos[0] - target
    loss[0] = wp.dot(delta, delta)  # 点积 = 距离平方


@wp.kernel
def step_kernel(x: wp.array(dtype=wp.vec3), grad: wp.array(dtype=wp.vec3), alpha: float):
    """梯度下降更新步。

    x[i] = x[i] - alpha * grad[i]

    wp.tid() 返回当前线程的索引
    """
    tid = wp.tid()
    x[tid] = x[tid] - grad[tid] * alpha


class Example:
    def __init__(self, viewer, args=None, verbose=False):
        self.fps = 60
        self.frame = 0
        self.frame_dt = 1.0 / self.fps
        self.sim_steps = 36          # 总仿真步数
        self.sim_substeps = 8        # 每步内子步数
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.verbose = verbose

        # 优化参数
        self.train_iter = 0         # 当前训练迭代次数
        self.train_rate = 0.02      # 学习率（梯度下降步长）
        self.target = (0.0, -2.0, 1.5)  # 目标位置

        # 损失值（标量数组，requires_grad=True 才能被反向传播）
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.loss_history = []

        self.viewer = viewer
        self.viewer.show_particles = True  # 在查看器中显示粒子

        # ===================================================================
        # 【场景构建】粒子 + 墙壁 + 地面
        # ===================================================================
        scene = newton.ModelBuilder(up_axis=newton.Axis.Z)

        # 添加一个粒子 — 对，就一个！
        #
        # 粒子 vs 刚体：
        #   粒子: 一个"点"，只有位置(vec3)+速度(vec3)，没有旋转，没有形状
        #         碰撞时用 particle_radius 当作"小球"处理
        #         占内存小，计算轻，适合大量（布料几千个、流体几万个）
        #   刚体: 有位姿(transform)+形状(Shape)，有旋转有碰撞几何
        #         占内存大，计算重，适合少量（几十个机器人连杆）
        #
        # 粒子不需要 add_shape！它天然就是一个"半径=particle_radius"的球
        # 碰撞检测: d = SDF(粒子位置) - particle_radius
        scene.add_particle(
            pos=wp.vec3(0.0, -0.5, 1.0),   # 初始位置
            vel=wp.vec3(0.0, 5.0, -5.0),    # 初始速度 ← 这就是我们要优化的参数！
            mass=1.0,
        )

        # 添加墙壁和地面（作为固定形状）
        # body=-1 表示这些形状固定在世界中（不参与动力学）
        ke = 1.0e4   # 接触弹簧刚度
        kf = 0.0     # 接触摩擦刚度
        kd = 1.0e1   # 接触阻尼
        mu = 0.2     # 摩擦系数

        # 墙壁：一个 box 形状，body=-1（固定到世界）
        scene.add_shape_box(
            body=-1,  # -1 = 世界固定体
            xform=wp.transform(wp.vec3(0.0, 2.0, 1.0), wp.quat_identity()),
            hx=1.0, hy=0.25, hz=1.0,
            cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu),
        )

        # 地面
        scene.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu),
        )

        # ===================================================================
        # 【关键】finalize(requires_grad=True)
        # ===================================================================
        # requires_grad=True 使所有 State 中的数组支持梯度追踪
        # 这意味着 state.particle_q, state.particle_qd 等都有 .grad 属性
        self.model = scene.finalize(requires_grad=True)

        # 设置软接触参数（粒子与形状之间的接触力）
        self.model.soft_contact_ke = ke
        self.model.soft_contact_kf = kf
        self.model.soft_contact_kd = kd
        self.model.soft_contact_mu = mu
        self.model.soft_contact_restitution = 1.0  # 弹性碰撞系数

        # ===================================================================
        # 【关键】使用 SolverSemiImplicit
        # ===================================================================
        # 源码: newton/_src/solvers/semi_implicit/solver_semi_implicit.py
        #
        # 半隐式欧拉积分（最简单的积分方式）：
        #   v_new = v + (F/m + gravity) × dt   ← 力→速度
        #   x_new = x + v_new × dt              ← 速度→位置
        #
        # step() 内部做的事：
        #   1. 算所有力（弹簧力、接触力、三角力...）→ 累积到 particle_f
        #   2. 用总力做一次积分
        #   就这两步！没有迭代。
        #
        # 为什么它最适合可微仿真？
        #   计算图是"单向直线": x₀→F(x₀)→v₁→x₁→F(x₁)→v₂→x₂→...
        #   没有循环、没有迭代 → Tape 完美记录 → backward() 高效反传
        #
        #   XPBD 有迭代循环（同一变量被读写多次）→ 梯度追踪更复杂
        self.solver = newton.solvers.SolverSemiImplicit(self.model)

        # ===================================================================
        # 【关键】每个时间步一个独立的 State
        # ===================================================================
        # 与普通仿真不同（只用2个State做双缓冲），
        # 可微仿真需要保留所有中间状态用于反向传播
        # 总状态数 = sim_steps * sim_substeps + 1（初始状态）
        self.states = [self.model.state() for _ in range(self.sim_steps * self.sim_substeps + 1)]

        # ===================================================================
        # 【补充】Control（控制输入）
        # ===================================================================
        # Control 是你给仿真系统发的"命令"：
        #   - joint_f:          直接给关节施加力/力矩
        #   - joint_target_pos: 关节目标位置（PD 控制器用）
        #   - joint_target_vel: 关节目标速度
        #   - tri_activations:  三角形元素激活力（肌肉布料用）
        #   - tet_activations:  四面体元素激活力（软体机器人用）
        #
        # 本示例只有一个自由粒子，没有关节，所以 control 里全是 None/空的。
        # 它出现在这里只是因为 solver.step() 接口要求传这个参数。
        # 如果是机器人场景（如 04_robot_cartpole），control 才真正起作用。
        #
        # 源码位置: newton/_src/sim/control.py（全文件只有 93 行，很短）
        self.control = self.model.control()

        # ===================================================================
        # 【补充】Rigid Contact vs Soft Contact + soft_contact_margin
        # ===================================================================
        #
        # Newton 有两种碰撞接触方式：
        #
        # 【Rigid Contact（硬接触）】—— 刚体之间的碰撞
        #   检测方法: GJK/MPR 精确几何检测
        #   接触力:   约束投影（直接修正位置，防止穿透）
        #   可微？    ❌ 不可微（"是否穿透"是 0/1 的阶跃函数，导数为 0 或无穷大）
        #
        # 【Soft Contact（软接触）】—— 粒子与碰撞体之间的接触
        #   检测方法: SDF（有符号距离场）查询，算出粒子到碰撞体的距离 d
        #   接触力:   弹簧力模型 f = ke × max(-d, 0)
        #             d > 0（没碰到）→ 力 = 0
        #             d = 0（刚贴着）→ 力 = 0
        #             d < 0（穿进去）→ 力 > 0，把粒子推出来
        #             就是一根弹簧：穿得越深，弹力越大
        #   可微？    ✅ 可微！距离 d 是连续函数，力也是连续函数
        #
        # 在 requires_grad=True 时，Newton 会：
        #   - 跳过 rigid contacts（不可微，不录入 Tape）
        #   - 只计算 soft contacts（可微，能被 tape.backward() 反传）
        #
        # -------------------------------------------------------------------
        # soft_contact_margin = 碰撞检测的"感知范围"
        # -------------------------------------------------------------------
        #
        # margin 控制"多远的距离内才去算距离 d"。
        # 算 d 需要计算量，所以 Newton 只对"足够近"的粒子-碰撞体对计算。
        #
        #   margin = 0.01（默认）：
        #     粒子距墙 5 米 → 超出 0.01 米感知范围 → 不算 d → 梯度 = 0
        #     优化器看不到任何方向信息，只能瞎猜
        #
        #   margin = 10.0（本示例用的）：
        #     粒子距墙 5 米 → 在 10 米感知范围内 → 算出 d = 5
        #     虽然 d > 0 所以力 = 0，但 d 本身是有梯度的！
        #     优化器能"闻到"墙的气味，知道往哪个方向调参数
        #
        # 为什么可微仿真要大 margin？
        #   margin = 0.01 时：
        #     轨迹: ●→→→→→→→→→→→→→→→→→●（飞过头了）
        #     梯度:  0  0  0  0  0  0  0  0  ← 全是0！优化器不知道该怎么调
        #
        #   margin = 10.0 时：
        #     轨迹: ●→→→→→→→→→→→→→→→→→●（同样飞过头了）
        #     梯度:  0  0 0.1 0.3 0.5 0.2 0  ← 经过墙附近时有梯度信号！
        #     优化器知道"应该往那个方向偏一点"
        #
        # 【常见疑问：margin 是"碰撞的门槛"吗？】
        #
        #   不是！margin 是"要不要算 d"的门槛，不是"碰不碰"的门槛。
        #   碰撞永远是 d < 0 才产生力：
        #
        #     距离 d = 5米, margin = 0.01:
        #       5 > 0.01 → 不算 d → 跳过 → 力 = 0，梯度 = 0
        #
        #     距离 d = 5米, margin = 10:
        #       5 < 10 → 算 d = 5 → d > 0 → 力 = 0（没穿透不产生力）
        #       但 d = 5 这个值对粒子位置可微 → 梯度 ≠ 0！
        #
        #     距离 d = -0.1米（穿进去了）, 不管 margin 多大：
        #       算 d = -0.1 → d < 0 → 力 = ke × 0.1（推出去）
        #
        # 【常见疑问：大于 margin 的距离完全没有梯度，怎么办？】
        #
        #   实际上不是问题！可微仿真的梯度主要来自两个来源：
        #
        #   来源1: 损失函数（永远有梯度）
        #     loss = ||最终位置 - 目标||²
        #     不管碰没碰到，loss 对 v0 的梯度永远存在
        #     优化器至少知道"最终位置偏了多远、偏向哪边"
        #
        #   来源2: 碰撞力的梯度（margin 内才有，锦上添花）
        #     margin 大 → 粒子经过墙附近时额外获得方向信息
        #     → 收敛更快、方向更准
        #
        #   所以 margin=10 不是"必须的"，而是"加速收敛的"。
        #   即使 margin=0.01，loss 的梯度仍然能让优化收敛，只是可能更慢。
        #
        # 总结:
        #   - margin 是"要不要计算距离 d"的门槛
        #   - 碰撞力的门槛永远是 d < 0（穿透了才有力）
        #   - 普通仿真: margin 小（0.01），省计算量
        #   - 可微仿真: margin 大（10.0），给优化器额外的方向信息，加速收敛
        #   - 本示例只有 1 个粒子，margin 大不会增加多少计算量
        #
        # 源码位置: newton/_src/geometry/kernels.py 第 696-835 行
        #           create_soft_contacts() 核函数
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase_mode=newton.BroadPhaseMode.EXPLICIT,
            soft_contact_margin=10.0,
            requires_grad=True,
        )
        self.contacts = self.collision_pipeline.contacts()
        # 对于简单场景（墙壁和地面都固定不动），碰撞对不会变化
        # 所以只需检测一次，之后每步复用相同的 contacts
        # （如果碰撞体会移动，则需要每步重新 collide）
        self.collision_pipeline.collide(self.states[0], self.contacts)

        self.viewer.set_model(self.model)
        self.capture()

    # ===================================================================
    # 【补充】下面所有函数的调用关系全景图
    # ===================================================================
    #
    # 程序入口（文件最后 if __name__ == "__main__"）：
    #   1. Example.__init__()     ← 构建场景（你现在在看的）
    #   2. check_grad()           ← 一次性梯度校验（启动前自测）
    #   3. newton.examples.run()  ← 进入主循环（窗口开着就一直跑）
    #
    # check_grad() 为什么在 run() 之前？
    #   它是一个自测：用两种方法算同一个梯度，看结果是否一致
    #   方法A（笨但可靠）：有限差分 (f(x+ε)-f(x-ε))/(2ε)
    #   方法B（快但可能有bug）：Tape 自动微分
    #   两个结果接近 → 代码正确 ✅
    #   终端打印的 "numeric grad" 和 "analytic grad" 就是这一步的输出
    #
    # newton.examples.run() 的内部（源码: newton/examples/__init__.py 第 173 行）：
    #   while viewer.is_running():    # 窗口没关就一直循环
    #       if not viewer.is_paused():
    #           example.step()        # 执行一步优化
    #       example.render()          # 渲染画面
    #
    # 函数嵌套关系：
    #
    #   step()                           每帧调一次（= 一次优化迭代）
    #    │
    #    ├── forward_backward()          前向+反向传播
    #    │    │
    #    │    ├── forward()              前向：跑完整个仿真轨迹
    #    │    │    │
    #    │    │    ├── simulate(0)       第 1 个时间步（内含 8 个子步）
    #    │    │    ├── simulate(1)       第 2 个时间步
    #    │    │    ├── ...
    #    │    │    ├── simulate(35)      第 36 个时间步
    #    │    │    └── loss_kernel()     算 loss = ||最终位置 - 目标||²
    #    │    │
    #    │    └── tape.backward(loss)    反向传播：从 loss 倒推所有梯度
    #    │
    #    └── step_kernel()               x = x - lr * gradient（更新初始速度）
    #
    # 一次 step() = "用当前参数跑一遍仿真 → 算梯度 → 更新参数"
    # 画面里每条新的彩色轨迹线，就是一次 step() 的结果

    def capture(self):
        """捕获前向+反向传播为 CUDA 图。"""
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.forward_backward()
            self.graph = capture.graph
        else:
            self.graph = None

    def forward_backward(self):
        """执行一次完整的前向传播 + 反向传播。

        Warp Tape 的工作流程：
        1. 创建 Tape 对象
        2. 在 with tape: 块内执行前向计算
        3. tape.backward(loss) 执行反向传播
        4. 每个 requires_grad 的数组的 .grad 被填充
        """
        self.tape = wp.Tape()
        with self.tape:
            self.forward()       # 前向传播：运行仿真 + 计算损失
        self.tape.backward(self.loss)  # 反向传播：计算所有梯度

    def forward(self):
        """前向传播：运行整个仿真轨迹，计算最终损失。"""
        for sim_step in range(self.sim_steps):
            self.simulate(sim_step)

        # 在最终状态上计算损失
        # loss = ||final_position - target||²
        wp.launch(
            loss_kernel, dim=1,
            inputs=[self.states[-1].particle_q, self.target, self.loss],
        )
        return self.loss

    def simulate(self, sim_step):
        """仿真一个时间步（包含多个子步）。

        【关键区别】这里没有 swap！
        每个子步从 states[t] 读取，写入 states[t+1]
        这样所有中间状态都保留，反向传播时可以逐步回溯
        """
        for i in range(self.sim_substeps):
            t = sim_step * self.sim_substeps + i
            self.states[t].clear_forces()
            # 注意：states[t] → states[t+1]，不做交换
            self.solver.step(self.states[t], self.states[t + 1], self.control, self.contacts, self.sim_dt)

    def step(self):
        """每帧执行：前向+反向+梯度更新。"""
        # 执行前向和反向传播
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.forward_backward()

        # 获取初始速度及其梯度
        x = self.states[0].particle_qd  # 初始速度（要优化的参数）

        if self.verbose:
            print(f"Train iter: {self.train_iter} Loss: {self.loss}")
            print(f"    x: {x} g: {x.grad}")

        # ===================================================================
        # 【梯度下降更新】
        # ===================================================================
        # x.grad 由 tape.backward() 填充
        # 更新规则：x_new = x_old - learning_rate * gradient
        wp.launch(step_kernel, dim=len(x), inputs=[x, x.grad, self.train_rate])

        # 清零梯度，为下一次迭代做准备
        self.tape.zero()

        self.train_iter += 1
        self.loss_history.append(self.loss.numpy()[0])

    def test_final(self):
        """验证解析梯度与数值梯度的一致性。

        这是可微仿真的标准验证方法：
        - 数值梯度：用有限差分法 (f(x+ε) - f(x-ε)) / (2ε) 近似
        - 解析梯度：用 Tape 反向传播计算
        - 两者应该接近（容差 tol=5e-2）
        """
        x_grad_numeric, x_grad_analytic = self.check_grad()
        assert_np_equal(x_grad_numeric, x_grad_analytic, tol=5e-2)
        assert all(np.array(self.loss_history) < 10.0)
        assert all(np.diff(self.loss_history[:-1]) < -1e-3)

    def render(self):
        """渲染轨迹优化过程。每16次迭代渲染一次以节省性能。"""
        if self.frame > 0 and self.train_iter % 16 != 0:
            return

        traj_verts = [self.states[0].particle_q.numpy()[0].tolist()]

        for i in range(self.sim_steps + 1):
            state = self.states[i * self.sim_substeps]
            traj_verts.append(state.particle_q.numpy()[0].tolist())

            self.viewer.begin_frame(self.frame * self.frame_dt)
            self.viewer.log_scalar("/loss", self.loss.numpy()[0])
            self.viewer.log_state(state)
            self.viewer.log_contacts(self.contacts, state)
            # 渲染目标位置（小方块）
            self.viewer.log_shapes(
                "/target",
                newton.GeoType.BOX,
                (0.1, 0.1, 0.1),
                wp.array([wp.transform(self.target, wp.quat_identity())], dtype=wp.transform),
                wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3),
            )
            # 渲染轨迹线（颜色随损失值变化）
            self.viewer.log_lines(
                f"/traj_{self.train_iter - 1}",
                wp.array(traj_verts[0:-1], dtype=wp.vec3),
                wp.array(traj_verts[1:], dtype=wp.vec3),
                bourke_color_map(0.0, 7.0, self.loss.numpy()[0]),
            )
            self.viewer.end_frame()
            self.frame += 1

    def check_grad(self):
        """数值梯度 vs 解析梯度对比。

        数值梯度（有限差分法）：
        ∂loss/∂x_i ≈ (loss(x + ε*e_i) - loss(x - ε*e_i)) / (2ε)

        解析梯度（Tape 反向传播）：
        ∂loss/∂x_i = x.grad[i]（由 tape.backward() 自动计算）
        """
        param = self.states[0].particle_qd
        x_c = param.numpy().flatten()
        x_grad_numeric = np.zeros_like(x_c)

        for i in range(len(x_c)):
            eps = 1.0e-3
            step = np.zeros_like(x_c)
            step[i] = eps

            param.assign(x_c + step)
            l_1 = self.forward().numpy()[0]

            param.assign(x_c - step)
            l_0 = self.forward().numpy()[0]

            x_grad_numeric[i] = (l_1 - l_0) / (eps * 2.0)

        param.assign(x_c)

        tape = wp.Tape()
        with tape:
            l = self.forward()
        tape.backward(l)
        x_grad_analytic = param.grad.numpy()[0].copy()

        print(f"numeric grad: {x_grad_numeric}")
        print(f"analytic grad: {x_grad_analytic}")
        tape.zero()

        return x_grad_numeric, x_grad_analytic


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages.")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args=args, verbose=args.verbose)
    example.check_grad()
    newton.examples.run(example, args)
check_grad