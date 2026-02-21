"""
Newton 源码解析 07：求解器内部实现
===================================

【本文件目的】
深入解析 Newton 各求解器的内部物理方程和算法流程。
理解 solver.step() 被调用时，GPU 上的物理计算到底在做什么。

【源码位置】
- 求解器基类:     newton/_src/solvers/solver.py
- XPBD 求解器:    newton/_src/solvers/xpbd/solver_xpbd.py
- XPBD 约束核:    newton/_src/solvers/xpbd/kernels.py
- 半隐式求解器:   newton/_src/solvers/semi_implicit/solver_semi_implicit.py
- 半隐式力核:     newton/_src/solvers/semi_implicit/kernels_body.py
-                  newton/_src/solvers/semi_implicit/kernels_contact.py
-                  newton/_src/solvers/semi_implicit/kernels_particle.py
- MuJoCo 求解器:  newton/_src/solvers/mujoco/solver_mujoco.py

================================================================================
一、共享基础：SolverBase 中的积分核函数
================================================================================

所有求解器共享两个基本积分器（symplectic Euler / 半隐式欧拉）：

【粒子积分 integrate_particles】
    v₁ = v₀ + (F/m + g) × dt      # 先更新速度
    x₁ = x₀ + v₁ × dt              # 再用新速度更新位置

    特殊处理:
    - 不活跃粒子 (ParticleFlags.ACTIVE==0) 跳过
    - 速度限幅: if ||v₁|| > v_max: v₁ *= v_max / ||v₁||
    - 每世界独立重力: g = gravity[particle_world[tid]]
    - 运动学粒子 (inv_mass=0) 不受力但仍移动

【刚体积分 integrate_rigid_body】
    线性部分:
        v₁ = v₀ + (F/m + g) × dt
        x₁ = x_com + v₁ × dt       # 在质心(COM)上积分

    角度部分（在body坐标系中计算，避免惯性张量变换）:
        ω_body = R⁻¹ × ω_world      # 世界角速度→body坐标系
        τ_body = R⁻¹ × τ_world - ω_body × (I × ω_body)  # 科里奥利力修正！
        ω₁_body = ω_body + I⁻¹ × τ_body × dt
        ω₁_world = R × ω₁_body

    四元数积分:
        r₁ = normalize(r₀ + quat(ω₁, 0) × r₀ × 0.5 × dt)

    角阻尼:
        ω₁ *= (1 - angular_damping × dt)

    最终位姿:
        q_new = transform(x₁ - R₁ × com, r₁)   # 从质心回到body原点

    注意: spatial_vector 的约定是 (linear, angular)/(top, bottom)
        spatial_top(qd) = 线速度 v
        spatial_bottom(qd) = 角速度 ω
        spatial_top(f) = 力 F
        spatial_bottom(f) = 力矩 τ

================================================================================
二、SolverXPBD - 扩展位置级动力学 (重点！)
================================================================================

XPBD 的核心思想：
    1. 预测：用外力积分出预测位置
    2. 投影：迭代修正位置以满足约束
    3. 导速：从位置变化导出速度

这是游戏和交互式仿真中最常用的方法，因为它无条件稳定。

【step() 完整流程】

┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: 预测 (Prediction)                                      │
│                                                                  │
│   保存 x₀ = particle_q (原始位置)                                │
│   integrate_particles()  → 得到预测位置 x_pred                   │
│   apply_joint_forces()   → 关节力映射到body空间力                │
│   integrate_bodies()     → 刚体预测位姿                          │
│                                                                  │
│ Phase 2: 约束投影 (Constraint Projection) × N iterations         │
│                                                                  │
│   for iter in range(iterations):                                 │
│     ① solve_particle_shape_contacts()  → 粒子-形状穿透修正       │
│     ② solve_particle_particle_contacts() → 粒子-粒子穿透修正     │
│     ③ solve_springs()                  → 弹簧距离约束             │
│     ④ bending_constraint()             → 弯曲二面角约束           │
│     ⑤ solve_tetrahedra()              → 四面体体积约束            │
│     ⑥ apply_particle_deltas()          → 累积粒子位置修正         │
│     ⑦ solve_body_joints()              → 刚体关节约束             │
│     ⑧ apply_body_deltas()              → 累积刚体位姿修正         │
│     ⑨ solve_body_contact_positions()   → 刚体接触穿透修正         │
│     ⑩ apply_body_deltas()              → 累积刚体接触修正         │
│                                                                  │
│ Phase 3: 恢复系数 (Restitution)                                  │
│                                                                  │
│   apply_rigid_restitution()            → 刚体碰撞反弹速度         │
│   apply_particle_shape_restitution()   → 粒子碰撞反弹速度         │
└─────────────────────────────────────────────────────────────────┘

【XPBD 约束求解的数学核心】

对于一个约束 C(x) = 0，XPBD 的更新公式：

    Δλ = -(C + α̃λ + γ × ∇C·v) / ((1+γ) × ∇C·W·∇Cᵀ + α̃)
    Δx = W × ∇Cᵀ × Δλ

    其中:
    - λ: 拉格朗日乘子（约束力的代理）
    - α̃ = 1/(k × dt²): 柔度参数（k=刚度）
    - γ = kd/(k × dt): 阻尼参数
    - W: 逆质量矩阵（对角）
    - ∇C: 约束梯度

    直觉：α̃ 越小（刚度越大），约束越硬

【弹簧约束 solve_springs】

    约束: C = ||x_i - x_j|| - L_rest
    梯度: ∇C_i = (x_i - x_j) / ||x_i - x_j||,  ∇C_j = -∇C_i
    更新: Δx_i = w_i × ∇C_i × Δλ / (w_i + w_j + α̃)

【弯曲约束 bending_constraint】

    两个共享一条边的三角形，维持它们之间的二面角:

       v_i          v_j          (i, j 是非共享顶点)
        \          /
    n₁  \  face1  / face2  n₂    (n₁, n₂ 是面法线)
         \      /
    ------v_k--v_l------          (k, l 是共享边)

    约束: C = θ - θ_rest
    其中 θ = atan2(sin_θ, cos_θ)
    sin_θ = dot(cross(n₁, n₂), e) / ||e||
    cos_θ = dot(n₁, n₂)

    梯度 d₁, d₂, d₃, d₄ 从面法线解析计算。

【四面体约束 solve_tetrahedra】

    用于软体仿真的体积有限元:

    变形梯度: F = Ds × Dm⁻¹
    其中 Ds = [x₁-x₀, x₂-x₀, x₃-x₀] (当前边矩阵)
         Dm⁻¹ 是参考构型的逆边矩阵（finalize时预计算）

    两个约束项:
    1. 偏差(stretch): C_dev = ||F||_F² - 3 (惩罚偏离单位变形)
       ∇C_dev = 2F × Dm⁻¹ᵀ

    2. 体积: C_vol = det(F) - 1 (保持体积)
       ∇C_vol = 通过列向量叉积计算

    Neo-Hookean 版本 (solve_tetrahedra2):
       shear: C_s = ||F||_F,  compliance = 1/(k_mu × dt² × V_rest)
       volume: C_v = det(F) - α,  α = 1 + k_mu/k_lambda

【关节约束 solve_body_joints】

    每种关节类型对应不同的约束集:

    FIXED: 6个约束 (3平移 + 3旋转) → 完全锁定
    REVOLUTE: 5个约束 (3平移 + 2旋转) → 只允许绕1轴旋转
    PRISMATIC: 5个约束 (2平移 + 3旋转) → 只允许沿1轴平移
    BALL: 3个约束 (3平移) → 允许任意旋转
    D6: 可配置的约束数 → 通用关节

    每个约束的位置修正:
    Δx = -(C × relaxation) / (w_parent + w_child + compliance/dt²)

    关节限位: 如果 q < limit_lower 或 q > limit_upper:
        额外的位置修正将关节坐标推回限位范围

【刚体接触约束 solve_body_contact_positions】

    对每个接触点:
    1. 将body局部接触点变换到世界空间
    2. 计算穿透深度 d = dot(n, p_b - p_a) - thickness
    3. 法线修正: Δx_n = max(-d, 0) × relaxation
    4. 摩擦修正: 库仑摩擦锥 → Δx_t = min(μ × |Δx_n|, |v_t × dt|)
    5. 通过 atomic_add 累积到 body_deltas

================================================================================
三、SolverSemiImplicit - 半隐式欧拉（力级别）
================================================================================

与 XPBD 的位置级方法不同，半隐式求解器在 **力级别** 工作：
计算所有力 → 累积到力缓冲区 → 一次积分

【step() 完整流程】

┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: 力累积 (全部用 atomic_add 累积到 particle_f / body_f)   │
│                                                                  │
│   ① eval_spring_forces()     → 弹簧弹性力 + 阻尼力              │
│   ② eval_triangle_forces()   → 三角面片 FEM 膜力 + 气动力        │
│   ③ eval_bending_forces()    → 二面角弯曲力                      │
│   ④ eval_tetrahedra_forces() → 四面体 Neo-Hookean 弹性力         │
│   ⑤ eval_body_joint_forces() → 关节弹簧-阻尼力                   │
│   ⑥ eval_particle_contact_forces() → 粒子-粒子惩罚接触力         │
│   ⑦ eval_body_contact_forces()     → 刚体惩罚接触力              │
│   ⑧ eval_particle_body_contact_forces() → 粒子-刚体接触力        │
│                                                                  │
│ Phase 2: 积分                                                    │
│                                                                  │
│   integrate_particles()      → 粒子半隐式欧拉                    │
│   integrate_bodies()         → 刚体半隐式欧拉                    │
└─────────────────────────────────────────────────────────────────┘

【弹簧力 eval_spring_forces】

    方向: dir = normalize(x_j - x_i)
    伸长量: stretch = ||x_j - x_i|| - L_rest
    伸长率: stretch_rate = dot(dir, v_j - v_i)
    力: f = dir × (ke × stretch + kd × stretch_rate)
    施加: atomic_sub(f_i, f)  atomic_add(f_j, f)  # 等大反向

【三角面片力 eval_triangle_forces】

    变形梯度: F = Xs × Dm⁻¹ (2×2, 面内变形)
    偏应力(Neo-Hookean): P = F × k_mu + dF/dt × k_damp
    面积保持: C = area/area_rest - α + activation
              力 = k_lambda × C × ∂C/∂x

    气动力:
        v_wind = 面片速度的平均值
        lift = k_lift × area × dot(normal, v_wind)² × normal
        drag = k_drag × area × dot(normal, v_wind) × v_wind

【四面体力 eval_tetrahedra_forces (Neo-Hookean)】

    变形梯度: F = Ds × Dm⁻¹ (3×3, 体积变形)

    偏差应力 (shear/deviatoric):
        Ic = tr(FᵀF)  (右Cauchy-Green张量的迹)
        P_dev = F × k_mu × (1 - 1/(Ic+1)) + dF/dt × k_damp

        注意: (1 - 1/(Ic+1)) 项来自 Smith et al. 2018
        它确保在零变形时应力为零，避免数值不稳定

    体积应力 (hydrostatic):
        J = det(F)  (体积比)
        f_vol = (J - α + activation) × k_lambda × ∂J/∂x

        ∂J/∂x 通过列向量叉积计算:
        ∂J/∂x₁ = cross(x₂-x₀, x₃-x₀)
        ∂J/∂x₂ = cross(x₃-x₀, x₁-x₀)
        ∂J/∂x₃ = cross(x₁-x₀, x₂-x₀)

【关节力 eval_body_joint_forces】

    位置目标力 (PD控制):
        f = ke × (target_pos - q) + kd × (target_vel - qd)

    关节限位力:
        if q < limit_lower:
            f = limit_ke × (limit_lower - q) - limit_kd × qd
        if q > limit_upper:
            f = limit_ke × (limit_upper - q) - limit_kd × qd

    关节附着力 (维持关节几何约束):
        x_err = parent_attachment_point - child_attachment_point
        f_attach = joint_attach_ke × x_err + joint_attach_kd × v_err

    这些力通过力矩臂转换为body上的空间力矩(wrench):
        wrench = spatial_vector(F, r × F + τ)

【接触力 eval_body_contact_forces】

    穿透距离: d = dot(n, p_b - p_a) - thickness
    if d < adhesion_distance:
        法线力: f_n = ke × d
        阻尼力: f_d = kd × min(v_n, 0)
        摩擦力: f_t = μ × |f_n + f_d| × normalize(v_t)
                (Huber范数平滑，避免零速度时梯度奇异)

================================================================================
四、XPBD vs 半隐式 核心区别对照
================================================================================

┌──────────────┬────────────────────┬────────────────────────────┐
│     方面      │      XPBD          │      半隐式欧拉            │
├──────────────┼────────────────────┼────────────────────────────┤
│  求解空间     │  位置修正 Δx       │  力累积 F                  │
│  迭代次数     │  多次 (2-10)       │  1次                       │
│  稳定性       │  无条件稳定        │  dt 需要足够小             │
│  刚度处理     │  柔度参数 α=1/k    │  直接刚度 ke, kd           │
│  接触处理     │  位置投影           │  惩罚力 f=ke×d             │
│  关节处理     │  几何约束投影       │  弹簧-阻尼惩罚             │
│  可微支持     │  支持(慎用)        │  天然支持                   │
│  典型子步     │  10, iter=10       │  32+                       │
│  典型用途     │  交互式仿真        │  可微仿真、优化             │
│  恢复系数     │  后处理速度修正     │  无显式恢复系数             │
│  摩擦模型     │  位置级库仑摩擦     │  Huber范数平滑库仑摩擦      │
└──────────────┴────────────────────┴────────────────────────────┘

================================================================================
五、MuJoCo 求解器简介
================================================================================

MuJoCo 求解器封装了 Google DeepMind 的 mujoco-warp 库。
它使用完全不同的方法：

- 基于关节空间的递推动力学 (Recursive Newton-Euler)
- 内置接触处理 (不需要 Newton CollisionPipeline)
- 支持 CG/Newton 等高级求解方法
- 自动处理正/逆运动学

step() 流程:
1. 将 Newton 的 state/control 映射到 MuJoCo 的 mjData
2. 调用 mujoco_warp.step() 执行物理步进
3. 将结果映射回 Newton 的 state

================================================================================
六、VBD 求解器简介
================================================================================

VBD (Vertex Block Descent) 专为布料和软体优化:

- 使用图着色 (builder.color()) 实现并行 Gauss-Seidel
- 同颜色的顶点可以并行更新，不同颜色顺序处理
- 支持自碰撞检测 (particle_enable_self_contact)
- 需要较高的接触刚度 (ke=1e6)
"""

# 建议阅读顺序：
# 1. newton/_src/solvers/solver.py                    → 积分核函数
# 2. newton/_src/solvers/xpbd/solver_xpbd.py          → XPBD step() 流程
# 3. newton/_src/solvers/xpbd/kernels.py               → XPBD 约束核函数
# 4. newton/_src/solvers/semi_implicit/solver_semi_implicit.py → 半隐式 step()
# 5. newton/_src/solvers/semi_implicit/kernels_body.py → 关节力、接触力
# 6. newton/_src/solvers/semi_implicit/kernels_particle.py → 弹簧力、三角力
