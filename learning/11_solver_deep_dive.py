"""
Newton 求解器专题：从直觉到源码
=================================

这个文件专门讲解 Newton 的各种物理求解器。
从"它在做什么"到"它怎么做的"到"源码在哪里看"。

阅读完本文件后，你可以带着理解去读源码。

================================================================================
一、求解器在仿真循环中的位置
================================================================================

    solver.step(state_in, state_out, control, contacts, dt)

    这一行代码是整个物理仿真的核心。在它被调用时：

    已经完成的:
      ✓ state_in.body_q/body_qd   → 所有刚体的当前位姿和速度
      ✓ state_in.particle_q/qd    → 所有粒子的当前位置和速度
      ✓ state_in.body_f/particle_f → 外力已经累积（或已清零）
      ✓ contacts                   → 碰撞检测已完成，碰撞点已填入
      ✓ control.joint_f/target_*   → 控制输入已设置

    step() 要做的:
      → 根据力、约束、碰撞，计算下一时刻的状态
      → 写入 state_out.body_q/body_qd, state_out.particle_q/qd

================================================================================
二、所有求解器的共同基础
================================================================================

所有求解器都继承 SolverBase，共享两个积分器：

【粒子积分 (半隐式欧拉)】

    v_new = v + (F/m + gravity) × dt    # 先更新速度
    x_new = x + v_new × dt              # 用新速度更新位置

    为什么叫"半隐式"？
    → 速度更新用旧位置的力（显式）
    → 位置更新用新速度（隐式）
    → 比全显式稳定，比全隐式简单

【刚体积分】

    线性部分: 和粒子一样（在质心上积分）
    角度部分: 需要在body坐标系中计算（避免转动惯量变换）

    ω_body = R⁻¹ × ω           # 角速度变换到body系
    τ_body = R⁻¹ × τ - ω × (I × ω)  # 包含科里奥利力修正！
    ω_new = ω + I⁻¹ × τ_body × dt    # 更新角速度
    r_new = normalize(r + quat(ω_new, 0) × r × 0.5 × dt)  # 更新四元数

    科里奥利力修正 -ω × (I × ω) 的直觉：
    → 陀螺效应！一个旋转的轮子，如果你试图改变旋转轴方向，
      它会产生一个垂直于施加力矩的响应
    → 如果不修正这个项，高速旋转的物体会表现异常

    源码: newton/_src/solvers/solver.py → integrate_rigid_body

================================================================================
三、SolverXPBD — 位置级约束求解（最推荐先学）
================================================================================

【核心思想：预测 → 修正 → 导速】

    第1步：预测
      "假装没有约束（弹簧、关节、碰撞），只受重力和外力"
      → 计算一个"预测位置" x_pred

    第2步：修正（迭代N次）
      "检查预测位置违反了哪些约束，直接把位置改对"
      → 穿透地面了？把物体推出来
      → 弹簧拉太长了？把两端拉近
      → 关节松了？把子体拉回正确位置

    第3步：导速
      "从位置变化反推速度"
      → v_new = (x_final - x_initial) / dt

【为什么XPBD无条件稳定？】

    因为不管时间步多大，约束修正都能直接把位置改对。
    不像力级别方法——力太大 × 步长太大 = 飞出去了。

    类比：
    力方法：在墙上贴弹簧，物体靠近时弹簧推它。
            如果时间步太大，物体一步就飞过弹簧，没被推到。
    XPBD：  物体穿过墙了？直接传送回墙外面。

【约束修正的数学 (XPBD 公式)】

    对一个约束 C(x)=0（比如"弹簧长度=rest_length"）：

    Δλ = -(C + α̃·λ) / (∇C·W·∇Cᵀ + α̃)
    Δx = W · ∇Cᵀ · Δλ

    解读：
    - C: 约束违反量（穿透了多少、弹簧拉长了多少）
    - ∇C: 约束梯度（"往哪个方向修正"）
    - W: 逆质量（质量大→修正小，质量小→修正大）
    - α̃ = 1/(stiffness × dt²): 柔度参数
      → stiffness 越大 → α̃ 越小 → 约束越硬
      → stiffness=∞ → α̃=0 → 完美刚性约束
    - λ: 拉格朗日乘子（累积的约束力）

【iterations 的直觉】

    想象一串弹簧连着的球: A--B--C

    迭代1: 修正AB弹簧，移动了B
           修正BC弹簧，又移动了B
           → AB弹簧又不满足了！

    迭代2: 再修AB，再修BC → 好一点了
    迭代3: 再来一遍 → 更好了
    ...
    迭代10: 几乎完美了

    约束之间互相耦合，一个的修正会破坏另一个。
    所以需要多次迭代让所有约束逐渐趋向同时满足。

【step() 完整流程】

    ┌─ 预测 ─────────────────────────────────────┐
    │ integrate_particles()    # 粒子预测位置      │
    │ apply_joint_forces()     # 关节力→body空间力 │
    │ integrate_bodies()       # 刚体预测位姿      │
    └────────────────────────────────────────────┘
    ┌─ 约束迭代 (× iterations 次) ──────────────┐
    │ solve_particle_shape_contacts()  # 粒子-形状穿透修正   │
    │ solve_particle_particle_contacts()  # 粒子-粒子穿透  │
    │ solve_springs()                  # 弹簧距离约束       │
    │ bending_constraint()             # 弯曲角度约束       │
    │ solve_tetrahedra()               # 四面体体积约束     │
    │ apply_particle_deltas()          # 累积粒子修正       │
    │ solve_body_joints()              # 关节约束           │
    │ apply_body_deltas()              # 累积刚体修正       │
    │ solve_body_contact_positions()   # 刚体碰撞约束       │
    │ apply_body_deltas()              # 累积碰撞修正       │
    └──────────────────────────────────────────────────────┘
    ┌─ 后处理 ──────────────────────────────────┐
    │ apply_rigid_restitution()   # 碰撞弹性恢复速度        │
    │ apply_particle_shape_restitution()  # 粒子弹性恢复    │
    └──────────────────────────────────────────────────────┘

    源码: newton/_src/solvers/xpbd/solver_xpbd.py → step()
    约束核函数: newton/_src/solvers/xpbd/kernels.py

================================================================================
四、SolverSemiImplicit — 力级别求解（可微仿真首选）
================================================================================

【核心思想：算力 → 积分】

    和 XPBD 完全相反的思路:
    XPBD: 先积分，再修正位置
    半隐式: 先算所有力，一次积分

    第1步: 遍历所有物理效应，把力累积到 particle_f / body_f
    第2步: 用累积的总力做一次半隐式欧拉积分

【为什么它适合可微仿真？】

    因为计算图是单向的：
    力 = f(x, v)
    v_new = v + f/m * dt
    x_new = x + v_new * dt

    → 链式法则可以从 x_new 一路反传梯度到 x
    → Warp Tape 天然支持

    XPBD 的迭代修正就麻烦：
    → 同一个变量被读写多次
    → 梯度追踪更复杂

【力的种类（step()中依次计算）】

    ① eval_spring_forces()
       弹簧力: f = ke × stretch + kd × stretch_rate
       stretch = ||x_i - x_j|| - rest_length

    ② eval_triangle_forces()
       三角面片力（连续介质力学 FEM）:
       变形梯度 F = 当前边矩阵 × 参考边矩阵⁻¹
       应力 P = F × k_mu (Neo-Hookean)
       + 面积保持 + 气动力(升力/阻力)

    ③ eval_bending_forces()
       弯曲力: 两个共享边的三角形，维持二面角
       f = ke × (θ - θ_rest) × 梯度方向

    ④ eval_tetrahedra_forces()
       四面体力（体积有限元 Neo-Hookean）:
       F = 变形梯度(3×3)
       偏差应力: P = k_mu × F × (1 - 1/(tr(FᵀF)+1))
       体积应力: f = k_lambda × (det(F) - 1) × ∂det(F)/∂x

    ⑤ eval_body_joint_forces()
       关节力: PD 控制 + 关节限位 + 附着弹簧
       f = ke × (target - q) + kd × (target_vel - qd)

    ⑥ eval_body_contact_forces()
       接触力: 惩罚弹簧
       f_n = ke × penetration + kd × min(v_normal, 0)
       f_friction = μ × |f_n| × normalize(v_tangent)

    ⑦ integrate_particles() + integrate_bodies()
       一次半隐式欧拉积分

    源码: newton/_src/solvers/semi_implicit/solver_semi_implicit.py → step()
    力核函数:
      newton/_src/solvers/semi_implicit/kernels_particle.py  (弹簧/三角/四面体)
      newton/_src/solvers/semi_implicit/kernels_body.py     (关节力)
      newton/_src/solvers/semi_implicit/kernels_contact.py  (接触力)

================================================================================
五、SolverMuJoCo — 关节空间求解
================================================================================

【核心思想：在关节空间而不是笛卡尔空间求解】

    XPBD/SemiImplicit 在笛卡尔空间工作：
      直接操作 body_q (3D位姿) 和 body_f (3D力)

    MuJoCo 在关节空间工作：
      操作 joint_q (关节角度) 和 joint_f (关节力矩)
      更适合关节型机器人（因为自由度少得多）

    一个12关节机器人:
      笛卡尔空间: 13个body × 7个坐标 = 91个变量
      关节空间:   12个关节 × 1个坐标 = 12个变量！

【特点】

    - 内置接触处理（不需要 Newton 的 CollisionPipeline）
    - 使用 CG/Newton 等高级数值方法
    - 自动处理正/逆运动学
    - 封装了 Google DeepMind 的 mujoco-warp 库

【使用要求】

    1. 调用前注册自定义属性:
       newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    2. contacts 可以为 None（MuJoCo 自己处理碰撞）

    源码: newton/_src/solvers/mujoco/solver_mujoco.py

================================================================================
六、SolverVBD — 能量最小化（布料/软体优化）
================================================================================

【核心思想：定义能量函数 → 迭代最小化】

    E(x) = 弹性能 + 接触能 + 重力势能 + ...

    每次迭代:
      对每个顶点 v_i:
        固定其他所有顶点
        找到使 E(x) 最小的 v_i 新位置
        更新 v_i

    这就是"Block Descent"——每次只优化一个"block"(顶点)

【为什么需要图着色？】

    "固定其他所有顶点" → 但GPU上要并行更新多个顶点
    如果 v_i 和 v_j 有弹簧连接，不能同时更新
    → 图着色保证并行更新的顶点之间无连接

    builder.color() 做的事:
    1. 构建连接图
    2. 贪心图着色（通常4-8种颜色）
    3. 求解时: 同色并行，异色串行

【为什么需要高 ke？】

    VBD 通过最小化"接触能量"来解决穿透
    接触能量 = ke × penetration²
    ke 太小 → 能量太小 → 优化器觉得"穿透一点无所谓"
    ke=1e6 → 穿透的能量代价很大 → 优化器会认真避免穿透

    源码: newton/_src/solvers/vbd/solver_vbd.py

================================================================================
七、SolverFeatherstone — 递推关节动力学
================================================================================

【核心思想：递推牛顿-欧拉算法】

    经典的机器人学方法:
    1. 前向传递: 从根到叶，计算每个body的速度和加速度
    2. 后向传递: 从叶到根，计算关节力和约束力
    3. 前向传递: 计算最终加速度

    优点: 对链式结构效率最高 O(n)，n=关节数
    缺点: 树形结构并行度低

    源码: newton/_src/solvers/featherstone/solver_featherstone.py

================================================================================
八、SolverImplicitMPM — 物质点法
================================================================================

【核心思想：粒子+网格混合方法】

    粒子(拉格朗日): 携带物质属性，跟着物质移动
    网格(欧拉): 用来求解运动方程

    每步:
    1. 粒子→网格: 把粒子的质量和动量转移到背景网格
    2. 在网格上求解: 重力、弹性、碰撞
    3. 网格→粒子: 把网格速度传回粒子，更新粒子位置

    适合: 沙子、泥土、雪、颗粒流、大变形软体

    源码: newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py

================================================================================
九、求解器选择速查表
================================================================================

    ┌──────────────┬───────────────────────────────────────────────┐
    │ 场景         │ 推荐求解器                                     │
    ├──────────────┼───────────────────────────────────────────────┤
    │ 刚体掉落碰撞 │ SolverXPBD (默认, 通用)                       │
    │ 关节机器人   │ SolverMuJoCo (精确关节, 自带碰撞)             │
    │ 布料仿真     │ SolverXPBD / SolverVBD / SolverStyle3D        │
    │ 软体/FEM     │ SolverVBD (需要color())                       │
    │ 可微仿真     │ SolverSemiImplicit (支持Tape反传)             │
    │ 沙/雪/颗粒   │ SolverImplicitMPM                             │
    │ 关节动力学   │ SolverFeatherstone (递推算法)                  │
    │ 多种混合     │ 考虑场景复杂度选择                              │
    └──────────────┴───────────────────────────────────────────────┘

================================================================================
十、源码阅读路线图
================================================================================

推荐按以下顺序阅读源码：

    Level 1: 理解接口
      newton/_src/solvers/solver.py          # 基类，integrate_*核函数

    Level 2: 理解最常用的求解器
      newton/_src/solvers/xpbd/solver_xpbd.py   # step()的完整流程
      newton/_src/solvers/xpbd/kernels.py        # 每个约束的具体实现

    Level 3: 理解可微仿真求解器
      newton/_src/solvers/semi_implicit/solver_semi_implicit.py
      newton/_src/solvers/semi_implicit/kernels_particle.py  # 弹簧/三角/tet力
      newton/_src/solvers/semi_implicit/kernels_body.py      # 关节力
      newton/_src/solvers/semi_implicit/kernels_contact.py   # 接触力

    Level 4: 特化求解器
      newton/_src/solvers/mujoco/solver_mujoco.py
      newton/_src/solvers/vbd/solver_vbd.py
      newton/_src/solvers/featherstone/solver_featherstone.py
      newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py

    每个求解器的关键函数:
      step()                  → 主入口，完整流程
      register_custom_attributes() → 注册求解器特有属性（如MuJoCo的参数）
      notify_model_changed()  → 模型修改时刷新内部缓存
      update_contacts()       → 将内部接触数据转为统一Contacts格式
"""
