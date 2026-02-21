"""
Newton 源码解析 08：软体、布料、粒子系统内部实现
=================================================

【本文件目的】
深入解析 Newton 的可变形体仿真：布料、软体、粒子（包括MPM）。
理解从场景构建到物理求解的完整链路。

【源码位置】
- 布料构建:     newton/_src/sim/builder.py → add_cloth_grid/add_cloth_mesh
- 软体构建:     newton/_src/sim/builder.py → add_soft_grid/add_soft_mesh
- 粒子构建:     newton/_src/sim/builder.py → add_particle/add_particles
- XPBD约束:     newton/_src/solvers/xpbd/kernels.py
- 半隐式力:     newton/_src/solvers/semi_implicit/kernels_particle.py
- MPM求解器:    newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py
- VBD求解器:    newton/_src/solvers/vbd/solver_vbd.py
- 粒子碰撞:     newton/_src/geometry/kernels.py

================================================================================
一、可变形体的数据表示
================================================================================

Newton 用 **粒子 (particles)** 作为可变形体的基本元素。
不同的拓扑连接定义了不同的可变形体类型：

┌────────────────────────────────────────────────────────────────┐
│                      粒子 (Particles)                          │
│                                                                │
│  particle_q[i] : vec3  → 位置                                 │
│  particle_qd[i]: vec3  → 速度                                 │
│  particle_f[i] : vec3  → 力（每步清零后累积）                   │
│  particle_mass[i]: float → 质量（0=固定/运动学）               │
│  particle_radius[i]: float → 碰撞半径                          │
│  particle_flags[i]: int → 活跃标志                             │
│  particle_world[i]: int → 所属世界                             │
└────────────────────────┬──────────────────────────────────────┘
                         │ 连接关系（存储为索引数组）
            ┌────────────┼─────────────┬───────────────┐
            ▼            ▼             ▼               ▼
     ┌──────────┐  ┌──────────┐  ┌──────────┐   ┌──────────┐
     │  弹簧     │  │  三角面   │  │  边缘     │   │  四面体   │
     │ (Spring)  │  │(Triangle)│  │ (Edge)    │   │(Tetrahed)│
     │           │  │          │  │           │   │          │
     │ 2个粒子   │  │ 3个粒子   │  │ 4个粒子   │   │ 4个粒子   │
     │ 距离约束  │  │ 面内变形  │  │ 弯曲角度  │   │ 体积变形  │
     │ (1D)     │  │ (2D)     │  │ 二面角    │   │ (3D)     │
     └──────────┘  └──────────┘  └──────────┘   └──────────┘
         ↑              ↑             ↑                ↑
       布料+软体       布料          布料             软体

================================================================================
二、布料构建过程 (add_cloth_grid → add_cloth_mesh)
================================================================================

源码位置: builder.py → add_cloth_grid, add_cloth_mesh

add_cloth_grid(dim_x=64, dim_y=32, cell_x=0.1, cell_y=0.1, ...):

    Step 1: 生成顶点网格
        vertices = [(i*cell_x, j*cell_y, 0) for i,j in grid]
        共 (dim_x+1) × (dim_y+1) = 65 × 33 = 2145 个顶点

    Step 2: 三角化
        每个四边形分成2个三角形:
        quad (i,j):
            tri_1: (i*stride+j, i*stride+j+1, (i+1)*stride+j)
            tri_2: ((i+1)*stride+j, i*stride+j+1, (i+1)*stride+j+1)
        共 dim_x × dim_y × 2 = 4096 个三角形

    Step 3: 调用 add_cloth_mesh(vertices, triangles, ...)

add_cloth_mesh(vertices, indices, ...):

    Step 1: 添加粒子（初始质量=0，后面通过面积分配）
        for vertex in vertices:
            add_particle(pos=transform(vertex, pos, rot, scale),
                        vel=..., mass=0.0, radius=particle_radius)

    Step 2: 添加三角面片元素
        for (i, j, k) in triangles:
            # 计算变形梯度的参考矩阵
            edge1 = vertices[j] - vertices[i]
            edge2 = vertices[k] - vertices[i]
            # 2D局部坐标系 (面内)
            D = 2x2 edge matrix in local frame
            area = |det(D)| / 2
            # 存储 inv(D) 作为 rest pose → 运行时计算 F = Xs × inv(D)
            add_triangle(i, j, k, inv(D), area, ke, ka, kd)

    Step 3: 质量分配（质量集中化）
        对每个三角形的3个顶点:
            particle_mass[vertex] += density × triangle_area / 3
        → 质量正比于周围三角形面积

    Step 4: 构建边缘 (弯曲元素)
        使用 MeshAdjacency 找共享边:
            对每条被两个三角形共享的边 (k, l):
                找两个三角形的非共享顶点 i, j
                计算二面角 rest_angle = atan2(sin_θ, cos_θ)
                add_edge(i, j, k, l, rest_angle, ke_bending, kd_bending)

    Step 5 (可选): 添加弹簧
        对每条三角形边添加结构弹簧
        对每条对角线添加剪切弹簧
        rest_length = 当前边长

    Step 6: 固定边界粒子
        if fix_left: 左边一列粒子 mass=0, flags &= ~ACTIVE
        if fix_right: 右边一列粒子 mass=0, flags &= ~ACTIVE
        etc.

================================================================================
三、软体构建过程 (add_soft_grid → add_soft_mesh)
================================================================================

源码位置: builder.py → add_soft_grid, add_soft_mesh

add_soft_grid(dim_x=12, dim_y=4, dim_z=4, cell_x=0.1, ...):

    Step 1: 生成3D顶点格点
        vertices = [(i*cell_x, j*cell_y, k*cell_z) for i,j,k in 3D_grid]
        共 (dim_x+1) × (dim_y+1) × (dim_z+1) 个顶点

    Step 2: 六面体分解为四面体（每个hex cell → 5个tet）
        对每个六面体单元 (i, j, k):
            8个角点索引: c = [000, 100, 110, 010, 001, 101, 111, 011]

            # 交替分解避免偏差
            if (i ^ j ^ k) & 1 == 0:
                5 tets = [(c0,c1,c3,c4), (c2,c1,c3,c6),
                          (c5,c4,c6,c1), (c7,c3,c4,c6),
                          (c4,c1,c6,c3)]
            else:
                5 tets (对称排列)

            每个 tet:
                edge_matrix = [v1-v0, v2-v0, v3-v0]  (3×3)
                volume = |det(edge_matrix)| / 6
                inv_edge_matrix = inv(edge_matrix)   → rest pose

    Step 3: 提取表面三角形
        用字典跟踪每个面（面由3个顶点集合表示）:
            dict[frozenset(v0,v1,v2)] = count
        被1个tet包含的面 → 表面
        被2个tet共享的面 → 内部（删除）
        表面三角形用于碰撞检测和渲染

    Step 4: 质量分配
        particle_mass[vertex] += density × tet_volume / 4

================================================================================
四、物理方程详解：布料的三角面片力
================================================================================

源码: semi_implicit/kernels_particle.py → eval_triangle_forces

这是连续介质力学在2D壳元素上的离散化：

【变形梯度 F (2×2)】

    当前状态边矩阵:
        Xs = [x₁-x₀, x₂-x₀] 在2D局部坐标系中 (2×2)

    变形梯度:
        F = Xs × Dm⁻¹   (Dm⁻¹ 是 rest pose 的逆)

    F 的物理含义:
        F 描述了从参考构型到当前构型的映射
        F = I (单位矩阵) → 没有变形
        det(F) > 1 → 面积增加（拉伸）
        det(F) < 1 → 面积减少（压缩）

【应力计算 (第一类 Piola-Kirchhoff 应力)】

    偏应力（膜力）:
        P = F × k_mu + Ḟ × k_damp
        k_mu: 剪切模量（抵抗面内变形）
        k_damp: 粘性阻尼
        Ḟ: 变形梯度的时间导数

    面积保持:
        C = area_current / area_rest - α + activation
        f_area = k_lambda × C × ∇C
        k_lambda: 体积模量（类比，在2D中是面积模量）
        activation: 肌肉激活（用于驱动布料变形）

【力的分配】

    节点力 = -P × Dm⁻ᵀ × area_rest
    力分配给3个顶点:
        f₁ = 第1列的节点力
        f₂ = 第2列的节点力
        f₀ = -(f₁ + f₂)  (动量守恒)

【气动力（可选）】

    面片速度: v = (v₀ + v₁ + v₂) / 3
    法线: n = normalize(cross(x₁-x₀, x₂-x₀))
    法线速度: v_n = dot(v, n)

    升力: f_lift = k_lift × area × v_n² × n
    阻力: f_drag = k_drag × area × v_n × v

================================================================================
五、物理方程详解：软体的四面体力 (Neo-Hookean)
================================================================================

源码: semi_implicit/kernels_particle.py → eval_tetrahedra_forces

【变形梯度 F (3×3)】

    边矩阵: Ds = [x₁-x₀, x₂-x₀, x₃-x₀]  (3×3)
    变形梯度: F = Ds × Dm⁻¹

    不变量:
        Ic = tr(FᵀF) = ||F||²_F  (右Cauchy-Green张量的迹)
        J = det(F)                 (体积比)

【Neo-Hookean 弹性模型 (Smith et al. 2018 稳定版)】

    偏差应力(shear):
        P_dev = k_mu × F × (1 - 1/(Ic + 1))

        为什么用 (1 - 1/(Ic+1)) 而不是直接 F？
        → 当 Ic → 0 (零变形) 时，(1 - 1/(Ic+1)) → 0
        → 确保零变形时应力为零
        → 标准 Neo-Hookean 在反转(det(F)<0)时不稳定，此修正解决了该问题

    粘性阻尼:
        P_damp = k_damp × Ḟ

    总偏差力:
        f_dev = -(P_dev + P_damp) × Dm⁻ᵀ × volume

【体积保持力】

    约束: J = det(F) → 应等于 1（保持体积）
    力: f_vol = k_lambda × (J - α + activation) × ∂J/∂x

    体积梯度（通过叉积）:
        ∂J/∂x₁ = cross(x₂-x₀, x₃-x₀) × |Dm⁻¹|  (叉积给出面法线×面积)
        ∂J/∂x₂ = cross(x₃-x₀, x₁-x₀) × |Dm⁻¹|
        ∂J/∂x₃ = cross(x₁-x₀, x₂-x₀) × |Dm⁻¹|
        ∂J/∂x₀ = -(∂J/∂x₁ + ∂J/∂x₂ + ∂J/∂x₃)    (力平衡)

    直觉: 体积梯度的方向是"使该顶点运动导致体积变化最大"的方向

================================================================================
六、MPM 求解器内部实现
================================================================================

源码: implicit_mpm/solver_implicit_mpm.py → SolverImplicitMPM

MPM (Material Point Method) 是一种混合拉格朗日-欧拉方法：
粒子携带物质属性（拉格朗日），在背景网格上求解运动方程（欧拉）。

【每步的计算流程】

┌─────────────────────────────────────────────────────────┐
│  1. 网格分配                                              │
│     从粒子位置创建背景网格                                 │
│     Sparse Nanogrid / Dense Grid3D                       │
│                                                          │
│  2. 粒子→网格 (P2G Transfer)                              │
│     动量: p_grid = Σ w_ip × m_p × v_p                   │
│     质量: m_grid = Σ w_ip × m_p                          │
│     速度: v_grid = p_grid / m_grid                       │
│     (w_ip 是粒子p到网格节点i的权重函数)                    │
│                                                          │
│  3. 碰撞体光栅化                                          │
│     将碰撞形状的 SDF/法线/速度/摩擦 光栅化到网格节点       │
│                                                          │
│  4. 弹性系统构建                                          │
│     从弹性变形梯度 F_e 构建刚度矩阵和应变 RHS             │
│                                                          │
│  5. 塑性系统构建                                          │
│     构建屈服面参数（Drucker-Prager / von Mises）          │
│                                                          │
│  6. 耦合求解 (solve_rheology)                             │
│     迭代求解: 速度 + 应力 + 碰撞 (Gauss-Seidel)          │
│                                                          │
│  7. 网格→粒子 (G2P Transfer)                              │
│     v_p = Σ w_ip × v_grid_i                              │
│     x_p += v_p × dt                                      │
│                                                          │
│  8. 应变更新                                              │
│     F_e ← F_e + ∇v × dt × F_e  (弹性应变积分)           │
│     SVD投影: 限制主应变到 [0.01, 4.0]                     │
│                                                          │
│  9. 投影 (project_outside)                                │
│     将穿透碰撞体的粒子推到表面外                           │
└─────────────────────────────────────────────────────────┘

【关键概念】

弹塑性分解:
    F = F_e × F_p  (总变形 = 弹性 × 塑性)
    弹性应变 F_e → 产生应力（可逆）
    塑性应变 F_p → 永久变形（不可逆）

屈服面 (Drucker-Prager):
    f(σ) = ||dev(σ)|| + μ × tr(σ) ≤ 0
    μ: 摩擦角（控制剪切强度随压力的变化）
    当 f > 0: 塑性流动，F_p 更新

APIC vs PIC:
    PIC: 简单的速度传递（有数值耗散）
    APIC: 额外传递速度梯度 ∇v（保持角动量，减少耗散）

【每粒子属性（注册为自定义属性）】

    Model 属性 (mpm namespace):
        young_modulus:      杨氏模量（弹性硬度）
        poisson_ratio:      泊松比（侧向收缩比）
        damping:           阻尼系数
        hardening:          硬化系数
        friction:          摩擦角
        yield_pressure:    屈服压力
        tensile_yield_ratio: 拉伸屈服比
        yield_stress:       屈服应力

    State 属性:
        particle_qd_grad:       速度梯度（APIC用）
        particle_elastic_strain: 弹性变形梯度 F_e (mat33)
        particle_Jp:            塑性体积变化 det(F_p)
        particle_transform:     粒子变换（可选）

================================================================================
七、不同可变形体在不同求解器中的表示对比
================================================================================

┌──────────────┬──────────────┬──────────────┬──────────────┐
│              │ 布料(XPBD)    │ 软体(VBD)     │ 颗粒(MPM)    │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ 拓扑         │ 三角面片网格  │ 四面体网格    │ 无连接粒子    │
│ 拉伸抗力     │ 边弹簧/三角应变│ Neo-Hookean   │ 网格弹性      │
│ 弯曲抗力     │ 边缘二面角    │ 由3D刚度隐含  │ 无            │
│ 体积保持     │ 无(2D)       │ det(F)≈1约束  │ 不可压缩性    │
│ 塑性变形     │ 无           │ 无            │ Drucker-Prager│
│ 碰撞检测     │ 粒子-形状SDF │ 粒子-形状SDF  │ 网格光栅化SDF │
│ 自碰撞       │ Hash Grid    │ Hash Grid    │ 无(网格处理)   │
│ 状态         │ q, qd        │ q, qd        │ q,qd,F_e,Jp  │
│ 求解器       │ XPBD/Style3D │ VBD          │ ImplicitMPM   │
│              │ /SemiImplicit│              │              │
└──────────────┴──────────────┴──────────────┴──────────────┘

================================================================================
八、实际运行的示例
================================================================================

下面提供一个可以实际运行的简单布料示例，带详细注释：
"""

import warp as wp

import newton
import newton.examples


class Example:
    """布料和粒子仿真示例。

    创建一块固定左边缘的布料，在重力下下垂。
    同时在布料下方放置一个球体作为碰撞障碍物。
    """

    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        builder = newton.ModelBuilder()

        # ===================================================================
        # 添加碰撞障碍物：一个固定球体
        # ===================================================================
        # body=-1 表示固定到世界（不参与动力学）
        builder.add_shape_sphere(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            radius=0.5,
        )

        # 添加地面
        builder.add_ground_plane()

        # ===================================================================
        # 构建布料网格
        # ===================================================================
        # add_cloth_grid 内部流程:
        # 1. 生成 (dim_x+1)×(dim_y+1) 个粒子
        # 2. 每个四边形 → 2个三角形
        # 3. 通过三角形面积分配质量
        # 4. 构建弯曲边缘（共享边上的二面角约束）
        # 5. fix_left=True → 左边一列粒子 mass=0（固定）
        builder.add_cloth_grid(
            pos=wp.vec3(-1.0, 0.0, 2.0),   # 布料左下角位置
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi * 0.5),
            vel=wp.vec3(0.0, 0.0, 0.0),    # 初始速度
            dim_x=32,        # X 方向单元数 → 33 个顶点
            dim_y=16,        # Y 方向单元数 → 17 个顶点
            cell_x=0.0625,   # 每个单元 X 方向尺寸
            cell_y=0.0625,   # 每个单元 Y 方向尺寸
            mass=0.1,        # 每个粒子的质量（会被面积加权重新分配）
            fix_left=True,   # 固定左边缘
            # 以下参数定义材质:
            edge_ke=1.0e1,   # 边弹簧刚度（拉伸抗力）
            tri_ke=1.0e3,    # 三角面片膜刚度
            tri_ka=1.0e3,    # 三角面片面积保持刚度
            tri_kd=1.0e1,    # 三角面片阻尼
            particle_radius=0.02,  # 粒子碰撞半径
        )

        self.model = builder.finalize()

        # 设置粒子-形状的软接触参数
        self.model.soft_contact_ke = 1.0e2   # 接触刚度
        self.model.soft_contact_kd = 1.0e0   # 接触阻尼
        self.model.soft_contact_mu = 1.0     # 摩擦系数

        # ===================================================================
        # 使用 XPBD 求解器（适合布料的位置级方法）
        # ===================================================================
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model, args, soft_contact_margin=0.1,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        """每帧仿真循环。

        内部发生的物理过程:
        1. clear_forces(): 清零 particle_f (粒子力) 和 body_f (刚体力)
        2. collide(): 检测粒子与球体/地面的软接触
           → 填充 contacts.soft_contact_* 数组
        3. solver.step() 内部 (XPBD):
           a. integrate_particles(): 预测粒子位置（重力+外力）
           b. 迭代约束投影 x10:
              - solve_particle_shape_contacts(): 修正粒子穿透球体/地面
              - solve_springs(): 修正弹簧长度偏差
              - bending_constraint(): 修正弯曲角度偏差
              - apply_particle_deltas(): 应用位置修正，导出速度
        4. 交换 state_0 / state_1
        """
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        """验证布料粒子在合理位置。"""
        newton.examples.test_particle_state(
            self.state_0,
            "particles are above the ground",
            lambda q, qd: q[2] > -0.05,  # 所有粒子应在地面以上
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
