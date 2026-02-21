"""
Newton 源码解析 18：Model 全流程 —— 从场景构建到物理仿真
=========================================================

【本文件目的】
这是 Newton 中最重要的一份文档。
Newton 的一切都从 ModelBuilder → Model 开始。
本文件详细解析这个流程的每一步，并标注所有源码位置。

【源码文件地图】

  newton/_src/sim/
  ├── builder.py   ← ModelBuilder 类（9121 行，最大的文件！）
  │                   CPU 端场景构建：add_body/add_joint/add_shape/finalize
  │
  ├── model.py     ← Model 类（1104 行）
  │                   GPU 端静态模型：所有 wp.array 数据存储
  │
  ├── state.py     ← State 类（~150 行）
  │                   GPU 端动态状态：body_q/body_qd/body_f/particle_q 等
  │
  ├── control.py   ← Control 类（93 行）
  │                   控制输入：joint_f/joint_target_pos/joint_target_vel
  │
  ├── contacts.py  ← Contacts 类（~170 行）
  │                   碰撞结果：rigid_contact_*/soft_contact_*
  │
  └── collide.py   ← CollisionPipeline 类（~850 行）
                      碰撞检测管线：宽相+窄相+软接触


================================================================================
一、全景图：5 个核心对象的生命周期
================================================================================

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│    ModelBuilder (CPU)                                                   │
│    ──────────────                                                       │
│    Python 列表存数据，像画蓝图                                           │
│                                                                         │
│    builder = newton.ModelBuilder()       # 创建空白蓝图                  │
│    builder.add_body(...)                 # 在蓝图上画刚体                │
│    builder.add_joint_revolute(...)       # 画关节                        │
│    builder.add_shape_box(...)            # 画碰撞形状                    │
│    builder.add_cloth_grid(...)           # 画布料                        │
│                                                                         │
│         │ finalize(device="cuda:0")                                     │
│         │ Python 列表 → wp.array（GPU 数组）                             │
│         ▼                                                               │
│                                                                         │
│    Model (GPU，静态，不变)                                               │
│    ──────────────────────                                                │
│    包含场景的所有常量数据（质量、惯性、关节类型、形状几何...）              │
│                                                                         │
│         │                                                               │
│         ├── model.state()     → State (GPU，动态，每步变化)              │
│         │                       位姿 body_q、速度 body_qd、力 body_f     │
│         │                                                               │
│         ├── model.control()   → Control (GPU，用户输入)                  │
│         │                       关节力 joint_f、目标位置/速度             │
│         │                                                               │
│         └── CollisionPipeline(model).contacts() → Contacts (GPU，碰撞结果)│
│                                                    接触点、法线、穿透深度 │
│                                                                         │
│    仿真循环：                                                            │
│    for step in range(N):                                                │
│        state.clear_forces()                                             │
│        collision_pipeline.collide(state, contacts)                      │
│        solver.step(state_in, state_out, control, contacts, dt)          │
│        state_in, state_out = state_out, state_in                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘


================================================================================
二、ModelBuilder：在 CPU 端构建场景蓝图
================================================================================

源码位置: newton/_src/sim/builder.py（9121 行）
公共 API:  newton.ModelBuilder（通过 newton/__init__.py 导出）

--------------------------------------------------------------------------------
2.1 创建 Builder
--------------------------------------------------------------------------------

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)

    源码: builder.py 第 549-749 行 __init__()

    内部初始化了大量空 Python 列表，等待 add_* 方法填充：
      self.body_mass = []          # 每个刚体的质量
      self.body_inertia = []       # 每个刚体的惯性张量
      self.body_com = []           # 每个刚体的质心
      self.body_q = []             # 每个刚体的初始位姿
      self.joint_type = []         # 每个关节的类型
      self.shape_geo_type = []     # 每个形状的几何类型
      self.particle_q = []         # 每个粒子的位置
      ...（几十个列表）

    此时所有数据都在 CPU 上的 Python 列表里，方便用 Python 语法灵活构建。

--------------------------------------------------------------------------------
2.2 添加刚体 (Body)
--------------------------------------------------------------------------------

    body_id = builder.add_body(
        xform=wp.transform(...),   # 初始位姿
        mass=1.0,                   # 质量（0=固定不动，即运动学体）
    )

    源码: builder.py 第 2645 行

    Body 是物理实体的载体。它本身没有形状——形状需要通过 add_shape_* 附加上去。
    一个 body 可以附加多个 shape（比如机器人的一条腿由胶囊+球组合）。

    body_id = -1 是特殊值，表示"世界"（固定参考系，不会移动）。
    地面、固定障碍物的 body 都是 -1。

    add_link() 是 add_body() 的包装，用于创建关节链中的连杆，
    会自动计算惯性（基于后续添加的 shape 的几何和密度）。
    源码: builder.py 第 2556 行

--------------------------------------------------------------------------------
2.3 添加关节 (Joint)
--------------------------------------------------------------------------------

    joint_id = builder.add_joint_revolute(
        parent=-1,              # 父体（-1=世界）
        child=body_0,           # 子体
        parent_xform=wp.transform(p=wp.vec3(0,0,5), q=rot),  # 关节在父体上的安装位置
        child_xform=wp.transform(p=wp.vec3(-1,0,0), q=q_id), # 关节在子体上的安装位置
        axis=wp.vec3(0,1,0),    # 旋转轴方向
    )

    关节类型及源码位置:
      add_joint_revolute()    第 2883 行   旋转关节（1 自由度，像门铰链）
      add_joint_prismatic()   第 2976 行   滑动关节（1 自由度，像抽屉导轨）
      add_joint_ball()        第 3067 行   球关节  （3 自由度，像肩关节）
      add_joint_fixed()       第 3140 行   固定关节（0 自由度，焊死）
      add_joint_free()        第 3186 行   自由关节（6 自由度，完全自由浮动）

    关节的核心概念——parent_xform 和 child_xform:
      物理引擎保证 parent_xform 的点和 child_xform 的点在世界中始终重合。
      parent_xform = 关节安装在父体的哪个位置
      child_xform  = 关节安装在子体的哪个位置
      详细图解见 learning/01_basic_pendulum_annotated.py 第 91-148 行

    add_articulation() 把多个关节组成一个关节链（articulation）：
    源码: builder.py 第 1317 行
      builder.add_articulation([j0, j1, j2], key="robot_arm")

--------------------------------------------------------------------------------
2.4 添加碰撞形状 (Shape)
--------------------------------------------------------------------------------

    shape_id = builder.add_shape_box(
        body=body_0,                    # 附加到哪个 body（-1=固定到世界）
        xform=wp.transform(...),        # 形状在 body 局部坐标系中的位置
        hx=0.5, hy=0.5, hz=0.5,        # 半尺寸（实际尺寸是 2*hx × 2*hy × 2*hz）
        cfg=newton.ModelBuilder.ShapeConfig(density=1000.0),  # 材质配置
    )

    形状类型及源码位置:
      add_shape_sphere()     第 4493 行   球体
      add_shape_box()        第 4605 行   长方体
      add_shape_capsule()    第 4646 行   胶囊体（圆柱+两端半球）
      add_shape_cylinder()   ~第 4700 行  圆柱体
      add_shape_cone()       ~第 4750 行  圆锥体
      add_shape_mesh()       第 4797 行   三角网格（任意形状）
      add_ground_plane()     第 4469 行   无限大地面

    Shape 和 Body 的关系:
      Body 是物理实体（有质量、惯性、位姿）
      Shape 是碰撞几何（附加在 body 上，用于碰撞检测）
      一个 body 可以有 0 个或多个 shape

    ShapeConfig 材质参数:
      density:      密度（kg/m³），自动算质量和惯性
      ke:           接触弹性刚度
      kd:           接触阻尼
      kf:           接触摩擦刚度
      mu:           摩擦系数
      restitution:  弹性碰撞系数（0=完全非弹性，1=完全弹性）

--------------------------------------------------------------------------------
2.5 添加粒子和布料
--------------------------------------------------------------------------------

    # 单个粒子
    builder.add_particle(pos=wp.vec3(0,0,5), vel=wp.vec3(0,0,0), mass=0.1)
    源码: builder.py 第 5775 行

    # 布料网格（批量生成粒子+三角形+弹簧）
    builder.add_cloth_grid(
        pos=wp.vec3(0,0,4), rot=...,
        dim_x=32, dim_y=16, cell_x=0.1, cell_y=0.1,
        mass=0.1, fix_left=True,
    )
    源码: builder.py 第 6346 行
    内部流程: 生成网格顶点 → 两两连成三角形 → 创建弹簧/边缘 → 设置质量

    # 软体网格（批量生成粒子+四面体）
    builder.add_soft_grid(
        pos=wp.vec3(0,0,2), rot=...,
        dim_x=4, dim_y=4, dim_z=4, cell_x=0.1, cell_y=0.1, cell_z=0.1,
        density=1000.0, k_mu=1e4, k_lambda=1e5,
    )
    源码: builder.py 第 6701 行

--------------------------------------------------------------------------------
2.6 多世界复制 (replicate)
--------------------------------------------------------------------------------

    # 先构建一个世界
    single = newton.ModelBuilder()
    # ... add_body, add_joint, add_shape ...

    # 复制 100 份
    scene = newton.ModelBuilder()
    scene.replicate(single, num_worlds=100)

    源码: builder.py 第 1286 行

    所有 100 个世界共享同一个 GPU 模型，物理上互不影响。
    强化学习训练的标准做法——同时跑 100 个环境。


================================================================================
三、finalize()：从蓝图到 GPU 模型（最关键的一步）
================================================================================

    model = builder.finalize(device="cuda:0", requires_grad=False)

    源码: builder.py 第 8148-9017 行（约 870 行！）

    这是 Newton 最核心的函数。它把 CPU 端的 Python 列表全部转成 GPU 端的 wp.array，
    同时做大量预计算和验证。

    完整流程（22 步）：

    ┌─ 验证阶段 ──────────────────────────────────────────────────────────┐
    │                                                                     │
    │  第 1 步: 验证世界结构（行 8189-8210）                                │
    │    - num_worlds 设置                                                │
    │    - 验证 world 顺序连续                                             │
    │    - 验证关节拓扑（父体必须在子体之前）                               │
    │    - 验证形状合法性                                                  │
    │                                                                     │
    │  第 2 步: 构建 world starts（行 8215）                               │
    │    - 计算每个 world 的起始索引（body/joint/shape/particle）           │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    ┌─ 创建 Model + 转换数据 ────────────────────────────────────────────┐
    │                                                                     │
    │  第 3 步: 创建空 Model 对象（行 8222-8229）                          │
    │                                                                     │
    │  第 4 步: 粒子数据 → GPU（行 8234-8257）                             │
    │    Python list → wp.array:                                          │
    │    particle_q, particle_qd, particle_mass, particle_inv_mass,       │
    │    particle_radius, particle_flags, particle_world                   │
    │                                                                     │
    │  第 5 步: 形状数据 → GPU（行 8260-8318）                             │
    │    shape_transform, shape_body, shape_type, shape_scale,            │
    │    材质参数（ke, kd, kf, mu, restitution...）                        │
    │    Mesh/SDF 几何数据的 finalize                                      │
    │                                                                     │
    │  第 6 步: 计算形状 AABB 和体素分辨率（行 8320-8469）                  │
    │    每个 shape 的局部空间 AABB（用于碰撞检测加速）                     │
    │                                                                     │
    │  第 7 步: 生成 SDF（行 8472-8623）                                   │
    │    为 mesh 形状生成有符号距离场（用于碰撞和 Hydroelastic）             │
    │                                                                     │
    │  第 8 步: 弹簧数据 → GPU（行 8626-8637）                             │
    │    spring_indices, spring_rest_length, spring_stiffness, spring_kd   │
    │                                                                     │
    │  第 9 步: 三角形数据 → GPU（行 8640-8646）                           │
    │    tri_indices, tri_poses, tri_materials, tri_areas                  │
    │                                                                     │
    │ 第 10 步: 边缘数据 → GPU（行 8649-8656）                             │
    │    edge_indices, edge_rest_angle, edge_bending_properties            │
    │                                                                     │
    │ 第 11 步: 四面体数据 → GPU（行 8659-8664）                           │
    │    tet_indices, tet_poses, tet_materials                             │
    │                                                                     │
    │ 第 12 步: 肌肉数据 → GPU（行 8667-8677）                             │
    │                                                                     │
    │ 第 13 步: 刚体数据 → GPU（行 8680-8776）                             │
    │    验证和修正惯性张量（确保物理合理）                                  │
    │    body_mass, body_inv_mass, body_inertia, body_inv_inertia,         │
    │    body_q, body_qd, body_com, body_world                             │
    │                                                                     │
    │ 第 14 步: 关节数据 → GPU（行 8778-8846）                             │
    │    joint_type, joint_parent, joint_child, joint_X_p, joint_X_c,     │
    │    joint_axis, joint_q, joint_qd, articulation_start                 │
    │    计算 joint_ancestor（每个关节的根关节）                             │
    │    动力学参数（armature, target_ke, target_kd, limits...）            │
    │                                                                     │
    │ 第 15-16 步: 等式约束 + Mimic 约束 → GPU（行 8849-8872）             │
    │                                                                     │
    │ 第 17 步: World 起始索引 → GPU（行 8875-8884）                       │
    │                                                                     │
    │ 第 18 步: 设置计数属性（行 8887-8902）                               │
    │    model.body_count, model.joint_count, model.shape_count,           │
    │    model.particle_count, model.tri_count, model.tet_count...         │
    │                                                                     │
    │ 第 19 步: 计算碰撞对（行 8904）                                      │
    │    find_shape_contact_pairs(model) 自动找出哪些 shape 对可能碰撞     │
    │                                                                     │
    │ 第 20 步: 重力和坐标轴（行 8906-8922）                               │
    │    设置 per-world 的 gravity 数组                                    │
    │                                                                     │
    │ 第 21 步: 自定义属性 → GPU（行 8924-9015）                           │
    │                                                                     │
    │ 第 22 步: return model（行 9017）                                    │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

    finalize() 之后，builder 的使命完成，后续只用 model。


================================================================================
四、Model：GPU 端的静态数据仓库
================================================================================

源码位置: newton/_src/sim/model.py（1104 行）

Model 包含场景的所有不随时间变化的数据。
所有数据都是 wp.array（GPU 数组），由 finalize() 创建。

--------------------------------------------------------------------------------
4.1 刚体相关属性
--------------------------------------------------------------------------------

    model.body_count          int           刚体总数
    model.body_q              transform[]   初始位姿（位置+旋转）    第 344 行
    model.body_qd             spatial_vec[] 初始速度（线速度+角速度） 第 346 行
    model.body_com            vec3[]        质心（局部坐标）         第 348 行
    model.body_mass           float[]       质量                    第 354 行
    model.body_inv_mass       float[]       质量倒数（0=固定体）     第 356 行
    model.body_inertia        mat33[]       惯性张量                第 350 行
    model.body_inv_inertia    mat33[]       惯性张量的逆             第 352 行
    model.body_world          int[]         所属世界编号             第 360 行

--------------------------------------------------------------------------------
4.2 关节相关属性
--------------------------------------------------------------------------------

    model.joint_count         int           关节总数
    model.joint_type          int[]         关节类型                第 386 行
    model.joint_parent        int[]         父体 ID                 第 390 行
    model.joint_child         int[]         子体 ID                 第 392 行
    model.joint_X_p           transform[]   关节在父体上的安装位置   第 396 行
    model.joint_X_c           transform[]   关节在子体上的安装位置   第 398 行
    model.joint_axis          vec3[]        关节轴方向              第 400 行
    model.joint_q             float[]       关节广义坐标             第 376 行
    model.joint_qd            float[]       关节广义速度             第 378 行

    关节索引系统（理解这个很重要）：
      joint_q_start[i]  = 关节 i 的广义坐标在 joint_q 数组中的起始位置
      joint_qd_start[i] = 关节 i 的自由度在 joint_qd 数组中的起始位置

      例: 3 个关节 [旋转(1dof), 球(3dof), 自由(7coord/6dof)]
        joint_q_start  = [0, 1, 4, 11]   ← 最后一个是 sentinel
        joint_qd_start = [0, 1, 4, 10]

--------------------------------------------------------------------------------
4.3 形状相关属性
--------------------------------------------------------------------------------

    model.shape_count         int           形状总数
    model.shape_transform     transform[]   形状在 body 局部坐标中的变换  第 194 行
    model.shape_body          int[]         形状附加到哪个 body           第 196 行
    model.shape_type          int[]         几何类型（球/盒/胶囊/...）    第 226 行
    model.shape_scale         vec3[]        缩放                         ~第 230 行
    model.shape_world         int[]         所属世界编号                  第 252 行

--------------------------------------------------------------------------------
4.4 粒子相关属性
--------------------------------------------------------------------------------

    model.particle_count      int           粒子总数
    model.particle_q          vec3[]        初始位置                     第 146 行
    model.particle_qd         vec3[]        初始速度                     第 148 行
    model.particle_mass       float[]       质量                        第 150 行
    model.particle_inv_mass   float[]       质量倒数（0=固定粒子）       第 152 行
    model.particle_radius     float[]       碰撞半径                     第 154 行
    model.particle_world      int[]         所属世界编号                  第 176 行

--------------------------------------------------------------------------------
4.5 从 Model 创建可变对象
--------------------------------------------------------------------------------

    state   = model.state()       # 创建一个 State（第 715 行）
    control = model.control()     # 创建一个 Control（第 763 行）

    这两个方法会读取 model 中的初始值（body_q, joint_q 等），
    复制到新创建的 State/Control 对象中。每次调用都创建一个全新的副本。

    所以可以：
      state_0 = model.state()   # 当前帧状态
      state_1 = model.state()   # 下一帧状态（用于双缓冲）


================================================================================
五、State：GPU 端的动态状态（每步都变）
================================================================================

源码位置: newton/_src/sim/state.py

State 包含随仿真时间变化的数据。solver.step() 的输入和输出都是 State。

    state.body_q        transform[]      刚体位姿           第 83 行
    state.body_qd       spatial_vector[] 刚体速度           第 86 行
    state.body_f        spatial_vector[] 刚体上的合力        第 101 行

    state.particle_q    vec3[]           粒子位置           第 74 行
    state.particle_qd   vec3[]           粒子速度           第 77 行
    state.particle_f    vec3[]           粒子上的合力        第 80 行

    state.joint_q       float[]          关节广义坐标        第 120 行
    state.joint_qd      float[]          关节广义速度        第 123 行

    state.clear_forces()   第 126 行   清零 body_f 和 particle_f（每步开头调）


================================================================================
六、仿真循环的完整数据流（结合源码位置）
================================================================================

    # 初始化
    builder = newton.ModelBuilder()                    # builder.py 第 549 行
    # ... add_body/add_joint/add_shape ...
    model = builder.finalize()                         # builder.py 第 8148 行

    solver = newton.solvers.SolverXPBD(model)          # 选择求解器
    state_0 = model.state()                            # model.py 第 715 行
    state_1 = model.state()                            # 双缓冲的第二个 state
    control = model.control()                          # model.py 第 763 行

    pipeline = newton.CollisionPipeline(model)         # collide.py 第 391 行
    contacts = pipeline.contacts()                     # contacts.py 第 23 行

    # 每帧
    for frame in range(num_frames):
        for substep in range(sim_substeps):

            # 1. 清零力
            state_0.clear_forces()                     # state.py 第 126 行
            #    body_f.zero_()   ← 刚体合力清零
            #    particle_f.zero_() ← 粒子合力清零

            # 2. 碰撞检测
            pipeline.collide(state_0, contacts)        # collide.py 第 620 行
            #    a. contacts.clear()              ← 清零碰撞计数器
            #    b. compute_shape_aabbs()         ← 计算每个 shape 的世界 AABB
            #    c. broad_phase.launch()          ← 宽相：AABB 重叠测试
            #    d. narrow_phase.launch()         ← 窄相：精确碰撞（GJK/MPR）
            #    e. create_soft_contacts()        ← 粒子-形状 SDF 碰撞

            # 3. 物理积分
            solver.step(state_0, state_1, control, contacts, dt)
            #    输入: state_0（当前状态）+ control（控制输入）+ contacts（碰撞信息）
            #    输出: state_1（下一时刻状态）
            #    内部:
            #      a. 计算所有力（弹簧/三角/碰撞/关节/重力...）
            #      b. 积分更新位置和速度
            #      c. (XPBD) 迭代投影约束

            # 4. 交换缓冲区
            state_0, state_1 = state_1, state_0
            #    下一步的 state_0 就是这一步算出的 state_1


================================================================================
七、源码阅读建议
================================================================================

【推荐阅读顺序】

  1. state.py（~150 行）
     最短最简单，先理解 State 有哪些属性。

  2. control.py（93 行）
     同样很短，理解 Control 的属性。

  3. model.py（1104 行）
     看 Model 类的属性声明（前 500 行），理解数据结构。
     然后看 state() 和 control() 方法（第 715、763 行），理解如何创建 State/Control。

  4. builder.py（9121 行，选择性阅读）
     - 先看 __init__（第 549 行）理解初始化了什么
     - 然后看 add_body（第 2645 行）理解添加刚体的过程
     - 然后看 add_joint_revolute（第 2883 行）理解关节
     - 最后看 finalize（第 8148 行）理解 CPU→GPU 的转换
       finalize 很长但是结构很规律：
       "验证 → 创建 Model → 逐类数据转换 → 设置计数 → 返回"

  5. collide.py（~850 行）
     理解碰撞检测管线的初始化和 collide() 方法。

  6. contacts.py（~170 行）
     理解碰撞结果的数据结构。

【阅读技巧】

  - builder.py 的 finalize() 虽然有 870 行，但结构非常规律：
    每种数据类型（粒子、形状、刚体、关节...）都是同样的模式：
      1. 从 Python list 转成 numpy array
      2. 从 numpy array 转成 wp.array（GPU）
      3. 赋值给 model 的属性

  - 在 Cursor 中用 Ctrl+G 跳到行号，直接到目标位置

  - model.py 中每个属性都有 docstring，悬停就能看到类型和说明
"""
