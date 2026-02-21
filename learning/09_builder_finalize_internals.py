"""
Newton 源码解析 09：Builder.finalize() 内部实现
=================================================

【本文件目的】
解析 builder.finalize() 这个最关键的函数做了什么。
它是 CPU 端 Python 数据 → GPU 端 Warp 数组的桥梁。

【源码位置】
    newton/_src/sim/builder.py → ModelBuilder.finalize() (约 1000 行代码)

================================================================================
finalize() 的完整流程
================================================================================

finalize(device="cuda:0", requires_grad=False) 的执行分为 8 个阶段：

┌────────────────────────────────────────────────────────────────┐
│  阶段 1: 验证 (Validation)                                     │
│                                                                │
│  检查数据一致性:                                                │
│  - World 连续性: 同一 world 的实体必须连续添加                   │
│  - Joint 成员: 每个 joint 的 parent/child body 必须有效         │
│  - Shape margin: contact_margin >= thickness                   │
│  - 结构不变量: 数组长度一致、引用有效                            │
│  - (可选) DFS 拓扑排序: 确保 joint 的依赖顺序正确               │
│                                                                │
│  如果验证失败 → 抛出详细的 ValueError 告诉你哪里出了问题         │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  阶段 2: World 结构 (_build_world_starts)                      │
│                                                                │
│  为每种实体构建 per-world 累积起始索引:                          │
│                                                                │
│  例如 3 个 world，每个有 [4, 6, 3] 个 body:                    │
│  body_world_start = [0, 4, 10, 13]                             │
│                                                                │
│  这样可以通过 world_start[w]..world_start[w+1] 获取            │
│  第 w 个 world 的所有 body                                      │
│                                                                │
│  对象: particle, body, shape, joint, articulation,             │
│        equality_constraint, tri, spring, edge, tet             │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  阶段 3: 粒子数据 → GPU                                        │
│                                                                │
│  Python lists → wp.array:                                      │
│  - particle_q:        [vec3] → wp.array(dtype=wp.vec3)        │
│  - particle_qd:       [vec3] → wp.array(dtype=wp.vec3)        │
│  - particle_mass:     [float] → wp.array(dtype=float)         │
│  - particle_inv_mass: 计算 1/mass (mass=0 → inv_mass=0)       │
│  - particle_radius, flags, world: 同上                         │
│                                                                │
│  创建 HashGrid(128,128,128): 用于粒子-粒子近邻查询              │
│  (只在存在 >1 个有碰撞半径的粒子时创建)                          │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  阶段 4: 碰撞几何 → GPU（最复杂的阶段）                         │
│                                                                │
│  4a. Shape 基本属性:                                            │
│      transform, body, type, material(ke,kd,kf,mu,ka,          │
│      restitution), collision_group, flags, thickness,          │
│      contact_margin → wp.array                                 │
│                                                                │
│  4b. 几何源定型:                                                │
│      对每个 Mesh 类型的 shape:                                   │
│        mesh.finalize(device=device)                            │
│        → 将顶点/三角形上传到 GPU，构建 BVH                     │
│      使用 hash 去重: 多个 shape 共享同一 mesh 时只定型一次       │
│                                                                │
│  4c. 计算局部 AABB (per-shape):                                │
│      每种形状类型有不同的 AABB 计算方式:                         │
│      - Sphere: AABB = [-r, -r, -r] to [r, r, r]              │
│      - Box: AABB = [-hx, -hy, -hz] to [hx, hy, hz]           │
│      - Capsule: AABB = [-r, -h-r, -r] to [r, h+r, r]         │
│      - Mesh: 从顶点计算 min/max                                │
│      - Plane: 有限平面用尺寸，无限平面用大球半径                 │
│      + 按 contact_margin + thickness 膨胀                      │
│                                                                │
│      缓存优化: 相同几何参数的 shapes 共享 AABB（如100个          │
│      同样的机器人连杆，只计算一次 AABB）                         │
│                                                                │
│  4d. 计算体素分辨率 (per-shape):                                │
│      用于接触归约(contact reduction):                            │
│      - 目标: ~100 个大致立方体形体素覆盖 shape                  │
│      - voxel_size = (volume / budget)^(1/3)                    │
│      - 接触归约时按体素分桶，每桶保留代表性接触                   │
│                                                                │
│  4e. SDF 生成 (CUDA only):                                     │
│      对需要 SDF 的 mesh shapes:                                 │
│      - 在 mesh 表面周围生成 NanoVDB 有符号距离场                 │
│      - wp.Volume 格式存储                                       │
│      - 用于 mesh-mesh 碰撞检测 (Stage 3c)                      │
│      - 缓存: 相同 mesh 的 SDF 只生成一次                       │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  阶段 5: 可变形元素 → GPU                                       │
│                                                                │
│  - spring_indices:  [int×2]  → wp.array(dtype=int)            │
│  - spring_rest_length, stiffness, damping                      │
│  - tri_indices:     [int×3]  → wp.array(dtype=int)            │
│  - tri_poses (inv_D): [mat22] → wp.array(dtype=wp.mat22)      │
│  - tri_areas: [float]                                          │
│  - edge_indices:    [int×4]  → wp.array(dtype=int)            │
│  - edge_rest_angle: [float]                                    │
│  - tet_indices:     [int×4]  → wp.array(dtype=int)            │
│  - tet_poses (inv_Dm): [mat33] → wp.array(dtype=wp.mat33)     │
│  - tet_volumes: [float]                                        │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  阶段 6: 刚体和关节 → GPU                                      │
│                                                                │
│  6a. 惯性验证/修正:                                             │
│      刚体的惯性张量必须满足三角不等式:                            │
│      I_xx + I_yy >= I_zz (及所有置换)                          │
│      如果违反 → 自动修正到最近的合法惯性张量                     │
│                                                                │
│  6b. 关节属性:                                                  │
│      joint_type, parent, child, parent_xform, child_xform,    │
│      axis, q_start, qd_start, limits, targets, armature,      │
│      act_mode → wp.array                                       │
│                                                                │
│  6c. Articulation 结构:                                        │
│      articulation_start: 每个 articulation 的第一个 joint 索引  │
│      articulation_world: 每个 articulation 所属的 world         │
│                                                                │
│  6d. 祖先映射 (ancestor map):                                   │
│      计算每个 body 到根 body 的路径                              │
│      用于关节坐标 → body 坐标的变换                              │
│                                                                │
│  6e. 等式约束、Mimic 约束 → GPU                                │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  阶段 7: 碰撞对预计算 (find_shape_contact_pairs)               │
│                                                                │
│  为 EXPLICIT 宽相模式预计算所有合法碰撞对:                       │
│                                                                │
│  对每对 shapes (i, j):                                          │
│    ✓ world 兼容 (同 world 或至少一个是全局 world=-1)            │
│    ✓ collision_group 兼容                                      │
│    ✓ 不在排除列表中                                             │
│    ✓ 至少一个 shape 有 COLLIDE 标志                             │
│    → 加入 shape_contact_pairs                                  │
│                                                                │
│  存储为 wp.array(dtype=wp.vec2i): 每个元素是 (shape_a, shape_b)│
│  运行时只需检测这些对的 AABB 是否重叠                            │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  阶段 8: 自定义属性 (Custom Attributes)                        │
│                                                                │
│  处理求解器注册的自定义属性 (如 MuJoCo 的内部参数):             │
│                                                                │
│  对每个 CustomAttribute:                                        │
│    1. 确定频率 (ONCE/JOINT/BODY/SHAPE/PARTICLE/...)            │
│    2. 根据频率确定数组长度                                      │
│    3. 将 Python 值列表 → wp.array                              │
│    4. 附加到 Model/State/Control/Contacts 的对应命名空间        │
│                                                                │
│  命名空间示例:                                                  │
│    model.mujoco.solver_iterations → int                        │
│    state.mpm.particle_elastic_strain → wp.array(mat33)         │
│    control.mujoco.ctrl → wp.array(float)                       │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                      返回 Model 对象
                     (完全 GPU 常驻)

================================================================================
finalize() 之后，Model 的内存布局
================================================================================

Model 对象在 GPU 上的内存布局:

    ┌─────────────────────── GPU VRAM ──────────────────────────┐
    │                                                           │
    │  粒子数据 (连续存储，按 world 分段):                        │
    │  ┌──────────────────────────────────────────────┐         │
    │  │ world 0 particles │ world 1 │ ... │ world N │         │
    │  └──────────────────────────────────────────────┘         │
    │  particle_q[N]:     vec3 × particle_count                 │
    │  particle_qd[N]:    vec3 × particle_count                 │
    │  particle_mass[N]:  float × particle_count                │
    │  particle_inv_mass: float × particle_count                │
    │                                                           │
    │  刚体数据:                                                 │
    │  body_q[N]:         transform × body_count                │
    │  body_qd[N]:        spatial_vector × body_count           │
    │  body_com[N]:       vec3 × body_count                     │
    │  body_mass[N]:      float × body_count                    │
    │  body_inertia[N]:   mat33 × body_count                    │
    │  body_inv_mass[N]:  float × body_count                    │
    │  body_inv_inertia:  mat33 × body_count                    │
    │                                                           │
    │  形状数据:                                                 │
    │  shape_transform:   transform × shape_count               │
    │  shape_body:        int × shape_count                     │
    │  shape_type:        int × shape_count (GeoType enum)      │
    │  shape_scale:       vec3 × shape_count                    │
    │  shape_source_ptr:  uint64 × shape_count (Mesh*/SDF*)     │
    │  shape_material:    float × shape_count × 7               │
    │                                                           │
    │  关节数据:                                                 │
    │  joint_type:        int × joint_count                     │
    │  joint_parent/child: int × joint_count                    │
    │  joint_X_p/X_c:     transform × joint_count              │
    │  joint_axis:        vec3 × joint_axis_count               │
    │  joint_q:           float × joint_coord_count             │
    │  joint_qd:          float × joint_dof_count               │
    │                                                           │
    │  可变形元素:                                                │
    │  spring_indices:    int × spring_count × 2                │
    │  tri_indices:       int × tri_count × 3                   │
    │  edge_indices:      int × edge_count × 4                  │
    │  tet_indices:       int × tet_count × 4                   │
    │  tri_poses:         mat22 × tri_count (inv reference)     │
    │  tet_poses:         mat33 × tet_count (inv reference)     │
    │                                                           │
    │  碰撞数据:                                                 │
    │  shape_contact_pairs: vec2i × pair_count                  │
    │  shape_local_aabb_*: vec3 × shape_count × 2               │
    │  shape_voxel_resolution: vec3 × shape_count               │
    │                                                           │
    │  World 索引:                                               │
    │  *_world_start:     int × (num_worlds + 1) (每种实体)     │
    │  gravity:           vec3 × num_worlds                     │
    └───────────────────────────────────────────────────────────┘

================================================================================
replicate() 的工作原理
================================================================================

replicate(source_builder, count, spacing) 用于多世界复制:

    1. 对 source_builder 的每个实体:
       - body, shape, joint, particle 等
       - 复制 count 次，每次偏移 spacing

    2. World 分配:
       - 每个副本分配不同的 world ID (0, 1, ..., count-1)
       - world=-1 的全局实体（如地面）只复制一次

    3. 索引重映射:
       - 复制的 joint 的 parent/child body 索引需要偏移
       - 复制的 shape 的 body 索引需要偏移
       - spring/tri/edge/tet 的粒子索引需要偏移

    4. 布局:
       - spacing=(5,5,0) → 在 XY 平面上网格排列
       - 自动计算: side_length = ceil(sqrt(count))
"""

# 建议阅读顺序:
# 1. newton/_src/sim/builder.py → 搜索 "def finalize" 开始阅读
# 2. 重点关注: _build_world_starts(), 粒子/形状/关节的转换逻辑
# 3. 理解 shape AABB 缓存和 SDF 生成的优化策略
# 4. 阅读 find_shape_contact_pairs() 理解碰撞对预计算
