"""
Newton 源码解析 06：碰撞检测管线内部实现
==========================================

【本文件目的】
这不是一个可运行的示例，而是对 Newton 碰撞检测管线源码的逐层解析。
通过阅读本文件，你将理解 collide() 调用时，GPU 上到底发生了什么。

【源码位置】
- 碰撞管线:   newton/_src/sim/collide.py          → CollisionPipeline 类
- 宽相检测:   newton/_src/geometry/broad_phase_nxn.py  → BroadPhaseAllPairs
- 宽相SAP:    newton/_src/geometry/broad_phase_sap.py  → BroadPhaseSAP
- 窄相检测:   newton/_src/geometry/narrow_phase.py     → NarrowPhase
- 基元碰撞:   newton/_src/geometry/collision_primitive.py → 各种 collide_* 函数
- 软接触:     newton/_src/geometry/kernels.py          → create_soft_contacts
- 接触数据:   newton/_src/sim/contacts.py             → Contacts 类

【碰撞检测管线全景图】

collision_pipeline.collide(state, contacts) 的完整执行流程：

┌──────────────────────────────────────────────────────────────────┐
│                    collide() 主函数                              │
├──────────────────────────────────────────────────────────────────┤
│  1. contacts.clear()         → 清零碰撞计数器                    │
│  2. compute_shape_aabbs()    → 为每个 shape 计算世界空间 AABB    │
│  3. broad_phase.launch()     → 宽相检测：AABB 重叠测试           │
│  4. prepare_geom_data()      → 准备几何数据（变换、缩放、厚度）    │
│  5. narrow_phase.launch()    → 窄相检测：精确碰撞计算             │
│  6. create_soft_contacts()   → 粒子-形状软接触检测               │
└──────────────────────────────────────────────────────────────────┘

================================================================================
步骤1: AABB 计算
================================================================================

每个 shape 在世界空间中计算其轴对齐包围盒 (AABB)。
AABB 是碰撞检测的第一层过滤——如果两个 AABB 不重叠，则不可能碰撞。

源码位置: collide.py → compute_shape_aabbs 核函数

伪代码：
    对每个 shape_id (并行执行):
        # 1. 计算世界空间变换
        if shape 属于某个 body:
            X_ws = body_q[body_id] * shape_transform[shape_id]  # 刚体位姿 × 形状局部变换
        else:
            X_ws = shape_transform[shape_id]  # 固定形状（如地面）

        # 2. 根据形状类型计算 AABB
        if 是平面/网格/SDF:
            # 使用保守的包围球
            radius = shape_collision_radius[shape_id]
            aabb = (pos - radius, pos + radius)
        else:
            # 使用 support function 计算紧凑 AABB
            # support(dir) 返回形状沿 dir 方向的最远点
            # 分别沿 ±x, ±y, ±z 方向查询 → 得到精确 AABB
            aabb = compute_tight_aabb_from_support(shape_data, orientation, pos)

        # 3. 按 contact_margin 膨胀 AABB（确保近距离物体也被检测到）
        aabb_lower[shape_id] = aabb.min - margin
        aabb_upper[shape_id] = aabb.max + margin

Support Function（支撑函数）是凸几何的核心概念：
    support(shape, direction) = argmax_{point in shape} dot(point, direction)
    即沿给定方向最远的点。球、盒、胶囊、圆柱等都有解析 support function。

================================================================================
步骤2: 宽相检测 (Broad Phase)
================================================================================

宽相的目标：快速找出"可能碰撞"的 shape 对，过滤掉不可能碰撞的对。

--- 模式A: NXN (全对检测) ---
源码位置: broad_phase_nxn.py → BroadPhaseAllPairs

原理：
    - 预处理阶段，按 world 分组所有 shapes
    - 每个 world 内有 n 个 shapes，产生 n*(n-1)/2 个候选对
    - 启动 sum(n_w*(n_w-1)/2) 个 GPU 线程，每个线程检测一对

核函数逻辑：
    thread_id → (world, local_pair_id) → (shape_a, shape_b)
    转换过程：
        1. 二分搜索确定属于哪个 world
        2. 下三角矩阵索引转换为 (row, col)
        3. 通过 world_index_map 映射到实际 shape ID

    过滤条件：
        - collision_group 检查（同组或通配组才碰撞）
        - world 兼容性检查
        - 显式排除对检查（二分搜索 shape_pairs_excluded）
        - AABB 重叠测试

    collision_group 语义：
        0  = 不与任何组碰撞
        >0 = 只与同组或负组碰撞
        <0 = 与所有组碰撞（除了对应正组）

--- 模式B: SAP (Sweep and Prune) ---
源码位置: broad_phase_sap.py → BroadPhaseSAP

原理：
    - 将所有 AABB 沿某个轴排序
    - 扫描排序后的列表，只测试相邻元素
    - 时间复杂度 O(N log N)，适合大规模场景

--- 模式C: EXPLICIT (预计算) ---
源码位置: broad_phase_nxn.py → BroadPhaseExplicit

原理：
    - finalize() 时已预计算了所有合法碰撞对
    - 运行时只需检测这些对的 AABB 是否重叠
    - 最快的模式，适合固定拓扑场景

================================================================================
步骤3: 窄相检测 (Narrow Phase)
================================================================================

窄相的目标：对宽相输出的候选对，精确计算碰撞点、法线、穿透深度。

源码位置: narrow_phase.py → NarrowPhase.launch_custom_write()

窄相采用 **三阶段路由架构**：

                     候选对 (来自宽相)
                           │
                   ┌───────┴──────┐
                   │ Stage 1:     │
                   │ 基元路由核函数 │
                   └───────┬──────┘
            ┌──────┬───────┼──────────┬──────────┐
            │      │       │          │          │
          解析   GJK/MPR  网格+凸体   网格+平面   网格+网格
         基元对   缓冲区    缓冲区     缓冲区      缓冲区
            │      │       │          │          │
          直接   Stage 2  Stage 3a  Stage 3b   Stage 3c
          写出  GJK/MPR  中相→三角  逐顶点碰撞  SDF碰撞
                                   (+ 接触归约)

--- Stage 1: 基元碰撞核函数 ---

对于简单的基元对（球-球、球-盒、平面-球等），使用解析公式直接计算：

    球-球:    法线 = normalize(c1-c0), 距离 = ||c1-c0|| - r1 - r2
    球-盒:    将球心投影到盒内最近点，计算距离
    球-胶囊:  找胶囊线段上最近点，变成球-球问题
    平面-球:  点到平面距离
    平面-盒:  测试8个角点，取4个最深的
    胶囊-胶囊: 两线段最近点对，变成球-球

每种基元对的最大接触点数：
    球-球: 1    球-盒: 1    球-胶囊: 1    球-圆柱: 1
    胶囊-胶囊: 2  平面-盒: 4  平面-胶囊: 2  平面-圆柱: 4

对于复杂对（盒-盒、圆柱-盒等），路由到 Stage 2 (GJK/MPR)。
对于涉及网格的对，路由到 Stage 3。

--- Stage 2: GJK/MPR 碰撞检测 ---

GJK (Gilbert-Johnson-Keerthi) 算法：
    目的：判断两个凸体是否相交，如果不相交则求最近距离

    核心思想：
    两个凸体 A 和 B 的 Minkowski 差 A-B = {a-b | a∈A, b∈B}
    A 和 B 相交  ⟺  原点 ∈ A-B

    算法迭代地构建 A-B 上的一个单纯形（simplex）：
    1. 选择一个方向 d
    2. 在 A-B 上沿 d 方向找 support point: s = support_A(d) - support_B(-d)
    3. 如果 s 没越过原点 → 不相交
    4. 将 s 加入 simplex，更新搜索方向
    5. 如果 simplex 包含原点 → 相交
    6. 否则回到步骤1

    support(A, d) = A 中沿 d 方向最远的点
    对于球: support(d) = center + radius * normalize(d)
    对于盒: support(d) = (sign(d.x)*hx, sign(d.y)*hy, sign(d.z)*hz)

MPR (Minkowski Portal Refinement) 算法：
    目的：当 GJK 判定相交时，求穿透深度和方向

    核心思想：
    在 Minkowski 差的边界上找一个"门户"（portal），
    逐步精化直到穿过原点的法线稳定 → 穿透方向和深度

--- Stage 3a: 网格-凸体碰撞 ---

分两步：
1. 中相 (Midphase): 查询网格 BVH，找出与凸体 AABB 重叠的三角面
2. 精确检测: 对每个三角面-凸体对，运行 GJK/MPR（三角面也是凸体！）

接触归约 (Contact Reduction):
    网格碰撞会产生大量接触点，需要归约到可管理数量：
    - 将接触点按空间体素分桶
    - 每个体素保留7个代表性接触（6个空间方向极值 + 1个最深穿透）
    - 大幅减少接触数量同时保持质量

--- Stage 3b: 网格-平面碰撞 ---

最简单的网格碰撞：逐顶点测试到平面的距离。
可选接触归约（共享内存体素分桶）。

--- Stage 3c: 网格-网格碰撞 (SDF) ---

基于预计算的有符号距离场 (SDF) 检测：
1. finalize() 时为网格生成 SDF (wp.Volume, NanoVDB格式)
2. 运行时查询一个网格的顶点在另一个网格的 SDF 中的距离
3. 使用 Tiled Launch + 共享内存进行接触归约

================================================================================
步骤4: 软接触检测 (Particle-Shape Contacts)
================================================================================

源码位置: geometry/kernels.py → create_soft_contacts

粒子（布料/软体/流体的顶点）与形状之间的接触检测：

核函数逻辑 (维度 = particle_count × shape_count):
    particle_id = tid / shape_count
    shape_id = tid % shape_count

    # 1. 过滤检查
    if 粒子不活跃 or 形状不碰撞粒子 or world不匹配:
        return

    # 2. 将粒子位置变换到形状局部空间
    local_pos = inverse(shape_world_transform) * particle_world_pos

    # 3. 根据形状类型计算 SDF 距离
    switch shape_type:
        SPHERE:    d = ||local_pos|| - radius
        BOX:       d = box_sdf(local_pos, half_extents)
        CAPSULE:   d = capsule_sdf(local_pos, radius, half_height)
        CYLINDER:  d = cylinder_sdf(local_pos, radius, half_height)
        MESH:      d = mesh_query_point_sign_normal(mesh, local_pos)
        PLANE:     d = dot(local_pos, plane_normal) - plane_offset
        ...

    # 4. 如果距离小于 margin + particle_radius → 记录接触
    if d < margin + radius:
        index = atomic_add(contact_count, 1)  # 原子计数器
        contact_particle[index] = particle_id
        contact_shape[index] = shape_id
        contact_normal[index] = sdf_gradient  # 接触法线 = SDF 梯度
        contact_body_pos[index] = 形状表面最近点

================================================================================
步骤5: 接触写入 (Contact Writer)
================================================================================

源码位置: collide.py → write_contact @wp.func

这是将碰撞数据写入 Contacts 缓冲区的核心函数：

    输入: ContactData (窄相输出)
        - shape_a, shape_b: 碰撞的两个形状
        - contact_point_center: 接触点中心（世界空间）
        - contact_normal_a_to_b: A→B 方向的法线
        - contact_distance: 接触距离（正=分离，负=穿透）
        - radius_eff_a/b: 有效半径
        - thickness_a/b: 厚度偏移

    处理逻辑:
        1. 计算有符号距离 d = distance - total_separation
        2. 如果 d > contact_margin → 太远，跳过
        3. 用 atomic_add 获取写入索引
        4. 将接触点变换到各自 body 的局部坐标系
        5. 写入所有接触数据到 Contacts 数组

    关键转换:
        world_point → body_local_point:
            X_bw = inverse(body_q[body_id])  # 世界→刚体 变换
            local_point = transform_point(X_bw, world_point)

================================================================================
Contacts 数据结构
================================================================================

源码位置: contacts.py → Contacts 类

刚体接触 (Rigid Contacts):
    rigid_contact_count:     int[1]     → 活跃接触数（原子计数器）
    rigid_contact_shape0/1:  int[N]     → 两个碰撞形状的索引
    rigid_contact_point0/1:  vec3[N]    → 接触点（body局部坐标系）
    rigid_contact_offset0/1: vec3[N]    → 厚度偏移
    rigid_contact_normal:    vec3[N]    → 接触法线
    rigid_contact_thickness0/1: float[N] → 有效厚度

软接触 (Soft Contacts, 粒子用):
    soft_contact_count:      int[1]     → 活跃接触数
    soft_contact_particle:   int[N]     → 粒子索引
    soft_contact_shape:      int[N]     → 形状索引
    soft_contact_body_pos:   vec3[N]    → 形状表面接触点
    soft_contact_normal:     vec3[N]    → 接触法线

性能优化:
    - 两个计数器打包到 _counter_array[2] 中，clear() 只需零化这2个int
    - rigid contacts 不追踪梯度（enable_backward=False）
    - soft contacts 支持梯度追踪（用于可微仿真）
"""

# 这个文件是纯文档，不可直接运行。
# 请参考对应的源码文件深入学习。
#
# 建议阅读顺序：
# 1. newton/_src/sim/collide.py         → 整体管线流程
# 2. newton/_src/geometry/broad_phase_nxn.py → 宽相逻辑
# 3. newton/_src/geometry/narrow_phase.py    → 窄相路由和GJK/MPR
# 4. newton/_src/geometry/collision_primitive.py → 基元碰撞公式
# 5. newton/_src/geometry/kernels.py         → 软接触和SDF函数
# 6. newton/_src/sim/contacts.py             → 接触数据结构
