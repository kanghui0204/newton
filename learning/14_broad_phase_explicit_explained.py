"""
Newton 参考文档 14：宽相碰撞检测三种模式详解（重点: EXPLICIT）
===============================================================

【源码位置】
  三种宽相模式定义:  newton/_src/sim/collide.py → BroadPhaseMode 枚举
  NXN 全对:         newton/_src/geometry/broad_phase_nxn.py → BroadPhaseAllPairs
  EXPLICIT 预计算:  newton/_src/geometry/broad_phase_nxn.py → BroadPhaseExplicit (第397行)
  SAP 扫描排序:     newton/_src/geometry/broad_phase_sap.py → BroadPhaseSAP
  碰撞对预计算:     newton/_src/sim/builder.py → find_shape_contact_pairs() (第9062行)

================================================================================
一、什么是"宽相碰撞检测"？为什么需要它？
================================================================================

碰撞检测分两步:
  宽相(Broad Phase): 快速过滤 → 哪些物体"可能"碰撞？
  窄相(Narrow Phase): 精确计算 → 这两个物体到底碰了吗？碰在哪？

为什么不直接做窄相？
  因为窄相很贵！GJK/MPR 算法需要多次迭代。

  如果场景有 100 个 shape:
    全部两两检测窄相 = 100×99/2 = 4950 次 GJK（太慢！）
    先用宽相过滤: 大部分 shape 离得很远，AABB 不重叠
    → 可能只有 50 对 AABB 重叠 → 只做 50 次 GJK（快100倍！）

  宽相用最简单的检测（AABB盒子是否重叠），把不可能碰撞的对排除掉。

================================================================================
二、三种宽相模式对比
================================================================================

┌─────────────────────────────────────────────────────────────────────┐
│  模式        │ 算法            │ 时间复杂度  │ 适用场景              │
├─────────────────────────────────────────────────────────────────────┤
│  NXN         │ 全对 AABB 检测  │ O(N²)      │ 小场景 (<100 shapes)  │
│  SAP         │ 排序+扫描       │ O(N log N) │ 大场景，动态拓扑       │
│  EXPLICIT    │ 只检预计算对     │ O(P)       │ 固定拓扑（默认！最快） │
│              │                 │ P=碰撞对数  │                       │
└─────────────────────────────────────────────────────────────────────┘

================================================================================
三、EXPLICIT 模式的完整流程（举例说明）
================================================================================

【场景: 3个球 + 1个地面】

  builder.add_ground_plane()                    # shape 0 (地面)
  builder.add_body(...); add_shape_sphere(...)   # shape 1 (球A)
  builder.add_body(...); add_shape_sphere(...)   # shape 2 (球B)
  builder.add_body(...); add_shape_sphere(...)   # shape 3 (球C)
  model = builder.finalize()

【步骤1: finalize() 时预计算碰撞对】

  源码: builder.py → find_shape_contact_pairs() (第9062行)

  算法（在 CPU 上执行，只在 finalize 时运行一次）:

    遍历所有 shape 对 (i, j)，过滤条件:
    ✓ 两个 shape 都有 COLLIDE_SHAPES 标志
    ✓ 在同一个 world (或至少一个是全局 world=-1)
    ✓ collision_group 兼容
    ✓ 不在排除列表中 (shape_collision_filter_pairs)

    场景中所有合法碰撞对:
      (0, 1)  地面 vs 球A   ← 要检测
      (0, 2)  地面 vs 球B   ← 要检测
      (0, 3)  地面 vs 球C   ← 要检测
      (1, 2)  球A  vs 球B   ← 要检测
      (1, 3)  球A  vs 球C   ← 要检测
      (2, 3)  球B  vs 球C   ← 要检测

    排除的对 (不检测):
      (0, 0), (1, 1), (2, 2), (3, 3)  ← 自己和自己不碰
      同一个 body 的多个 shape 之间（如果设了 collision_filter）

    结果存储为:
      model.shape_contact_pairs = wp.array([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])
      model.shape_contact_pair_count = 6

【步骤2: 每帧 collide() 时只检这6个对】

  源码: collide.py → CollisionPipeline.collide() (第562行)
        broad_phase_nxn.py → BroadPhaseExplicit.launch() (第411行)

  运行时（每帧执行）:

    1. 计算所有 shape 的 AABB（并行 kernel，每 shape 一个线程）

    2. BroadPhaseExplicit.launch():
       只启动 6 个 GPU 线程（= 碰撞对数量）
       每个线程检测一个预计算对的 AABB 是否重叠

       线程0: 检测 (0,1) 地面 vs 球A → AABB 重叠？ → 是 → 记录候选
       线程1: 检测 (0,2) 地面 vs 球B → AABB 重叠？ → 是 → 记录候选
       线程2: 检测 (0,3) 地面 vs 球C → AABB 重叠？ → 否 → 跳过（球C在空中）
       线程3: 检测 (1,2) 球A vs 球B  → AABB 重叠？ → 否 → 跳过（相距很远）
       线程4: 检测 (1,3) 球A vs 球C  → AABB 重叠？ → 否 → 跳过
       线程5: 检测 (2,3) 球B vs 球C  → AABB 重叠？ → 否 → 跳过

       结果: 候选对 = [(0,1), (0,2)]  只有2对需要窄相检测！

    3. 对候选对做窄相检测（GJK/MPR）→ 精确碰撞结果

【对比: 如果用 NXN 模式】

    NXN 不预计算碰撞对，每帧都要:
    启动 N×(N-1)/2 = 4×3/2 = 6 个线程（这里和 EXPLICIT 一样多）
    但每个线程不仅要检 AABB，还要:
      ① 从 thread_id 计算 shape 对 (二分搜索 + 下三角索引转换)
      ② 检查 world 兼容性
      ③ 检查 collision_group
      ④ 检查排除列表（又一次二分搜索）
      ⑤ 最后才检 AABB

    EXPLICIT 的线程只需:
      ① 读取预计算对 shape_pairs[tid]
      ② 检 AABB
    省掉了 ①②③④ 的计算！

【为什么 EXPLICIT 是默认模式？】

  大多数场景的碰撞拓扑在 finalize() 后就固定了:
  - 机器人和地面的碰撞对不会变
  - 多世界复制后每个世界内部的碰撞对不会变
  - 关节链内部的碰撞过滤不会变

  既然碰撞对不变，为什么每帧都重新算？预计算一次就够了！

  唯一需要用 NXN/SAP 的情况:
  - 新物体在运行时动态添加到场景中
  - collision_group 在运行时动态改变
  → 这些情况在 Newton 中不常见

================================================================================
四、多世界场景的碰撞对优化
================================================================================

【单机器人: 10个shape + 1个地面 = 11个shape】

  NXN: 11×10/2 = 55 个线程
  EXPLICIT: 可能只有 ~20 个合法碰撞对（排除了同 body 的 shape 对）

【100个世界 × 同一个机器人】

  总 shape 数 = 100 × 10 + 1(共享地面) = 1001

  NXN: 1001×1000/2 = 500,500 个线程
       大部分都在检"不同世界的shape" → 白白浪费！
       (world 0 的手臂和 world 99 的脚不可能碰撞)

  EXPLICIT: 100 × 20(每世界碰撞对) = 2,000 个线程
            只检合法碰撞对，完全不浪费
            → 快 250 倍！

  这就是为什么多世界仿真中 EXPLICIT 的优势特别大。

================================================================================
五、AABB 重叠检测 — 宽相的核心运算
================================================================================

AABB = Axis-Aligned Bounding Box（轴对齐包围盒）

  每个 shape 用一个"不旋转的盒子"包起来:
    aabb_lower = (min_x, min_y, min_z)
    aabb_upper = (max_x, max_y, max_z)

  两个 AABB 是否重叠？三个轴分别检查:
    overlap = (a.upper.x >= b.lower.x) and (a.lower.x <= b.upper.x)
          and (a.upper.y >= b.lower.y) and (a.lower.y <= b.upper.y)
          and (a.upper.z >= b.lower.z) and (a.lower.z <= b.upper.z)

  如果任何一个轴不重叠 → 两个 shape 不可能碰撞 → 跳过窄相
  如果三个轴都重叠 → "可能"碰撞 → 送去窄相精确检测

  图示:
         ┌─────┐
         │  A  │
         │  ┌──┼──┐
         └──┼──┘  │   ← AABB 重叠区域（可能碰撞）
            │  B  │
            └─────┘

         ┌─────┐
         │  A  │     ┌─────┐
         │     │     │  B  │   ← AABB 不重叠（不可能碰撞）
         └─────┘     └─────┘

================================================================================
六、什么是"固定拓扑"？举例说明
================================================================================

"拓扑" = "谁和谁可能碰撞"的关系图（和位置无关，只和身份有关）

【固定拓扑 — 碰撞关系从头到尾不变】

  例: 一个机器人站在地面上

    永远要检测的碰撞对:
      脚 vs 地面
      膝盖 vs 地面
      左手 vs 右手（自碰撞）
      身体 vs 地面

    永远不检测的碰撞对:
      左大腿 vs 左小腿（关节相连，已过滤）
      world 0 的手 vs world 1 的脚（不同世界）

    不管机器人跑到哪里、摆什么姿势，这个"谁和谁检测"的列表都不变
    → 固定拓扑 → EXPLICIT 最合适

  再例: 100个世界各有一个机器人
    每个世界内部碰撞对相同，世界之间不碰撞
    → 碰撞对列表在整个仿真过程中不变 → 固定拓扑

【动态拓扑 — 碰撞关系会变】

  例1: 游戏中不断生成新敌人
    t=0:   场景 3 个物体 → 碰撞对 3 个
    t=100: 新增 5 个敌人 → 碰撞对 28 个
    → 预计算的列表过时了 → 必须每帧重算 → 用 NXN/SAP

  例2: 运行时改变 collision_group
    t=0:   球A group=1, 球B group=1 → 同组要碰
    t=100: 把球A的group改成2 → 不同组不碰了
    → 碰撞对列表变了 → 预计算失效 → 用 NXN/SAP

【Newton 大多数场景都是固定拓扑】
  因为 Newton 的设计是: finalize() 后场景结构不再改变
  所以 EXPLICIT 是默认模式

================================================================================
七、EXPLICIT 怎么控制"谁和谁碰、谁和谁不碰"？
================================================================================

EXPLICIT 不需要你单独"配合"什么——finalize() 时自动处理了所有过滤逻辑。
你只需要在构建场景时正确设置碰撞属性：

【工具1: collision_group — 碰撞分组】

  源码: builder.py → ShapeConfig.collision_group (第186行)

  规则:
    group = 0 → 不与任何东西碰撞（幽灵）
    group > 0 → 只与同组或负组碰撞
    group < 0 → 与所有组碰撞（除了对应正组的取反）

  例子:
    builder.add_shape_sphere(body_a, cfg=ShapeConfig(collision_group=1))  # 组1
    builder.add_shape_sphere(body_b, cfg=ShapeConfig(collision_group=1))  # 组1
    builder.add_shape_sphere(body_c, cfg=ShapeConfig(collision_group=2))  # 组2

    碰撞结果:
      (A, B) → 同组1 → 碰撞 ✓
      (A, C) → 不同组(1 vs 2) → 不碰撞 ✗
      (B, C) → 不同组(1 vs 2) → 不碰撞 ✗

  用途: 把不同类别的物体分开
    所有障碍物: group=1
    所有机器人: group=2
    → 障碍物之间不碰撞（优化），机器人之间不碰撞

【工具2: collision_filter_parent — 关节链碰撞过滤】

  源码: builder.py → ShapeConfig.collision_filter_parent (第188行)

  当你创建关节时:
    add_joint_revolute(parent=body_a, child=body_b, collision_filter_parent=True)
    → 自动过滤 body_a 和 body_b 上所有 shape 之间的碰撞

  意义: 用关节连接的两个 body 通常"紧挨着"
        如果不过滤，它们的 shape 会永远穿透 → 产生无意义的碰撞力
        默认 True → 自动过滤关节相邻body之间的碰撞

  例: 机器人手臂
    上臂body --- 关节(collision_filter_parent=True) --- 前臂body
    → 上臂和前臂的 shape 不会碰撞检测（因为它们通过关节连接）

【工具3: shape_collision_filter_pairs — 手动排除特定对】

  源码: builder.py → shape_collision_filter_pairs (第658行)
        builder.add_collision_filter(shape_a, shape_b)

  用法:
    shape_a = builder.add_shape_sphere(body_a)
    shape_b = builder.add_shape_sphere(body_b)
    builder.add_collision_filter(shape_a, shape_b)  # 这两个永远不碰撞

  → find_shape_contact_pairs() 在构建碰撞对时会跳过这些对

【工具4: has_shape_collision — 单个形状关闭碰撞】

  源码: builder.py → ShapeConfig.has_shape_collision (第190行)

  用法:
    cfg = ShapeConfig(has_shape_collision=False)
    builder.add_shape_sphere(body, cfg=cfg)  # 这个shape不参与碰撞
    → 只用于可视化，不检测碰撞

【工具5: enable_self_collisions — 关节链内部碰撞】

  用于 add_usd() / add_urdf():
    builder.add_usd("robot.usda", enable_self_collisions=False)
    → False: 同一个关节链内的所有 shape 之间不碰撞（默认）
    → True:  允许自碰撞（如手碰到腿）

【这些过滤在 EXPLICIT 中怎么配合？】

  finalize() → find_shape_contact_pairs() 自动做了以下过滤:

    遍历每对 (shape_i, shape_j):
      ✗ collision_group 不兼容 → 跳过
      ✗ 不同 world 且都不是全局 → 跳过
      ✗ 在 shape_collision_filter_pairs 排除列表中 → 跳过
      ✗ has_shape_collision=False → 跳过
      ✓ 全部通过 → 加入碰撞对列表

  你不需要额外做任何事！只需要在构建时正确设置属性：
    - collision_group 分组
    - collision_filter_parent=True（关节默认开启）
    - enable_self_collisions=False（导入时默认关闭）

  finalize() 会把所有过滤逻辑整合进预计算的碰撞对列表
  之后 EXPLICIT 只检测这个列表中的对

================================================================================
八、EXPLICIT 也是对每对检测 AABB，和 NXN 有什么区别？
================================================================================

你可能会问：EXPLICIT 也是"对每对检测 AABB"，那它和 NXN 的 O(N²) 有什么本质区别？

区别在于"对哪些对"检测：

  NXN:      对所有 N×(N-1)/2 个对检测（包含大量"垃圾对"）
  EXPLICIT: 只对预计算的 P 个合法碰撞对检测（P 远小于 N²）

  NXN 的每个线程还要做额外工作：
    ① 从 thread_id 算出 shape 对（二分搜索+下三角索引转换）
    ② 检查 world 兼容性
    ③ 检查 collision_group
    ④ 检查排除列表（又一次二分搜索）
    ⑤ 最后才检 AABB
    → 99% 的线程做完①②③④后发现"不用检"，白白浪费

  EXPLICIT 的每个线程只做：
    ① 读取 shape_pairs[tid]（预计算好的）
    ② 检 AABB
    → 没有任何浪费

  具体数字对比（100个世界 × 10个shape + 1个共享地面）:

    NXN:
      N = 1001
      线程数 = 1001 × 1000 / 2 = 500,500
      其中 ~99% 是"不同世界的shape对" → 过滤后才发现不用检
      → 浪费 ~495,000 次过滤计算

    EXPLICIT:
      P ≈ 100 × 20(每世界内的合法碰撞对) = 2,000
      线程数 = 2,000
      每个线程都是有意义的检测
      → 零浪费，快 250 倍

  两者本质区别：
    NXN 是"先发射所有线程，每个线程自己判断该不该干活"
    EXPLICIT 是"只发射需要干活的线程"

================================================================================
九、用户代码中如何选择宽相模式
================================================================================

  # 方式1: 通过 CollisionPipeline 参数
  pipeline = newton.CollisionPipeline(
      model,
      broad_phase_mode=newton.BroadPhaseMode.EXPLICIT,  # 默认值
  )

  # 方式2: 通过命令行参数（在示例中）
  # uv run -m newton.examples basic_shapes --broad-phase-mode nxn
  # uv run -m newton.examples basic_shapes --broad-phase-mode sap
  # uv run -m newton.examples basic_shapes --broad-phase-mode explicit

  # 方式3: 通过示例基础设施
  pipeline = newton.examples.create_collision_pipeline(model, args)
  # args.broad_phase_mode 来自命令行
"""
