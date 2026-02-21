"""
Newton 参考文档 13：body_q vs shape_transform + Warp Tile/Shared Memory
========================================================================

================================================================================
一、body_q vs shape_transform 的区别
================================================================================

这两个都是 wp.transform（7个float：位置+旋转），但含义完全不同：

【body_q — 刚体在世界中的当前位姿（动态，每步都变）】

  存在哪里: State.body_q （State = 动态量）
  含义:     第i个 body 此刻在世界坐标系中的位置和朝向
  变化:     每一步都在变（body 在运动！）

  例: body_q[0] = transform((0,0,5), quat_identity)
      → body 0 此刻在世界坐标 (0,0,5)，没有旋转
      下一步可能变成 transform((0,0,4.9), ...)  → body 掉了一点

【shape_transform — 形状相对于 body 的固定偏移（静态，永远不变）】

  存在哪里: Model.shape_transform （Model = 静态量）
  含义:     第j个 shape 相对于它所属 body 的偏移
  变化:     永远不变（形状贴在 body 上，不会相对 body 运动）

  例: shape_transform[0] = transform((-0.5,0,0), quat_identity)
      → shape 0 的中心在 body 原点左边 0.5 米处

【组合：shape 在世界中的实际位置】

  碰撞检测需要知道 shape 在世界中的位置：

  shape世界位姿 = body_q[body_id] × shape_transform[shape_id]
                  "body在世界中在哪"   "shape相对body偏移多少"

  源码: collide.py → compute_shape_aabbs 核函数（第199行）
    if rigid_id == -1:
        X_ws = shape_transform[shape_id]    # 固定形状（地面等）
    else:
        X_ws = body_q[rigid_id] * shape_transform[shape_id]  # 动态形状

【例子：哑铃 = 1个body + 3个shape】

  builder.add_body(xform=..., key="dumbbell")          # body 0
  builder.add_shape_sphere(body, xform=tf((-0.5,0,0))) # shape 0: 左球
  builder.add_shape_sphere(body, xform=tf((+0.5,0,0))) # shape 1: 右球
  builder.add_shape_box(body)                           # shape 2: 中间杆

  Model (永远不变):
    shape_transform[0] = tf((-0.5,0,0))   # 左球在body左边0.5m
    shape_transform[1] = tf((+0.5,0,0))   # 右球在body右边0.5m
    shape_transform[2] = tf((0,0,0))       # 杆在body中心
    shape_body = [0, 0, 0]                 # 3个shape都属于body 0

  State (每步在变):
    t=0: body_q[0] = tf((0,0,5))  → 左球世界位置 = (0,0,5)×(-0.5,0,0) = (-0.5,0,5)
    t=1: body_q[0] = tf((0,0,4.9)) → 左球世界位置 = (-0.5,0,4.9) → body掉了,shape跟着掉

  body_q 变 → shape 的世界位置跟着变
  shape_transform 不变 → shape 相对 body 的偏移永远固定

================================================================================
二、Warp 的 Tile API 和 Shared Memory
================================================================================

【回答：Warp 能用 shared memory 吗？】

  能！Warp 通过 Tile API 提供 shared memory 访问。
  这不是像 CUDA C 那样直接声明 __shared__，
  而是通过高层 tile 操作自动管理 shared memory。

【Warp Tile API 的核心概念】

  普通 kernel (wp.launch):
    每个线程独立工作，不能共享数据
    没有 block 内协作

  Tiled kernel (wp.launch_tiled):
    一个 block 内的线程可以协作
    通过 tile 操作在 shared memory 中读写数据
    类似 CUDA 的 cooperative groups + shared memory

【基本用法】

  @wp.kernel
  def my_tiled_kernel(
      input: wp.array2d(dtype=float),
      output: wp.array2d(dtype=float),
  ):
      # 从全局内存加载数据到 shared memory（tile）
      # shape: tile 的尺寸 (行, 列)
      a = wp.tile_load(input, shape=(32, 32))

      # 在 shared memory 中创建零矩阵
      b = wp.tile_zeros(shape=(32, 32), dtype=float)

      # 显式声明使用 shared memory 存储
      c = wp.tile_ones(shape=(32, 32), dtype=float, storage="shared")

      # tile 算术运算（在 shared memory 中执行，block内线程协作）
      d = wp.tile_matmul(a, c)   # 矩阵乘法
      result = a + b             # 元素加法

      # 从 shared memory 写回全局内存
      wp.tile_store(output, result)

  # 启动方式：用 launch_tiled 而不是 launch
  wp.launch_tiled(
      my_tiled_kernel,
      dim=[batch_size],          # tile 的数量
      inputs=[input_array],
      outputs=[output_array],
      block_dim=256,             # 每个 block 的线程数
  )

【可用的 Tile 操作】

  创建 tile:
    wp.tile_zeros(shape, dtype)              → 全零 tile
    wp.tile_ones(shape, dtype)               → 全一 tile
    wp.tile_load(array, shape, offset)       → 从全局内存加载到 shared memory
    wp.tile_ones(..., storage="shared")      → 显式指定 shared memory

  写回:
    wp.tile_store(array, tile)               → 从 shared memory 写回全局内存

  运算:
    wp.tile_matmul(A, B)                     → 矩阵乘法
    wp.tile_view(tile, offset, shape)        → 获取 tile 的子视图
    wp.tile_assign(dst, src, offset)         → 将 src 赋值到 dst 的子区域
    tile + tile, tile * scalar               → 元素运算

  高级:
    wp.tile_reduce(tile, op)                 → 归约操作（求和等）
    wp.tile_sort(tile)                       → 排序

【Newton 中实际使用 Shared Memory 的地方】

  1. Featherstone 求解器 — 关节空间矩阵运算
     源码: newton/_src/solvers/featherstone/kernels.py 第1064行

     @wp.kernel
     def eval_dense_gemm_tile(J_arr, M_arr, H_arr):
         articulation = wp.tid()
         # 加载雅可比矩阵到 shared memory
         J = wp.tile_load(J_arr[articulation], shape=(6*num_joints, num_dofs))
         P = wp.tile_zeros(shape=(6*num_joints, num_dofs), dtype=float)
         for i in range(num_joints):
             # 加载 6×6 质量矩阵块
             M_body = wp.tile_load(M_arr[articulation], shape=(6,6), offset=(i*6, i*6))
             J_body = wp.tile_view(J, offset=(i*6, 0), shape=(6, num_dofs))
             # 矩阵乘法 in shared memory
             P_body = wp.tile_matmul(M_body, J_body)
             wp.tile_assign(P, P_body, offset=(i*6, 0))
         # H = J^T × M × J (在 shared memory 中完成)
         H = wp.tile_matmul(wp.tile_transpose(J), P)
         wp.tile_store(H_arr[articulation], H)

     用途: 计算关节空间惯性矩阵 H = Jᵀ M J
     好处: 矩阵乘法在 shared memory 中完成，避免反复读写全局内存

  2. 窄相碰撞检测 — 网格-平面和网格-网格接触归约
     源码: newton/_src/geometry/narrow_phase.py 第1517行

     wp.launch_tiled(
         kernel=self.mesh_plane_contacts_kernel,
         dim=(self.num_tile_blocks,),
         inputs=...,
         block_dim=self.tile_size_mesh_plane,
     )

     用途: 每个 tile block 处理一个 mesh-plane 对
           在 shared memory 中做接触点的体素分桶归约
           大幅减少需要输出的接触点数量

  3. MPM 求解器 — 流变学耦合求解
     源码: newton/_src/solvers/implicit_mpm/solve_rheology.py

     用途: Gauss-Seidel 迭代求解速度-应力-碰撞的耦合系统
           在 shared memory 中存储网格节点数据

  4. SAP 宽相排序
     源码: newton/_src/geometry/broad_phase_sap.py

     用途: Sweep-and-Prune 的 AABB 排序

  5. IK 求解器 — 矩阵运算
     源码: newton/_src/sim/ik/ik_lm_optimizer.py, ik_lbfgs_optimizer.py

     用途: Levenberg-Marquardt/LBFGS 的矩阵分解和求解

【Warp Tile vs CUDA __shared__】

  CUDA C:
    __shared__ float smem[256];    ← 手动声明、手动同步 (__syncthreads)
    低层、灵活、容易出错

  Warp Tile:
    a = wp.tile_zeros(...)         ← 自动管理 shared memory
    wp.tile_matmul(a, b)           ← 自动同步
    高层、安全、不需要手动同步

  Warp 在编译时自动计算每个 kernel 需要的 shared memory 大小，
  自动处理同步（不需要 __syncthreads），大幅降低使用门槛。

【Warp Tile 测试文件（学习参考）】

  warp/tests/tile/test_tile_shared_memory.py  ← shared memory 专项测试
  warp/tests/tile/test_tile_matmul.py         ← 矩阵乘法
  warp/tests/tile/test_tile_load.py           ← 加载/存储
  warp/tests/tile/test_tile_reduce.py         ← 归约
  warp/tests/tile/test_tile_sort.py           ← 排序

  这些测试文件是学习 Tile API 最好的入门材料。
"""
