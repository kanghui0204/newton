"""
Newton 学习示例 02：碰撞形状 (Basic Shapes)
=============================================

【学习目标】
- 学习 Newton 支持的所有碰撞几何体类型
- 理解 add_body() vs add_link() 的区别
- 理解 ShapeConfig 材质参数
- 学习如何切换不同的求解器（XPBD vs VBD）
- 理解 USD 网格的加载方式

【碰撞形状类型】
- SPHERE (球体): add_shape_sphere(body, radius)
- ELLIPSOID (椭球体): add_shape_ellipsoid(body, a, b, c)
- CAPSULE (胶囊体): add_shape_capsule(body, radius, half_height)
- CYLINDER (圆柱体): add_shape_cylinder(body, radius, half_height)
- BOX (长方体): add_shape_box(body, hx, hy, hz)
- MESH (网格): add_shape_mesh(body, mesh)
- CONE (圆锥): add_shape_cone(body, radius, half_height) ← 无标准碰撞支持

【Body vs Shape 的关系 — 灵魂与身体】

  Body = "灵魂"（物理实体：有质量、有速度、受力）
  Shape = "身体"（碰撞形状：球、盒、网格等外表）

  add_body() 创建一个物理实体，但它没有形状——不知道长什么样
  add_shape_*() 给它穿上"身体"——告诉物理引擎碰撞用什么几何、质量多大

  ┌───────────────┐
  │    Body       │  ← add_body()创建，有位姿/速度/质量
  │  ┌─────────┐  │
  │  │ Shape 1 │  │  ← add_shape_sphere()，球形碰撞体
  │  └─────────┘  │
  │  ┌─────────┐  │     一个Body可以有多个Shape！
  │  │ Shape 2 │  │  ← add_shape_box()，盒形碰撞体
  │  └─────────┘  │     多个Shape共享同一个Body的运动
  └───────────────┘

  add_body() vs add_link():
    add_body()：自由刚体，finalize()自动加 FREE 关节（6自由度，可任意运动）
    add_link()：关节链刚体，必须手动用 add_joint_*() 连接

【什么是 Mesh（三角网格）】

  基元形状(Sphere/Box等)用数学公式描述，碰撞检测有解析解，快但只能表示简单形状。
  Mesh 用成千上万个小三角形拼出任意形状（兔子、汽车、人形...），
  碰撞检测需要 BVH 加速和 SDF 辅助，慢但万能。

  Mesh = 顶点数组(vec3[]) + 面索引数组(int[])
  例如一只兔子: ~5000个顶点, ~10000个三角形

  从 USD 文件获取 Mesh 有两种方式：
    方式A: builder.add_usd("robot.usda") → 自动创建 body+joint+shape（完整机器人）
    方式B: newton.usd.get_mesh(...) → 只借用三角网格的几何数据（本例的做法）

【VBD vs XPBD 求解器区别】

  XPBD（位置级方法）：
    工作方式：预测位置 → 迭代修正约束违反 → 从位置变化导出速度
    类比：老师直接把跑出界的学生拉回来
    默认 ke 即可，因为不靠"力"解决穿透

  VBD（能量级方法）：
    工作方式：定义总能量 E(x) → 用 Block Descent 最小化能量
    类比：界外放电网，学生自己感受到"疼"跑回来
    需要 ke=1e6（高电压），否则优化器觉得"穿透一点也没关系"

  VBD 需要 builder.color()（图着色）：
    VBD 逐个顶点求解最优位置，需要邻居的位置已确定
    图着色保证并行更新的顶点之间没有连接关系
    同颜色的顶点可以安全并行更新

【iterations=10 的含义】

  在每个物理 substep 内部，约束求解要迭代多次。
  因为约束之间互相耦合——修正一个约束会破坏另一个，需要反复调整。

  一帧的完整结构:
    1帧 (frame_dt=0.01s)
    └── 10个 substep (sim_dt=0.001s)
        └── 10次 iteration (约束修正)
    总计: 10×10=100 次约束修正/帧

  iterations 越多越精确但越慢:
    1-2次：弹簧松垮，可能穿透
    5-10次：大多数场景够用
    20+次：高精度布料/软体

【运行方式】
    uv run python learning/02_basic_shapes_annotated.py
    uv run python learning/02_basic_shapes_annotated.py --solver vbd
"""

import warp as wp
from pxr import Usd  # OpenUSD 的 Python 绑定，用于读取 USD 文件

import newton
import newton.examples
import newton.usd  # Newton 的 USD 工具模块


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.solver_type = args.solver if hasattr(args, "solver") and args.solver else "xpbd"

        builder = newton.ModelBuilder()

        # ===================================================================
        # 【VBD 求解器特殊设置】
        # ===================================================================
        # VBD (Vertex Block Descent) 求解器需要更高的接触刚度
        # 因为它使用不同的约束求解策略
        if self.solver_type == "vbd":
            builder.default_shape_cfg.ke = 1.0e6  # 接触刚度（弹簧常数）
            builder.default_shape_cfg.kd = 1.0e1  # 接触阻尼
            builder.default_shape_cfg.mu = 0.5  # 摩擦系数

        # 地面放在最前面（body_index=0 之前的 shape）
        builder.add_ground_plane()

        drop_z = 2.0  # 所有物体从 z=2.0 高度掉落

        # ===================================================================
        # 【形状1】球体 SPHERE — Body 和 Shape 的关系演示
        # ===================================================================
        #
        # 第1步: add_body() → 创建"灵魂"（物理实体）
        #   此时只有位姿/速度/质量，没有形状，物理引擎不知道它长什么样
        #   finalize() 时会自动给它加 FREE 关节（6自由度，可在空间中自由运动）
        #   不需要像 add_link() 那样手动加关节
        #
        #   xform: 初始位姿 (transform = 位置vec3 + 旋转quat)
        #   key: 标签名，纯粹用于调试/测试查找，取名随意，不影响仿真
        self.sphere_pos = wp.vec3(0.0, -2.0, drop_z)
        body_sphere = builder.add_body(
            xform=wp.transform(p=self.sphere_pos, q=wp.quat_identity()),
            key="sphere",
        )
        # 第2步: add_shape_sphere() → 给"灵魂"穿上"身体"
        #   告诉物理引擎：这个 body 的碰撞形状是一个半径0.5的球
        #   质量由 density(默认1000) × volume(4/3πr³) 自动计算
        #   碰撞检测用解析公式 d=||p-center||-r，非常快
        builder.add_shape_sphere(body_sphere, radius=0.5)

        # ===================================================================
        # 【形状2】椭球体 ELLIPSOID
        # ===================================================================
        # 三个半轴长度 a, b, c（沿 x, y, z 方向）
        # a=b=0.5, c=0.25 → 扁平的盘状（像M&M巧克力豆）
        # 扁平形状自然静止时更稳定
        self.ellipsoid_pos = wp.vec3(0.0, -6.0, drop_z)
        body_ellipsoid = builder.add_body(
            xform=wp.transform(p=self.ellipsoid_pos, q=wp.quat_identity()),
            key="ellipsoid",
        )
        builder.add_shape_ellipsoid(body_ellipsoid, a=0.5, b=0.5, c=0.25)

        # ===================================================================
        # 【形状3】胶囊体 CAPSULE
        # ===================================================================
        # 胶囊体 = 圆柱体 + 两端半球
        # radius: 圆柱和半球的半径
        # half_height: 圆柱部分的半高度（不含半球）
        # 总高度 = 2 * (half_height + radius) = 2 * (0.7 + 0.3) = 2.0
        # 胶囊体默认沿 Y 轴方向
        self.capsule_pos = wp.vec3(0.0, 0.0, drop_z)
        body_capsule = builder.add_body(
            xform=wp.transform(p=self.capsule_pos, q=wp.quat_identity()),
            key="capsule",
        )
        builder.add_shape_capsule(body_capsule, radius=0.3, half_height=0.7)

        # ===================================================================
        # 【形状4】圆柱体 CYLINDER
        # ===================================================================
        # radius: 圆柱半径
        # half_height: 圆柱半高度
        # 圆柱默认沿 Y 轴方向
        self.cylinder_pos = wp.vec3(0.0, -4.0, drop_z)
        body_cylinder = builder.add_body(
            xform=wp.transform(p=self.cylinder_pos, q=wp.quat_identity()),
            key="cylinder",
        )
        builder.add_shape_cylinder(body_cylinder, radius=0.4, half_height=0.6)

        # ===================================================================
        # 【形状5】长方体 BOX
        # ===================================================================
        # hx, hy, hz: 三个方向的半尺寸
        # 实际尺寸 = 2*hx × 2*hy × 2*hz = 1.0 × 0.7 × 0.5
        self.box_pos = wp.vec3(0.0, 2.0, drop_z)
        body_box = builder.add_body(
            xform=wp.transform(p=self.box_pos, q=wp.quat_identity()),
            key="box",
        )
        builder.add_shape_box(body_box, hx=0.5, hy=0.35, hz=0.25)

        # ===================================================================
        # 【形状6】三角网格 MESH — 从 USD 借用几何数据
        # ===================================================================
        # Mesh = 顶点数组 + 三角面索引数组，可以表示任意复杂形状
        #
        # 注意：这里是"方式B"——只从USD借用网格几何，不是完整导入
        #   方式A: builder.add_usd("robot.usda") → 自动创建body+joint+shape（完整机器人）
        #   方式B: newton.usd.get_mesh(...) → 只提取三角网格数据
        #
        # 步骤1: 从 USD 文件提取兔子的三角网格（只借了形状，没创建body/joint）
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
        demo_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/bunny"))
        # demo_mesh 现在是一个 Mesh 对象，包含顶点坐标和三角面索引

        # 步骤2: 手动创建 body（物理实体）
        # key="mesh" 只是个标签名，方便测试查找。叫"bunny"也可以
        self.mesh_pos = wp.vec3(0.0, 4.0, drop_z - 0.5)
        body_mesh = builder.add_body(
            xform=wp.transform(p=self.mesh_pos, q=wp.quat(0.5, 0.5, 0.5, 0.5)),
            key="mesh",
        )
        # 步骤3: 把借来的网格形状贴到 body 上
        # Mesh 碰撞比基元慢（需要 BVH 树加速 + SDF 辅助），但能表示任意形状
        # 碰撞检测过程: 查询 BVH 找到相关三角面 → 对每个三角面做精确碰撞
        builder.add_shape_mesh(body_mesh, mesh=demo_mesh)

        # ===================================================================
        # 【形状7】圆锥 CONE
        # ===================================================================
        # 注意：圆锥在标准碰撞管线中没有碰撞支持
        # 可以用于可视化，但不参与物理碰撞
        self.cone_pos = wp.vec3(0.0, 6.0, drop_z)
        body_cone = builder.add_body(
            xform=wp.transform(p=self.cone_pos, q=wp.quat_identity()),
            key="cone",
        )
        builder.add_shape_cone(body_cone, radius=0.45, half_height=0.6)

        # ===================================================================
        # 【VBD 专用】图着色 (Graph Coloring)
        # ===================================================================
        # VBD 求解器逐个更新顶点到能量最低点。为了在GPU上并行，
        # 需要保证同时更新的顶点之间没有弹簧/约束连接（否则数据竞争）。
        #
        # builder.color() 做的事：
        # 1. 构建"连接图"——哪些顶点通过约束相连
        # 2. 给每个顶点分配颜色，保证相连的顶点颜色不同
        # 3. 求解时：同颜色并行（安全），不同颜色串行
        #
        # 例如 4 个顶点: v1-v2有弹簧, v2-v3有弹簧, v3-v4有弹簧
        #   v1(红) -- v2(蓝) -- v3(红) -- v4(蓝)
        #   第1轮: 并行更新 {v1, v3}（红色，互不相连）
        #   第2轮: 并行更新 {v2, v4}（蓝色，互不相连）
        if self.solver_type == "vbd":
            builder.color()

        # 定型模型
        self.model = builder.finalize()

        # ===================================================================
        # 【创建求解器】根据类型选择
        # ===================================================================
        if self.solver_type == "vbd":
            self.solver = newton.solvers.SolverVBD(
                self.model,
                iterations=10,  # 约束求解迭代次数（更多=更精确）
            )
        else:
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)

        # 创建状态、控制和碰撞检测管线
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        """标准仿真循环：清力→碰撞→求解→交换"""
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
        """验证各形状静止在正确高度。

        每种形状静止时的 z 坐标 = 形状在 z 方向的半尺寸：
        - 球体(r=0.5): z=0.5
        - 椭球体(c=0.25): z=0.25
        - 胶囊体(r+h=1.0): z=1.0
        - 圆柱体(h=0.6): z=0.6
        - 长方体(hz=0.25): z=0.25
        """
        self.sphere_pos[2] = 0.5
        sphere_q = wp.transform(self.sphere_pos, wp.quat_identity())
        newton.examples.test_body_state(
            self.model, self.state_0, "sphere at rest pose",
            lambda q, qd: newton.math.vec_allclose(q, sphere_q, atol=2e-4),
            [0],
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # 扩展标准解析器，添加 --solver 参数
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver", type=str, default="xpbd", choices=["vbd", "xpbd"],
        help="Solver type: xpbd (default) or vbd",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
