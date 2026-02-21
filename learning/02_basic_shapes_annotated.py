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
        # 【形状1】球体 SPHERE
        # ===================================================================
        # add_body(): 创建自由浮动的刚体（不属于关节链）
        # 与 add_link() 的区别：
        #   - add_body(): 自由刚体，需要用自由关节(free joint)约束
        #     （finalize 时自动添加）
        #   - add_link(): 专门用于关节链的刚体
        #
        # xform: 初始位姿 (wp.transform = 位置 + 四元数旋转)
        # key: 可选的名称字符串，用于调试和测试查找
        self.sphere_pos = wp.vec3(0.0, -2.0, drop_z)
        body_sphere = builder.add_body(
            xform=wp.transform(p=self.sphere_pos, q=wp.quat_identity()),
            key="sphere",
        )
        # add_shape_sphere(body, radius) → 半径为0.5的球
        # body: 这个形状附着到哪个刚体
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
        # 【形状6】三角网格 MESH
        # ===================================================================
        # 从 USD 文件中加载网格几何
        # 1. 用 pxr.Usd 打开 USD 文件
        # 2. 用 newton.usd.get_mesh() 提取网格数据
        # 3. 返回 newton.Mesh 对象（包含顶点和三角面）
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
        demo_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        self.mesh_pos = wp.vec3(0.0, 4.0, drop_z - 0.5)
        body_mesh = builder.add_body(
            xform=wp.transform(p=self.mesh_pos, q=wp.quat(0.5, 0.5, 0.5, 0.5)),
            key="mesh",
        )
        # add_shape_mesh(body, mesh) → 使用三角网格作为碰撞几何
        # 网格碰撞比基元碰撞更慢，但能表示任意形状
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
        # 【VBD 专用】图着色
        # ===================================================================
        # VBD 求解器使用 Gauss-Seidel 并行策略，需要图着色
        # 确保同一颜色的约束可以并行求解而不冲突
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
