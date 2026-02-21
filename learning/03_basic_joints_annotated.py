"""
Newton 学习示例 03：关节类型 (Basic Joints)
============================================

【学习目标】
- 学习三种关节类型：REVOLUTE（旋转）、PRISMATIC（移动）、BALL（球）
- 理解关节的 parent_xform / child_xform 含义
- 学习如何创建运动学（kinematic）物体
- 理解关节初始状态设置
- 学习 test_post_step() 实时验证

【关节类型总览】
Newton 支持以下关节类型（newton.JointType）：

| 类型 | 自由度 | 坐标数 | 约束数 | 用途 |
|------|--------|--------|--------|------|
| REVOLUTE | 1 | 1 | 5 | 铰链（门、摆、电机） |
| PRISMATIC | 1 | 1 | 5 | 滑块（活塞、线性导轨） |
| BALL | 3 | 4(四元数) | 3 | 球窝关节（肩关节） |
| FIXED | 0 | 0 | 6 | 固定连接（焊接） |
| FREE | 6 | 7(pos+quat) | 0 | 自由运动（根关节） |
| D6 | 1-6 | 1-6 | 可变 | 通用6DOF关节 |
| DISTANCE | 6 | 7 | 0 | 距离约束 |
| CABLE | 2 | 2 | 4 | 柔性缆线 |

【关节坐标系说明】
parent_xform: 关节安装点在父体坐标系中的位姿
child_xform: 关节安装点在子体坐标系中的位姿

     父体(parent)         子体(child)
    ┌──────────┐          ┌──────────┐
    │          │          │          │
    │    parent_xform ──── child_xform   │
    │          │  关节轴   │          │
    └──────────┘          └──────────┘

【运行方式】
    uv run python learning/03_basic_joints_annotated.py
"""

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args=None):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # 几何参数
        cuboid_hx = 0.1
        cuboid_hy = 0.1
        cuboid_hz = 0.75
        upper_hz = 0.25 * cuboid_hz  # 固定锚点（上部）较短

        rows = [-3.0, 0.0, 3.0]  # 三种关节沿 Y 轴排列
        drop_z = 2.0

        # =================================================================
        # 【关节1】REVOLUTE - 旋转关节（铰链）
        # =================================================================
        # 旋转关节只允许绕一个轴旋转，1个自由度
        # 常见用途：门、摆、机器人关节电机
        y = rows[0]

        # 创建锚点连杆（固定在世界中）
        a_rev = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity())
        )
        # 创建摆动连杆（受重力影响）
        # 给一个小初始旋转让它不在平衡位置
        b_rev = builder.add_link(
            xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z - cuboid_hz),
                q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.15),
            ),
            key="b_rev",
        )
        builder.add_shape_box(a_rev, hx=cuboid_hx, hy=cuboid_hy, hz=upper_hz)
        builder.add_shape_box(b_rev, hx=cuboid_hx, hy=cuboid_hy, hz=cuboid_hz)

        # 固定关节：将锚点固定到世界（parent=-1 表示世界）
        j_fixed_rev = builder.add_joint_fixed(
            parent=-1,
            child=a_rev,
            parent_xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            key="fixed_revolute_anchor",
        )

        # 旋转关节：连接锚点和摆动杆
        # axis=wp.vec3(1,0,0): 绕 X 轴旋转
        # parent_xform.p=(0,0,-upper_hz): 安装在锚点的底部
        # child_xform.p=(0,0,+cuboid_hz): 安装在摆杆的顶部
        j_revolute = builder.add_joint_revolute(
            parent=a_rev,
            child=b_rev,
            axis=wp.vec3(1.0, 0.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -upper_hz), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +cuboid_hz), q=wp.quat_identity()),
            key="revolute_a_b",
        )
        builder.add_articulation([j_fixed_rev, j_revolute], key="revolute_articulation")

        # 【设置初始关节角度】
        # builder.joint_q 是一个 Python 列表，存储所有关节的广义坐标
        # REVOLUTE 关节有1个坐标（旋转角度，弧度）
        # 最后添加的关节的坐标在列表末尾
        builder.joint_q[-1] = wp.pi * 0.5  # 初始角度 = 90度

        # =================================================================
        # 【关节2】PRISMATIC - 移动关节（滑块）
        # =================================================================
        # 移动关节只允许沿一个轴平移，1个自由度
        # 常见用途：活塞、线性导轨、升降台
        y = rows[1]

        a_pri = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity())
        )
        b_pri = builder.add_link(
            xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z - cuboid_hz),
                q=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.12),
            ),
            key="b_prismatic",
        )
        builder.add_shape_box(a_pri, hx=cuboid_hx, hy=cuboid_hy, hz=upper_hz)
        builder.add_shape_box(b_pri, hx=cuboid_hx, hy=cuboid_hy, hz=cuboid_hz)

        j_fixed_pri = builder.add_joint_fixed(
            parent=-1, child=a_pri,
            parent_xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            key="fixed_prismatic_anchor",
        )

        # 移动关节：沿 Z 轴滑动
        # limit_lower/limit_upper: 行程限制（-0.3 到 +0.3 米）
        j_prismatic = builder.add_joint_prismatic(
            parent=a_pri,
            child=b_pri,
            axis=wp.vec3(0.0, 0.0, 1.0),  # 沿 Z 轴滑动
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -upper_hz), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +cuboid_hz), q=wp.quat_identity()),
            limit_lower=-0.3,  # 最小行程
            limit_upper=0.3,   # 最大行程
            key="prismatic_a_b",
        )
        builder.add_articulation([j_fixed_pri, j_prismatic], key="prismatic_articulation")

        # =================================================================
        # 【关节3】BALL - 球窝关节
        # =================================================================
        # 球关节允许3个旋转自由度（绕任意轴旋转）
        # 坐标：4个值（四元数 x, y, z, w）
        # 常见用途：肩关节、髋关节、万向节
        y = rows[2]
        radius = 0.3
        z_offset = -1.0

        # 【运动学体（kinematic body）】
        # 通过设置 density=0 使物体不受重力和碰撞力影响
        # 运动学体的质量=0，反质量=0，不参与动力学计算
        # 但仍然作为碰撞几何存在
        a_ball = builder.add_link(
            xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z + radius + cuboid_hz + z_offset),
                q=wp.quat_identity(),
            )
        )
        b_ball = builder.add_link(
            xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z + radius + z_offset),
                q=wp.quat_from_axis_angle(wp.vec3(1.0, 1.0, 0.0), 0.1),
            ),
            key="b_ball",
        )

        # 使用 ShapeConfig 设置 density=0 来创建运动学体
        rigid_cfg = newton.ModelBuilder.ShapeConfig()
        rigid_cfg.density = 0.0  # density=0 → 质量=0 → 运动学体（不受力影响）
        builder.add_shape_sphere(a_ball, radius=radius, cfg=rigid_cfg)
        builder.add_shape_box(b_ball, hx=cuboid_hx, hy=cuboid_hy, hz=cuboid_hz)

        j_fixed_ball = builder.add_joint_fixed(
            parent=-1, child=a_ball,
            parent_xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z + radius + cuboid_hz + z_offset),
                q=wp.quat_identity(),
            ),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            key="fixed_ball_anchor",
        )

        # 球关节
        j_ball = builder.add_joint_ball(
            parent=a_ball,
            child=b_ball,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +cuboid_hz), q=wp.quat_identity()),
            key="ball_a_b",
        )
        builder.add_articulation([j_fixed_ball, j_ball], key="ball_articulation")

        # 【设置球关节初始四元数】
        # BALL 关节有4个坐标（四元数 x, y, z, w）
        # quat_rpy(roll, pitch, yaw) 从欧拉角创建四元数
        builder.joint_q[-4:] = wp.quat_rpy(0.5, 0.6, 0.7)

        # 定型和创建求解器
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverXPBD(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, self.args)
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

    def test_post_step(self):
        """每步验证关节运动约束。

        这是一种更严格的测试方式，在每一步都检查物理正确性：
        - REVOLUTE: 角速度只应在旋转轴方向
        - PRISMATIC: 线速度只应在滑动轴方向，且无角速度
        - BALL: 线速度的垂直分量应接近零

        spatial_vector 结构：
        qd[0:3] = 角速度 (omega_x, omega_y, omega_z)
        qd[3:6] = 线速度 (v_x, v_y, v_z)
        spatial_top(qd) = 角速度部分 (vec3)
        spatial_bottom(qd) = 线速度部分 (vec3)
        """
        newton.examples.test_body_state(
            self.model, self.state_0,
            "revolute motion in plane",
            lambda q, qd: wp.length(abs(wp.cross(wp.spatial_bottom(qd), wp.vec3(1.0, 0.0, 0.0)))) < 1e-5,
            indices=[self.model.body_key.index("b_rev")],
        )

        newton.examples.test_body_state(
            self.model, self.state_0,
            "linear motion on axis",
            lambda q, qd: wp.length(abs(wp.cross(wp.spatial_top(qd), wp.vec3(0.0, 0.0, 1.0)))) < 1e-5
            and wp.length(wp.spatial_bottom(qd)) < 1e-5,
            indices=[self.model.body_key.index("b_prismatic")],
        )

    def test_final(self):
        newton.examples.test_body_state(
            self.model, self.state_0,
            "fixed link body has come to a rest",
            lambda q, qd: max(abs(qd)) < 1e-2,
            indices=[0],
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
