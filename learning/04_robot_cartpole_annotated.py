"""
Newton 学习示例 04：机器人仿真 - Cartpole (Robot Cartpole)
==========================================================

【学习目标】
- 学习从 USD 文件导入关节机器人
- 理解 MuJoCo 求解器的使用方式
- 学习多世界复制（replicate）用于批量仿真
- 理解 MuJoCo 求解器不需要碰撞管线的场景
- 学习 register_custom_attributes 的用途

【MuJoCo 求解器 vs 其他求解器】

  MuJoCo 求解器源码: newton/_src/solvers/mujoco/solver_mujoco.py (~2000行)
  它是 Newton 和 Google DeepMind 的 mujoco-warp 库之间的桥梁。

  step() 的内部流程 (源码第2032行):
    1. apply_mjc_control()    → Newton 的 control → MuJoCo 的 ctrl
    2. update_mjc_data()      → Newton 的 state → MuJoCo 的 qpos/qvel
    3. mujoco_warp_step()     → 调用 mujoco-warp 执行物理计算
    4. update_newton_state()  → MuJoCo 的结果 → Newton 的 state_out

  MuJoCo 特点:
  - 自带接触处理（不需要 Newton 的 CollisionPipeline）
  - 内部自动做正运动学（从 joint_q → body_q）
  - 适合关节型机器人（机械臂、人形机器人等）
  - 需要 register_custom_attributes() 注册自定义属性

  其他求解器（XPBD, Featherstone 等）:
  - 需要 Newton 的 CollisionPipeline 做碰撞检测
  - 需要手动调用 eval_fk() 做正运动学
  - XPBD 适合通用刚体，Featherstone 适合关节动力学

【多世界复制】
replicate() 可以将一个 ModelBuilder 复制 N 次，
每个副本是一个独立的"世界"（world），
物理上互不影响，但在同一个 GPU 上批量计算。
这是强化学习中常用的并行环境模式。

【运行方式】
    uv run python learning/04_robot_cartpole_annotated.py
    uv run python learning/04_robot_cartpole_annotated.py --num-worlds 100
"""

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, num_worlds=8):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_worlds = num_worlds
        self.viewer = viewer

        # ===================================================================
        # 【第1步】构建单个 Cartpole
        # ===================================================================
        # 创建一个"模板"builder，描述单个 cartpole
        cartpole = newton.ModelBuilder()

        # 【重要】MuJoCo 求解器需要注册自定义属性
        # 源码位置: newton/_src/solvers/mujoco/solver_mujoco.py 第278行
        #
        # 这必须在 add_usd()/add_mjcf() 之前调用！
        # 原因: add_usd() 解析 USD 时遇到 "mjc:condim=3" 这样的属性，
        #       需要有地方存放。如果没提前注册 → 属性被忽略 → MuJoCo 用错误默认值
        #
        # 注册了哪些属性（都存在 model.mujoco.xxx 命名空间下）:
        #   每个 shape: condim(接触维度), solimp(接触刚度), solmix(材料混合), gap(间隙)
        #   每个 joint: stiffness(弹簧), damping(阻尼), solref/solimp(限位参数)
        #   控制输入:   ctrl(MuJoCo格式的控制数组)
        #
        # 不使用 MuJoCo 求解器 → 不需要调这行
        newton.solvers.SolverMuJoCo.register_custom_attributes(cartpole)

        # 设置默认参数（应用于后续所有添加的形状/关节/刚体）
        # 源码: builder.py 第562行 → self.default_shape_cfg = ShapeConfig()
        #
        # default_* 设置全局默认值，之后所有 add_shape/add_joint 自动使用
        # 如果某个形状/关节需要不同的值，可以单独覆盖：
        #   方式1: 传参覆盖 builder.add_shape_sphere(body, radius=0.3, density=5000.0)
        #   方式2: 用自定义 cfg
        #          special = newton.ModelBuilder.ShapeConfig(density=5000.0)
        #          builder.add_shape_box(body, hx=0.5, cfg=special)
        cartpole.default_shape_cfg.density = 100.0  # 形状密度(kg/m³)
        cartpole.default_joint_cfg.armature = 0.1    # 关节阻尼（抑制高频振动）
        cartpole.default_body_armature = 0.1         # 刚体阻尼

        # ===================================================================
        # 【第2步】从 USD 导入关节机器人
        # ===================================================================
        # add_usd() 解析 USD 文件中的刚体、关节、形状定义
        # 源码: newton/_src/utils/import_usd.py
        #
        # 它自动完成以下工作：
        #   1. 解析 USD 中的 body/joint/shape 定义
        #   2. 调用 add_body(), add_joint_*(), add_shape_*()
        #   3. 自动调用 _finalize_imported_articulation() 创建 articulation
        #      → 你不需要手动调 add_articulation()！
        #   4. 读取 MuJoCo 自定义属性（如 mjc:condim）存入注册的属性
        #
        # enable_self_collisions: 是否启用自碰撞（关节链内部碰撞）
        # collapse_fixed_joints: 将固定关节的刚体合并为一个
        #   → 减少刚体数量，提高性能
        cartpole.add_usd(
            newton.examples.get_asset("cartpole.usda"),
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )

        # 设置初始关节位置
        # cartpole.usda 定义了3个关节 DOF
        # joint_q[-3:] 设置最后3个关节坐标
        cartpole.joint_q[-3:] = [0.0, 0.3, 0.0]

        # ===================================================================
        # 【第3步】多世界复制
        # ===================================================================
        # 创建一个新的场景 builder，将 cartpole 复制 N 次
        # 每个副本是一个独立的"世界"，编号 0 到 num_worlds-1
        # spacing: 相邻世界之间的间距 (x, y, z)
        builder = newton.ModelBuilder()
        builder.replicate(cartpole, self.num_worlds, spacing=(1.0, 2.0, 0.0))

        self.model = builder.finalize()

        # ===================================================================
        # 【第4步】创建 MuJoCo 求解器
        # ===================================================================
        # SolverMuJoCo 不需要额外参数即可工作
        # 但可以指定 iterations, ls_iterations, njmax, nconmax 等
        #
        # 【重要】SolverMuJoCo 是 Newton 和 mujoco-warp 之间的"翻译官"
        #
        #   mujoco-warp：Google DeepMind 的开源库（Apache-2.0）
        #   仓库：https://github.com/google-deepmind/mujoco_warp
        #   安装：pip install mujoco-warp（不在 Newton 仓库里，是独立的第三方包）
        #   作用：真正的物理计算引擎（接触求解、关节动力学等核心算法）
        #
        #   SolverMuJoCo（在 Newton 里）的工作：
        #     Newton的State/Control → 转换 → MuJoCo格式 → 调用mujoco-warp → 转换回 → Newton的State
        #     它自己不做物理计算，只做数据格式的"翻译"
        #
        #   源码: newton/_src/solvers/mujoco/solver_mujoco.py (~2000行，大部分是数据转换代码)
        #   mujoco-warp 源码: https://github.com/google-deepmind/mujoco_warp
        self.solver = newton.solvers.SolverMuJoCo(self.model)

        # 也可以尝试其他求解器（取消注释测试）：
        # self.solver = newton.solvers.SolverSemiImplicit(
        #     self.model, joint_attach_ke=1600.0, joint_attach_kd=20.0
        # )
        # self.solver = newton.solvers.SolverFeatherstone(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # 【注意】MuJoCo 求解器可以不使用 Newton 碰撞管线
        # cartpole 场景没有碰撞需求，所以 contacts=None
        self.contacts = None

        # 正运动学：从 joint_q → 计算 body_q (每个body在世界中的位姿)
        #
        # MuJoCo 可以自己做 FK 吗？ → 可以！
        # MuJoCo 的 step() 内部会调用 update_newton_state(eval_fk=True)
        # 自动从 joint_q 算出 body_q (源码第2033行: eval_fk = not self._has_ref)
        #
        # 那为什么这里还要手动调？
        # 因为 step() 还没被调用。如果不调 eval_fk：
        #   state_0.body_q 全是初始值（可能不正确）
        #   → 第一帧的 render() 画出来的位置不对
        #   → 第一帧的 viewer.apply_forces() 用错误的位姿计算力
        #
        # 总结：
        #   XPBD/Featherstone → 必须调 eval_fk（它们直接操作body_q）
        #   MuJoCo → 非必须，但推荐调（确保第一帧正确）
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        """标准仿真循环，但没有碰撞检测步骤。

        注意与前面示例的区别：
        - 没有 collision_pipeline.collide()
        - contacts=None 传给 solver.step()
        - MuJoCo 求解器如果需要碰撞，会在内部处理
        """
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            # contacts=None：不使用碰撞检测
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """验证多世界一致性。

        因为所有世界的初始条件相同，结果应该完全一致。
        这是多世界仿真的一个重要正确性验证。
        """
        num_bodies_per_world = self.model.body_count // self.num_worlds

        # 检查所有世界的 cart（滑块）行为一致
        newton.examples.test_body_state(
            self.model, self.state_0,
            "cart is at ground level and has correct orientation",
            lambda q, qd: q[2] == 0.0 and newton.math.vec_allclose(q.q, wp.quat_identity()),
            indices=[i * num_bodies_per_world for i in range(self.num_worlds)],
        )

        # 验证速度跨世界一致
        qd = self.state_0.body_qd.numpy()
        world0_cart_vel = wp.spatial_vector(*qd[0])
        newton.examples.test_body_state(
            self.model, self.state_0,
            "cart velocities match across worlds",
            lambda q, qd: newton.math.vec_allclose(qd, world0_cart_vel),
            indices=[i * num_bodies_per_world for i in range(self.num_worlds)],
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=100, help="Total number of simulated worlds.")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args.num_worlds)
    newton.examples.run(example, args)
