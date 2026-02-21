"""
Newton 学习示例 01：双摆仿真 (Basic Pendulum)
==============================================

【学习目标】
- 理解 Newton 仿真的完整流程：Builder → Model → State → Solver
- 学习如何使用 ModelBuilder 构建关节链（articulation）
- 理解双缓冲状态交换模式
- 理解 CUDA 图捕获的性能优化机制

【核心概念】
- ModelBuilder：CPU 端的场景构建器，用 Python 列表存储数据
- Model：finalize() 后生成的 GPU 端静态模型（wp.array）
- State：动态仿真状态（位姿、速度、力）
- Control：控制输入（关节力、目标位置/速度）
- CollisionPipeline：碰撞检测管线（宽相+窄相）
- SolverXPBD：位置级动力学求解器

【运行方式】
    uv run python learning/01_basic_pendulum_annotated.py
    或者运行原版：
    uv run -m newton.examples basic_pendulum
"""

import warp as wp  # Warp：Newton 底层的 GPU 并行计算框架

import newton  # Newton 物理引擎的公共 API
import newton.examples  # Newton 示例基础设施（参数解析、运行循环、查看器创建）


class Example:
    """双摆示例类。

    Newton 的所有示例都遵循相同的结构：
    - __init__(): 构建场景、创建求解器和状态
    - step(): 每帧调用，执行仿真步进
    - render(): 每帧调用，渲染当前状态
    - test_final(): 测试模式下验证仿真正确性
    """

    def __init__(self, viewer, args=None):
        # ===================================================================
        # 【第1部分】仿真时间参数
        # ===================================================================
        # fps: 渲染帧率（每秒多少帧）
        # frame_dt: 每帧的时间步长（秒）
        # sim_substeps: 每帧内的子步数（更多子步 = 更精确但更慢）
        # sim_dt: 每个子步的时间步长
        #
        # 典型设置：fps=100, substeps=10 → sim_dt=0.001秒
        # 更小的 sim_dt 提高稳定性，但消耗更多计算
        self.fps = 100
        self.frame_dt = 1.0 / self.fps  # = 0.01 秒/帧
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps  # = 0.001 秒/子步

        self.viewer = viewer
        self.args = args

        # ===================================================================
        # 【第2部分】场景构建 - 使用 ModelBuilder
        # ===================================================================
        # ModelBuilder 是 Newton 的核心构建器，所有场景都从这里开始
        # 此时数据还在 CPU 端（Python 列表），finalize() 后才转到 GPU
        builder = newton.ModelBuilder()

        # 定义连杆（link）的几何尺寸
        hx = 1.0  # 半长度 X（长度方向）
        hy = 0.1  # 半长度 Y（宽度方向）
        hz = 0.1  # 半长度 Z（高度方向）

        # --- 创建连杆（link = 刚体 + 用于关节链） ---
        # add_link() 创建一个适合加入关节链的刚体
        # 与 add_body() 的区别：add_link() 返回的 body 会被关节约束
        link_0 = builder.add_link()
        # 给连杆0附加一个长方体碰撞形状
        # add_shape_box(body, hx, hy, hz) → 实际尺寸是 2*hx × 2*hy × 2*hz
        builder.add_shape_box(link_0, hx=hx, hy=hy, hz=hz)

        link_1 = builder.add_link()
        builder.add_shape_box(link_1, hx=hx, hy=hy, hz=hz)

        # --- 创建关节 ---
        # 旋转关节（revolute joint）：1个自由度，绕指定轴旋转
        # 就像门铰链——只能绕一个轴旋转

        # 首先创建一个旋转，使摆在 XZ 平面内运动（从侧面看）
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)

        # 【关节0：连接世界到 link_0】
        #
        # parent=-1: 父体是"世界"（固定参考系，不动的"天花板"）
        # child=link_0: 子体是第一根摆杆
        #
        # axis: 旋转轴方向（在关节坐标系中）。这里是Y轴 → 杆在XZ平面摆动
        #
        # parent_xform: 关节在父体（世界）上的安装位置
        #   p=(0,0,5) → 关节挂在世界坐标 z=5 的位置（天花板上的钉子）
        #   q=rot → 附带旋转，让摆从侧面呈现
        #
        # child_xform: 关节在子体(link_0)上的安装位置
        #   p=(-hx,0,0) = (-1,0,0) → 安装在link_0的左端
        #   q=quat_identity() → 不附加额外旋转
        #
        # 物理引擎确保：parent_xform的点 和 child_xform的点 在世界中始终重合
        # → link_0 的左端始终挂在 (0,0,5) → 像钟摆一样摆动
        #
        # 图解:
        #   (0,0,5) ← 天花板钉子 (parent_xform)
        #       │
        #       │ 铰链 (绕Y轴旋转)
        #       │
        #   ┌───●────────────────┐
        #   │x=-1  link_0中心  x=+1│
        #   └────────────────────┘
        #   ↑
        #   child_xform.p=(-1,0,0) "关节接在link_0的左端"
        j0 = builder.add_joint_revolute(
            parent=-1,  # -1 = 世界（固定参考系）
            child=link_0,
            axis=wp.vec3(0.0, 1.0, 0.0),  # 绕 Y 轴旋转
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=rot),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
            # quat_identity() = 不旋转 = "安装时不歪着装"
        )

        # 【关节1：连接 link_0 到 link_1】
        #
        # parent_xform.p=(+hx,0,0) = (+1,0,0) → 安装在 link_0 的右端
        # child_xform.p=(-hx,0,0) = (-1,0,0) → 安装在 link_1 的左端
        # 两个安装点在世界中始终重合 → link_1 挂在 link_0 的右端
        #
        # 图解:
        #   link_0                        link_1
        #   ┌──────────────────●─┐    ┌─●──────────────────┐
        #   │ x=-1   中心   x=+1 │    │ x=-1   中心   x=+1 │
        #   └──────────────────┼─┘    └─┼──────────────────┘
        #                parent_xform  child_xform
        #                 (+1,0,0)     (-1,0,0)
        #                      └──关节j1──┘ 这两点始终重合!
        j1 = builder.add_joint_revolute(
            parent=link_0,
            child=link_1,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        )

        # --- 创建关节链（articulation）---
        # 把多个关节组成一个"关节链"，告诉求解器这些关节构成一个连贯的机构
        # key="pendulum" 是标签名，方便调试查找，不影响仿真
        builder.add_articulation([j0, j1], key="pendulum")

        # --- 添加地面 ---
        # add_ground_plane() 在 z=0 处添加一个无限大的地面
        # 地面的 body=-1（固定到世界），是一个 PLANE 形状
        builder.add_ground_plane()

        # ===================================================================
        # 【第3部分】模型定型 - finalize()
        # ===================================================================
        # finalize() 是关键步骤！它将 CPU 端的 Python 列表转换为 GPU 端的 wp.array
        # 内部做了大量工作：
        #   1. 验证场景结构（关节、形状等）
        #   2. Python 列表 → wp.array（转到GPU）
        #   3. 构建碰撞网格的 SDF
        #   4. 计算刚体惯性
        #   5. 构建关节拓扑和碰撞对
        #   6. 为 Gauss-Seidel 求解器做图着色
        self.model = builder.finalize()

        # ===================================================================
        # 【第4部分】创建求解器
        # ===================================================================
        # SolverXPBD: Extended Position-Based Dynamics
        # - 适合通用刚体仿真
        # - 在位置级别求解约束（而非力级别）
        # - 稳定性好，但不如力级别方法精确
        self.solver = newton.solvers.SolverXPBD(self.model)

        # ===================================================================
        # 【第5部分】创建仿真状态
        # ===================================================================
        # model.state() 从模型的初始条件克隆出一个 State 对象
        # State 包含所有随时间变化的量：
        #   - body_q: 刚体位姿 (wp.transform, 7维: pos + quaternion)
        #   - body_qd: 刚体空间速度 (wp.spatial_vector, 6维: angular + linear)
        #   - body_f: 刚体外力 (wp.spatial_vector, 每步清零)
        #   - joint_q: 广义关节坐标
        #   - joint_qd: 广义关节速度
        #
        # 【双缓冲模式】创建两个 State 用于 ping-pong 交换
        # solver.step(state_0 → state_1) 从 state_0 读取，写入 state_1
        # 下一步交换两者，避免读写同一内存
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Control 存储控制输入（关节力、目标位置/速度等）
        # 这里我们不施加任何控制，让摆自由运动
        self.control = self.model.control()

        # ===================================================================
        # 【第6部分】正运动学初始化
        # ===================================================================
        # eval_fk (Forward Kinematics): 从关节坐标 → 计算刚体在世界中的位姿
        #
        # Newton 内部有两套描述状态的方式：
        #   关节空间: joint_q = [θ₀, θ₁]  (只有2个float，紧凑！)
        #   笛卡尔空间: body_q = [transform₀, transform₁] (每个body的完整位姿)
        #
        # 正运动学就是"翻译"过程：
        #   给定 θ₀=0, θ₁=0 (两个关节角度都是0)
        #   → 算出 link_0 在世界中的位置和朝向 → 写入 body_q[0]
        #   → 算出 link_1 在世界中的位置和朝向 → 写入 body_q[1]
        #
        # 为什么需要手动调用？
        #   XPBD/Featherstone 等求解器直接操作 body_q（笛卡尔空间），
        #   初始化时必须调用 eval_fk 把 joint_q → body_q
        #   之后仿真过程中 solver.step() 会自动维护 body_q
        #
        #   MuJoCo 求解器内部自己做正运动学，所以不需要手动调用
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # ===================================================================
        # 【第7部分】碰撞检测管线
        # ===================================================================
        # CollisionPipeline 负责检测场景中的碰撞
        # 包含两个阶段：
        #   1. 宽相（Broad Phase）：用 AABB 快速剔除不可能碰撞的物体对
        #      - NXN: O(N²) 全对检测（简单但慢）
        #      - SAP: 扫描排序剪枝（适合大规模场景）
        #      - EXPLICIT: 预计算碰撞对（默认，最快）
        #   2. 窄相（Narrow Phase）：精确计算碰撞点、法线、穿透深度
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, self.args)
        # contacts() 分配碰撞结果的缓冲区
        self.contacts = self.collision_pipeline.contacts()

        # 将模型传给查看器用于渲染
        self.viewer.set_model(self.model)

        # 捕获 CUDA 图用于性能优化
        self.capture()

    def capture(self):
        """CUDA 图捕获 - 性能优化的关键。

        CUDA 图将整个仿真循环记录为一个 GPU 操作序列，
        之后可以"重放"这个序列，完全跳过 Python 解释器的开销。
        这对于每帧有大量小 kernel 的仿真特别有效。

        工作原理：
        1. wp.ScopedCapture() 开始录制
        2. 执行一次 simulate()，所有 GPU 操作被记录
        3. 录制结束，得到 capture.graph
        4. 之后每帧调用 wp.capture_launch(graph) 重放

        注意：CUDA 图只适用于 GPU 设备，CPU 模式下直接执行 Python 代码
        """
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()  # 录制一次完整的仿真循环
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        """执行一帧的物理仿真（包含多个子步）。

        这是 Newton 仿真的核心循环，每个子步包含：
        1. clear_forces(): 清零外力累积器
        2. apply_forces(): 应用查看器施加的力（如鼠标拖拽）
        3. collide(): 检测碰撞，填充 contacts 缓冲区
        4. step(): 求解器积分一个时间步
        5. swap(): 交换状态缓冲区

        子步数越多，仿真越稳定但越慢。
        典型设置：frame_dt=0.01, substeps=10 → sim_dt=0.001
        """
        for _ in range(self.sim_substeps):
            # 清零 state_0 中的力累积器 (body_f 和 particle_f)
            # 每个子步开始时必须清零，否则力会累积
            self.state_0.clear_forces()

            # 查看器可以通过鼠标拖拽向物体施加力
            self.viewer.apply_forces(self.state_0)

            # 碰撞检测：更新 self.contacts 中的碰撞信息
            # 输入：当前状态 state_0（用于获取刚体位姿）
            # 输出：contacts 中填充碰撞点、法线、穿透深度等
            self.collision_pipeline.collide(self.state_0, self.contacts)

            # 物理求解器积分：从 state_0 计算 state_1
            # 输入：当前状态、控制输入、碰撞信息、时间步长
            # 输出：下一时刻的状态写入 state_1
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # 【关键】交换状态缓冲区
            # 下一个子步中，state_1 变成输入，state_0 变成输出
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """每帧调用一次，由 newton.examples.run() 驱动。

        如果有 CUDA 图，直接重放（几乎零 Python 开销）；
        否则执行 Python 版仿真循环。
        """
        if self.graph:
            wp.capture_launch(self.graph)  # 重放 CUDA 图
        else:
            self.simulate()  # 直接执行 Python 版

        self.sim_time += self.frame_dt

    def test_final(self):
        """仿真结束后的验证测试。

        使用 test_body_state() 对指定刚体评估 lambda 谓词函数。
        如果任何刚体不满足条件，抛出 ValueError。

        test_body_state(model, state, test_name, test_fn, indices)
        - test_fn(q, qd) → bool，其中 q=位姿(transform), qd=速度(spatial_vector)
        - q[0..2] 是位置 (x,y,z)，q[3..6] 是四元数
        - qd[0..2] 是角速度，qd[3..5] 是线速度
        """
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pendulum links in correct area",
            lambda q, qd: abs(q[0]) < 1e-5 and abs(q[1]) < 1.0 and q[2] < 5.0 and q[2] > 0.0,
            [0, 1],  # 测试 link_0 和 link_1
        )

        def check_velocities(_, qd):
            check = abs(qd[0]) < 1e-4 and abs(qd[6]) < 1e-4
            check = check and abs(qd[1]) < 10.0 and abs(qd[2]) < 5.0 and abs(qd[3]) < 10.0 and abs(qd[4]) < 10.0
            return check

        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pendulum links have reasonable velocities",
            check_velocities,
            [0, 1],
        )

    def render(self):
        """每帧渲染当前仿真状态。

        查看器API模式：
        1. begin_frame(time): 开始一帧
        2. log_state(state): 渲染所有刚体和粒子
        3. log_contacts(contacts, state): 渲染碰撞点（可选）
        4. end_frame(): 结束一帧
        """
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Newton 示例的标准入口模式：
    # 1. 创建参数解析器（含通用参数：--device, --viewer, --num-frames 等）
    # 2. init() 解析参数并创建对应的 Viewer
    # 3. 创建 Example 实例
    # 4. run() 驱动主循环：while viewer.is_running(): step() + render()
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
