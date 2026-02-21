"""
Newton 学习示例 05：可微仿真 (Differentiable Simulation)
========================================================

【学习目标】
- 理解可微物理仿真的核心概念
- 学习 Warp 的 Tape（自动微分磁带）机制
- 理解为什么需要 requires_grad=True
- 学习粒子系统（particle）仿真
- 理解基于梯度的轨迹优化

【可微仿真原理】
传统仿真：给定初始条件 → 运行物理 → 得到最终状态
可微仿真：给定初始条件 → 运行物理 → 计算损失 → 反向传播梯度 → 优化初始条件

这个例子的目标：
- 一个粒子从初始位置以某速度出发
- 碰到墙壁和地面后弹跳
- 目标：优化初始速度，使粒子最终落在目标位置

工作流程：
1. 前向传播：运行仿真，计算 loss = ||最终位置 - 目标||²
2. 反向传播：通过 Warp Tape 计算 ∂loss/∂初始速度
3. 梯度下降：更新初始速度
4. 重复直到收敛

【Warp Tape 机制】
wp.Tape() 类似 PyTorch 的 autograd：
- 记录所有 wp.launch() 操作
- tape.backward(loss) 自动执行反向传播
- 每个有 requires_grad=True 的 wp.array 都有 .grad 属性

【与非微仿真的关键区别】
1. finalize(requires_grad=True) - 启用梯度追踪
2. 每个时间步一个独立的 State - 不交换、不覆盖
3. 使用 SolverSemiImplicit - 唯一支持反向传播的求解器
4. CollisionPipeline(requires_grad=True) - 碰撞检测也可微

【运行方式】
    uv run python learning/05_diffsim_ball_annotated.py
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.tests.unittest_utils import assert_np_equal
from newton.utils import bourke_color_map


# ===================================================================
# 【Warp 核函数】
# ===================================================================
# @wp.kernel 定义 GPU 并行核函数
# 这些函数在 GPU 上并行执行，每个线程处理一个元素

@wp.kernel
def loss_kernel(pos: wp.array(dtype=wp.vec3), target: wp.vec3, loss: wp.array(dtype=float)):
    """计算粒子到目标的距离平方作为损失。

    这个核函数在1个线程上执行（dim=1）
    loss = ||pos[0] - target||²

    注意：核函数不能 return 值，必须写入输出数组
    """
    delta = pos[0] - target
    loss[0] = wp.dot(delta, delta)  # 点积 = 距离平方


@wp.kernel
def step_kernel(x: wp.array(dtype=wp.vec3), grad: wp.array(dtype=wp.vec3), alpha: float):
    """梯度下降更新步。

    x[i] = x[i] - alpha * grad[i]

    wp.tid() 返回当前线程的索引
    """
    tid = wp.tid()
    x[tid] = x[tid] - grad[tid] * alpha


class Example:
    def __init__(self, viewer, args=None, verbose=False):
        self.fps = 60
        self.frame = 0
        self.frame_dt = 1.0 / self.fps
        self.sim_steps = 36          # 总仿真步数
        self.sim_substeps = 8        # 每步内子步数
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.verbose = verbose

        # 优化参数
        self.train_iter = 0         # 当前训练迭代次数
        self.train_rate = 0.02      # 学习率（梯度下降步长）
        self.target = (0.0, -2.0, 1.5)  # 目标位置

        # 损失值（标量数组，requires_grad=True 才能被反向传播）
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.loss_history = []

        self.viewer = viewer
        self.viewer.show_particles = True  # 在查看器中显示粒子

        # ===================================================================
        # 【场景构建】粒子 + 墙壁 + 地面
        # ===================================================================
        scene = newton.ModelBuilder(up_axis=newton.Axis.Z)

        # 添加一个粒子
        # 粒子（particle）vs 刚体（body）：
        #   - 粒子：只有位置和速度，没有旋转，用 vec3 表示
        #   - 刚体：有位姿（位置+旋转），用 transform 表示
        # 粒子更轻量，适合大规模模拟（布料、流体）
        scene.add_particle(
            pos=wp.vec3(0.0, -0.5, 1.0),   # 初始位置
            vel=wp.vec3(0.0, 5.0, -5.0),    # 初始速度（这就是我们要优化的！）
            mass=1.0,
        )

        # 添加墙壁和地面（作为固定形状）
        # body=-1 表示这些形状固定在世界中（不参与动力学）
        ke = 1.0e4   # 接触弹簧刚度
        kf = 0.0     # 接触摩擦刚度
        kd = 1.0e1   # 接触阻尼
        mu = 0.2     # 摩擦系数

        # 墙壁：一个 box 形状，body=-1（固定到世界）
        scene.add_shape_box(
            body=-1,  # -1 = 世界固定体
            xform=wp.transform(wp.vec3(0.0, 2.0, 1.0), wp.quat_identity()),
            hx=1.0, hy=0.25, hz=1.0,
            cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu),
        )

        # 地面
        scene.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu),
        )

        # ===================================================================
        # 【关键】finalize(requires_grad=True)
        # ===================================================================
        # requires_grad=True 使所有 State 中的数组支持梯度追踪
        # 这意味着 state.particle_q, state.particle_qd 等都有 .grad 属性
        self.model = scene.finalize(requires_grad=True)

        # 设置软接触参数（粒子与形状之间的接触力）
        self.model.soft_contact_ke = ke
        self.model.soft_contact_kf = kf
        self.model.soft_contact_kd = kd
        self.model.soft_contact_mu = mu
        self.model.soft_contact_restitution = 1.0  # 弹性碰撞系数

        # ===================================================================
        # 【关键】使用 SolverSemiImplicit
        # ===================================================================
        # SolverSemiImplicit 是唯一完全支持可微仿真的求解器
        # 它使用半隐式欧拉积分：
        #   v_{n+1} = v_n + F/m * dt
        #   x_{n+1} = x_n + v_{n+1} * dt
        self.solver = newton.solvers.SolverSemiImplicit(self.model)

        # ===================================================================
        # 【关键】每个时间步一个独立的 State
        # ===================================================================
        # 与普通仿真不同（只用2个State做双缓冲），
        # 可微仿真需要保留所有中间状态用于反向传播
        # 总状态数 = sim_steps * sim_substeps + 1（初始状态）
        self.states = [self.model.state() for _ in range(self.sim_steps * self.sim_substeps + 1)]
        self.control = self.model.control()

        # 碰撞检测也需要 requires_grad=True
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase_mode=newton.BroadPhaseMode.EXPLICIT,
            soft_contact_margin=10.0,  # 软接触的搜索范围
            requires_grad=True,
        )
        self.contacts = self.collision_pipeline.contacts()
        # 对于简单场景（固定墙壁+地面），碰撞只需检测一次
        self.collision_pipeline.collide(self.states[0], self.contacts)

        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        """捕获前向+反向传播为 CUDA 图。"""
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.forward_backward()
            self.graph = capture.graph
        else:
            self.graph = None

    def forward_backward(self):
        """执行一次完整的前向传播 + 反向传播。

        Warp Tape 的工作流程：
        1. 创建 Tape 对象
        2. 在 with tape: 块内执行前向计算
        3. tape.backward(loss) 执行反向传播
        4. 每个 requires_grad 的数组的 .grad 被填充
        """
        self.tape = wp.Tape()
        with self.tape:
            self.forward()       # 前向传播：运行仿真 + 计算损失
        self.tape.backward(self.loss)  # 反向传播：计算所有梯度

    def forward(self):
        """前向传播：运行整个仿真轨迹，计算最终损失。"""
        for sim_step in range(self.sim_steps):
            self.simulate(sim_step)

        # 在最终状态上计算损失
        # loss = ||final_position - target||²
        wp.launch(
            loss_kernel, dim=1,
            inputs=[self.states[-1].particle_q, self.target, self.loss],
        )
        return self.loss

    def simulate(self, sim_step):
        """仿真一个时间步（包含多个子步）。

        【关键区别】这里没有 swap！
        每个子步从 states[t] 读取，写入 states[t+1]
        这样所有中间状态都保留，反向传播时可以逐步回溯
        """
        for i in range(self.sim_substeps):
            t = sim_step * self.sim_substeps + i
            self.states[t].clear_forces()
            # 注意：states[t] → states[t+1]，不做交换
            self.solver.step(self.states[t], self.states[t + 1], self.control, self.contacts, self.sim_dt)

    def step(self):
        """每帧执行：前向+反向+梯度更新。"""
        # 执行前向和反向传播
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.forward_backward()

        # 获取初始速度及其梯度
        x = self.states[0].particle_qd  # 初始速度（要优化的参数）

        if self.verbose:
            print(f"Train iter: {self.train_iter} Loss: {self.loss}")
            print(f"    x: {x} g: {x.grad}")

        # ===================================================================
        # 【梯度下降更新】
        # ===================================================================
        # x.grad 由 tape.backward() 填充
        # 更新规则：x_new = x_old - learning_rate * gradient
        wp.launch(step_kernel, dim=len(x), inputs=[x, x.grad, self.train_rate])

        # 清零梯度，为下一次迭代做准备
        self.tape.zero()

        self.train_iter += 1
        self.loss_history.append(self.loss.numpy()[0])

    def test_final(self):
        """验证解析梯度与数值梯度的一致性。

        这是可微仿真的标准验证方法：
        - 数值梯度：用有限差分法 (f(x+ε) - f(x-ε)) / (2ε) 近似
        - 解析梯度：用 Tape 反向传播计算
        - 两者应该接近（容差 tol=5e-2）
        """
        x_grad_numeric, x_grad_analytic = self.check_grad()
        assert_np_equal(x_grad_numeric, x_grad_analytic, tol=5e-2)
        assert all(np.array(self.loss_history) < 10.0)
        assert all(np.diff(self.loss_history[:-1]) < -1e-3)

    def render(self):
        """渲染轨迹优化过程。每16次迭代渲染一次以节省性能。"""
        if self.frame > 0 and self.train_iter % 16 != 0:
            return

        traj_verts = [self.states[0].particle_q.numpy()[0].tolist()]

        for i in range(self.sim_steps + 1):
            state = self.states[i * self.sim_substeps]
            traj_verts.append(state.particle_q.numpy()[0].tolist())

            self.viewer.begin_frame(self.frame * self.frame_dt)
            self.viewer.log_scalar("/loss", self.loss.numpy()[0])
            self.viewer.log_state(state)
            self.viewer.log_contacts(self.contacts, state)
            # 渲染目标位置（小方块）
            self.viewer.log_shapes(
                "/target",
                newton.GeoType.BOX,
                (0.1, 0.1, 0.1),
                wp.array([wp.transform(self.target, wp.quat_identity())], dtype=wp.transform),
                wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3),
            )
            # 渲染轨迹线（颜色随损失值变化）
            self.viewer.log_lines(
                f"/traj_{self.train_iter - 1}",
                wp.array(traj_verts[0:-1], dtype=wp.vec3),
                wp.array(traj_verts[1:], dtype=wp.vec3),
                bourke_color_map(0.0, 7.0, self.loss.numpy()[0]),
            )
            self.viewer.end_frame()
            self.frame += 1

    def check_grad(self):
        """数值梯度 vs 解析梯度对比。

        数值梯度（有限差分法）：
        ∂loss/∂x_i ≈ (loss(x + ε*e_i) - loss(x - ε*e_i)) / (2ε)

        解析梯度（Tape 反向传播）：
        ∂loss/∂x_i = x.grad[i]（由 tape.backward() 自动计算）
        """
        param = self.states[0].particle_qd
        x_c = param.numpy().flatten()
        x_grad_numeric = np.zeros_like(x_c)

        for i in range(len(x_c)):
            eps = 1.0e-3
            step = np.zeros_like(x_c)
            step[i] = eps

            param.assign(x_c + step)
            l_1 = self.forward().numpy()[0]

            param.assign(x_c - step)
            l_0 = self.forward().numpy()[0]

            x_grad_numeric[i] = (l_1 - l_0) / (eps * 2.0)

        param.assign(x_c)

        tape = wp.Tape()
        with tape:
            l = self.forward()
        tape.backward(l)
        x_grad_analytic = param.grad.numpy()[0].copy()

        print(f"numeric grad: {x_grad_numeric}")
        print(f"analytic grad: {x_grad_analytic}")
        tape.zero()

        return x_grad_numeric, x_grad_analytic


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages.")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args=args, verbose=args.verbose)
    example.check_grad()
    newton.examples.run(example, args)
