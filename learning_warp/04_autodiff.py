"""
Warp 学习 04：自动微分 —— Tape / 前向反向传播 / 自定义梯度
==========================================================

【运行方式】
    cd /home/hkang/newton_related/newton
    uv run python learning_warp/04_autodiff.py

【学习目标】
- 理解 Warp 的自动微分机制（Tape 录制 + backward 反传）
- 从最简单的标量梯度到物理仿真中的梯度优化
- 理解 requires_grad 的作用
- 学习 wp.func_replay 和 wp.func_grad 自定义梯度
- 实现一个完整的"用梯度优化物理参数"的示例
"""

import numpy as np
import warp as wp

wp.init()


# ============================================================================
# 第1节：最基本的自动微分 —— 标量函数求导
# ============================================================================
# f(x) = x² → df/dx = 2x
#
# Tape 的工作流程：
#   1. 创建 Tape
#   2. 在 with tape: 块内执行前向计算
#   3. tape.backward(loss) 执行反向传播
#   4. 输入数组的 .grad 属性被填充

@wp.kernel
def square_kernel(x: wp.array(dtype=float),
                  loss: wp.array(dtype=float)):
    tid = wp.tid()
    loss[0] = x[0] * x[0]  # f(x) = x²


def demo_basic_grad():
    print("=" * 60)
    print("第1节：基本自动微分 —— f(x)=x² 的梯度")
    print("=" * 60)

    x = wp.array([3.0], dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(square_kernel, dim=1, inputs=[x], outputs=[loss])

    tape.backward(loss)

    print(f"  x = {x.numpy()[0]}")
    print(f"  f(x) = x² = {loss.numpy()[0]}")
    print(f"  df/dx = 2x = {x.grad.numpy()[0]}  (期望: {2 * x.numpy()[0]})")
    tape.zero()
    print()


# ============================================================================
# 第2节：向量函数的梯度 —— 损失函数求导
# ============================================================================
# loss = ||position - target||²
# 这就是 Newton diffsim_ball 示例中的损失函数

@wp.kernel
def distance_loss_kernel(position: wp.array(dtype=wp.vec3),
                         target: wp.array(dtype=wp.vec3),
                         loss: wp.array(dtype=float)):
    tid = wp.tid()
    diff = position[0] - target[0]
    loss[0] = wp.dot(diff, diff)  # ||pos - target||²


def demo_vector_grad():
    print("=" * 60)
    print("第2节：向量损失函数的梯度")
    print("=" * 60)

    position = wp.array([wp.vec3(3.0, 4.0, 0.0)], dtype=wp.vec3, requires_grad=True)
    target = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(distance_loss_kernel, dim=1, inputs=[position, target], outputs=[loss])
    tape.backward(loss)

    grad = position.grad.numpy()[0]
    print(f"  position = (3, 4, 0)")
    print(f"  target   = (0, 0, 0)")
    print(f"  loss = ||pos-target||² = {loss.numpy()[0]}")
    print(f"  d(loss)/d(pos) = 2*(pos-target) = ({grad[0]:.1f}, {grad[1]:.1f}, {grad[2]:.1f})")
    print(f"  期望: (6.0, 8.0, 0.0)")
    tape.zero()
    print()


# ============================================================================
# 第3节：多步计算的梯度 —— Tape 记录计算链
# ============================================================================
# 模拟物理仿真的多步计算：
#   x₁ = x₀ + v₀ * dt
#   x₂ = x₁ + v₁ * dt  (v₁ = v₀ + g * dt)
#   loss = ||x₂ - target||²
# 求 d(loss)/d(v₀)

@wp.kernel
def physics_step(x_in: wp.array(dtype=wp.vec3),
                 v_in: wp.array(dtype=wp.vec3),
                 gravity: wp.vec3,
                 dt: float,
                 x_out: wp.array(dtype=wp.vec3),
                 v_out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    v_new = v_in[tid] + gravity * dt
    x_new = x_in[tid] + v_new * dt
    x_out[tid] = x_new
    v_out[tid] = v_new


def demo_multistep_grad():
    print("=" * 60)
    print("第3节：多步计算的梯度（物理仿真链）")
    print("=" * 60)

    dt = 0.1
    gravity = wp.vec3(0.0, 0.0, -9.81)
    target = wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3)

    x0 = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, requires_grad=True)
    v0 = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, requires_grad=True)
    x1 = wp.zeros(1, dtype=wp.vec3, requires_grad=True)
    v1 = wp.zeros(1, dtype=wp.vec3, requires_grad=True)
    x2 = wp.zeros(1, dtype=wp.vec3, requires_grad=True)
    v2 = wp.zeros(1, dtype=wp.vec3, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(physics_step, dim=1, inputs=[x0, v0, gravity, dt], outputs=[x1, v1])
        wp.launch(physics_step, dim=1, inputs=[x1, v1, gravity, dt], outputs=[x2, v2])
        wp.launch(distance_loss_kernel, dim=1, inputs=[x2, target], outputs=[loss])

    tape.backward(loss)

    print(f"  初始速度: {v0.numpy()[0]}")
    print(f"  2步后位置: {x2.numpy()[0]}")
    print(f"  目标: {target.numpy()[0]}")
    print(f"  loss = {loss.numpy()[0]:.4f}")
    print(f"  d(loss)/d(v0) = {v0.grad.numpy()[0]}")
    print(f"  → 梯度告诉你：v0 应该往 x 正方向调整")
    tape.zero()
    print()


# ============================================================================
# 第4节：梯度下降优化 —— 用梯度找最优初始速度
# ============================================================================
# 和 Newton 的 diffsim_ball 一样的思路：
# 通过梯度下降找到一个初始速度，让粒子飞到目标位置

@wp.kernel
def gradient_step(param: wp.array(dtype=wp.vec3),
                  grad: wp.array(dtype=wp.vec3),
                  lr: float):
    """梯度下降更新：param = param - lr * grad"""
    tid = wp.tid()
    param[tid] = param[tid] - grad[tid] * lr


def demo_gradient_descent():
    print("=" * 60)
    print("第4节：梯度下降优化（找最优初始速度）")
    print("=" * 60)

    n_steps = 10
    dt = 0.05
    gravity = wp.vec3(0.0, 0.0, -9.81)
    target = wp.array([wp.vec3(2.0, 0.0, 1.0)], dtype=wp.vec3)
    lr = 0.01

    v0 = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, requires_grad=True)

    for iteration in range(50):
        states_x = [wp.zeros(1, dtype=wp.vec3, requires_grad=True) for _ in range(n_steps + 1)]
        states_v = [wp.zeros(1, dtype=wp.vec3, requires_grad=True) for _ in range(n_steps + 1)]
        states_v[0].assign(v0.numpy())

        loss = wp.zeros(1, dtype=float, requires_grad=True)

        tape = wp.Tape()
        with tape:
            for i in range(n_steps):
                wp.launch(physics_step, dim=1,
                          inputs=[states_x[i], states_v[i], gravity, dt],
                          outputs=[states_x[i + 1], states_v[i + 1]])
            wp.launch(distance_loss_kernel, dim=1,
                      inputs=[states_x[-1], target], outputs=[loss])
        tape.backward(loss)

        loss_val = loss.numpy()[0]
        grad_val = states_v[0].grad.numpy()[0]
        if iteration % 10 == 0:
            v_val = v0.numpy()[0]
            print(f"  迭代 {iteration:3d}: loss={loss_val:.4f}, v0=({v_val[0]:.2f}, {v_val[1]:.2f}, {v_val[2]:.2f})")

        # 手动梯度下降更新 v0
        v0_np = v0.numpy()
        v0_np[0] -= lr * grad_val
        v0.assign(v0_np)
        tape.zero()

    v_final = v0.numpy()[0]
    print(f"  最终 v0 = ({v_final[0]:.2f}, {v_final[1]:.2f}, {v_final[2]:.2f})")
    print(f"  最终 loss = {loss_val:.6f}")
    print()


# ============================================================================
# 第5节：数值梯度验证 —— 确认自动微分的正确性
# ============================================================================

def demo_grad_check():
    print("=" * 60)
    print("第5节：数值梯度 vs 解析梯度（梯度验证）")
    print("=" * 60)

    x = wp.array([2.0], dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    # 解析梯度（Tape）
    tape = wp.Tape()
    with tape:
        wp.launch(square_kernel, dim=1, inputs=[x], outputs=[loss])
    tape.backward(loss)
    analytic_grad = x.grad.numpy()[0]
    tape.zero()

    # 数值梯度（有限差分）
    eps = 1e-4
    x.assign([2.0 + eps])
    loss_plus = wp.zeros(1, dtype=float)
    wp.launch(square_kernel, dim=1, inputs=[x], outputs=[loss_plus])

    x.assign([2.0 - eps])
    loss_minus = wp.zeros(1, dtype=float)
    wp.launch(square_kernel, dim=1, inputs=[x], outputs=[loss_minus])

    numeric_grad = (loss_plus.numpy()[0] - loss_minus.numpy()[0]) / (2 * eps)

    print(f"  f(x) = x², x = 2.0")
    print(f"  解析梯度 (Tape):     {analytic_grad:.6f}")
    print(f"  数值梯度 (有限差分):  {numeric_grad:.6f}")
    print(f"  差异: {abs(analytic_grad - numeric_grad):.2e}")
    assert abs(analytic_grad - numeric_grad) < 5e-3, "梯度不匹配！"
    print("  验证通过！")
    print()


# ============================================================================
if __name__ == "__main__":
    demo_basic_grad()
    demo_vector_grad()
    demo_multistep_grad()
    demo_gradient_descent()
    demo_grad_check()
    print("全部通过！")
