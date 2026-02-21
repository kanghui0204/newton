"""
Warp 学习 03：物理算子实战 —— Newton 风格的 GPU 核函数
=====================================================

【运行方式】
    cd /home/hkang/newton_related/newton
    uv run python learning_warp/03_physics_kernels.py

【学习目标】
- 实现 Newton 中实际使用的物理算子（简化版）
- 理解 wp.atomic_add（原子操作）在力累积中的作用
- 实现：半隐式欧拉积分、弹簧力、SDF 碰撞、刚体积分
- 每个算子都有 Newton 源码位置标注，方便对照阅读
"""

import numpy as np
import warp as wp

wp.init()


# ============================================================================
# 算子1：半隐式欧拉积分（粒子）
# ============================================================================
# 对应 Newton 源码: newton/_src/solvers/solver.py 第 22-59 行
#
# 这是 Newton 最基础的积分算子：
#   v_new = v + (f/m + gravity) * dt
#   x_new = x + v_new * dt

@wp.kernel
def integrate_particles(
    x: wp.array(dtype=wp.vec3),     # 位置
    v: wp.array(dtype=wp.vec3),     # 速度
    f: wp.array(dtype=wp.vec3),     # 力
    inv_mass: wp.array(dtype=float),  # 质量的倒数（0=固定不动）
    gravity: wp.vec3,
    dt: float,
    x_out: wp.array(dtype=wp.vec3),
    v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    w = inv_mass[tid]
    if w == 0.0:
        x_out[tid] = x[tid]
        v_out[tid] = wp.vec3(0.0)
        return

    # 半隐式欧拉：先算新速度，再用新速度算新位置
    v_new = v[tid] + (f[tid] * w + gravity) * dt
    x_new = x[tid] + v_new * dt

    x_out[tid] = x_new
    v_out[tid] = v_new


def demo_integration():
    print("=" * 60)
    print("算子1：半隐式欧拉积分")
    print("=" * 60)

    n = 3
    x = wp.array([wp.vec3(0.0, 0.0, 10.0)] * n, dtype=wp.vec3)
    v = wp.zeros(n, dtype=wp.vec3)
    f = wp.zeros(n, dtype=wp.vec3)
    # 第3个粒子固定（inv_mass=0）
    inv_mass = wp.array([1.0, 1.0, 0.0], dtype=float)
    x_out = wp.zeros(n, dtype=wp.vec3)
    v_out = wp.zeros(n, dtype=wp.vec3)

    gravity = wp.vec3(0.0, 0.0, -9.81)

    for step in range(100):
        wp.launch(integrate_particles, dim=n,
                  inputs=[x, v, f, inv_mass, gravity, 0.01],
                  outputs=[x_out, v_out])
        x, x_out = x_out, x
        v, v_out = v_out, v

    positions = x.numpy()
    print(f"  粒子0 (自由): z = {positions[0][2]:.3f}  (应下落到约 5.1)")
    print(f"  粒子1 (自由): z = {positions[1][2]:.3f}")
    print(f"  粒子2 (固定): z = {positions[2][2]:.3f}  (应保持 10.0)")
    print()


# ============================================================================
# 算子2：弹簧力计算（使用 atomic_add）
# ============================================================================
# 对应 Newton 源码: newton/_src/solvers/semi_implicit/kernels_particle.py 第 21-65 行
#
# 弹簧连接两个粒子，如果距离偏离原始长度就产生恢复力。
# 关键点：一根弹簧影响两个粒子的力，需要用 atomic_add 避免竞争。
#
# 为什么需要 atomic_add？
#   假设弹簧0连接粒子(0,1)，弹簧1连接粒子(1,2)
#   两个线程同时往 f[1] 加力 → 数据竞争！
#   atomic_add 保证"加法是原子操作"，不会丢失。

@wp.kernel
def eval_springs(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    spring_indices: wp.array(dtype=int),    # [i0,j0, i1,j1, ...]
    spring_rest_length: wp.array(dtype=float),
    spring_ke: wp.array(dtype=float),       # 弹簧刚度
    spring_kd: wp.array(dtype=float),       # 弹簧阻尼
    f: wp.array(dtype=wp.vec3),             # 力（累积到这里）
):
    tid = wp.tid()  # 每个线程处理一根弹簧

    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]

    xi = x[i]
    xj = x[j]
    vi = v[i]
    vj = v[j]

    dx = xi - xj
    length = wp.length(dx)
    rest = spring_rest_length[tid]

    if length < 1.0e-6:
        return

    direction = dx / length

    # 弹簧力 = 刚度 × (当前长度 - 原始长度) × 方向
    stretch = length - rest
    # 阻尼力 = 阻尼系数 × 相对速度在弹簧方向的分量
    relative_vel = wp.dot(vi - vj, direction)

    force_magnitude = spring_ke[tid] * stretch + spring_kd[tid] * relative_vel
    force = direction * force_magnitude

    # 原子操作：安全地累加力（多个弹簧可能同时操作同一个粒子）
    wp.atomic_sub(f, i, force)   # 粒子 i 受到反方向的力
    wp.atomic_add(f, j, force)   # 粒子 j 受到正方向的力


def demo_springs():
    print("=" * 60)
    print("算子2：弹簧力计算（atomic_add）")
    print("=" * 60)

    # 3个粒子，2根弹簧：0-1, 1-2，原始长度 1.0
    x = wp.array([wp.vec3(0.0, 0.0, 0.0),
                   wp.vec3(1.5, 0.0, 0.0),    # 被拉伸了（距离 1.5 > 原始 1.0）
                   wp.vec3(3.0, 0.0, 0.0)], dtype=wp.vec3)
    v = wp.zeros(3, dtype=wp.vec3)
    f = wp.zeros(3, dtype=wp.vec3)

    indices = wp.array([0, 1, 1, 2], dtype=int)
    rest_length = wp.array([1.0, 1.0], dtype=float)
    ke = wp.array([100.0, 100.0], dtype=float)
    kd = wp.array([1.0, 1.0], dtype=float)

    wp.launch(eval_springs, dim=2, inputs=[x, v, indices, rest_length, ke, kd, f])

    forces = f.numpy()
    print(f"  粒子0 力: ({forces[0][0]:+.1f}, 0, 0)  ← 被弹簧0拉向粒子1")
    print(f"  粒子1 力: ({forces[1][0]:+.1f}, 0, 0)  ← 被两根弹簧拉（抵消一部分）")
    print(f"  粒子2 力: ({forces[2][0]:+.1f}, 0, 0)  ← 被弹簧1拉向粒子1")
    print(f"  注意：粒子1 的力 = 弹簧0 的贡献 + 弹簧1 的贡献（atomic_add 保证正确）")
    print()


# ============================================================================
# 算子3：SDF 碰撞检测（球体）
# ============================================================================
# 对应 Newton 源码: newton/_src/geometry/kernels.py 第 695-836 行
#
# SDF = Signed Distance Field（有符号距离场）
# 正值 = 在外面，负值 = 在里面
# 碰撞力 = ke * max(-d, 0) * 法线方向

@wp.func
def sphere_sdf(center: wp.vec3, radius: float, point: wp.vec3) -> float:
    """球体的 SDF：返回点到球面的距离（负值表示在球内）。"""
    return wp.length(point - center) - radius


@wp.func
def sphere_sdf_normal(center: wp.vec3, point: wp.vec3) -> wp.vec3:
    """球体 SDF 的梯度方向（法线）。"""
    delta = point - center
    length = wp.length(delta)
    if length > 1.0e-6:
        return delta / length
    return wp.vec3(0.0, 0.0, 1.0)


@wp.kernel
def eval_sphere_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    sphere_center: wp.vec3,
    sphere_radius: float,
    contact_ke: float,
    contact_kd: float,
    f: wp.array(dtype=wp.vec3),
):
    """检测粒子与球体的碰撞，计算接触力。"""
    tid = wp.tid()

    pos = x[tid]
    vel = v[tid]

    d = sphere_sdf(sphere_center, sphere_radius, pos)

    if d < 0.0:
        n = sphere_sdf_normal(sphere_center, pos)
        # 法向速度
        vn = wp.dot(vel, n)
        # 接触力 = 弹簧力（推出去）+ 阻尼力（减速）
        fn = contact_ke * (-d) - contact_kd * wp.min(vn, 0.0)
        wp.atomic_add(f, tid, n * fn)


@wp.func
def ground_sdf(point: wp.vec3) -> float:
    """地面 SDF：z=0 平面。"""
    return point[2]


@wp.kernel
def eval_ground_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    contact_ke: float,
    contact_kd: float,
    f: wp.array(dtype=wp.vec3),
):
    """检测粒子与地面的碰撞。"""
    tid = wp.tid()

    d = ground_sdf(x[tid])

    if d < 0.0:
        n = wp.vec3(0.0, 0.0, 1.0)
        vn = wp.dot(v[tid], n)
        fn = contact_ke * (-d) - contact_kd * wp.min(vn, 0.0)
        wp.atomic_add(f, tid, n * fn)


def demo_collision():
    print("=" * 60)
    print("算子3：SDF 碰撞检测")
    print("=" * 60)

    n = 5
    x = wp.array([
        wp.vec3(0.0, 0.0, 5.0),    # 空中
        wp.vec3(0.5, 0.0, 0.5),    # 球内（球心(0,0,0.5)半径0.5→球面z=0~1）
        wp.vec3(0.0, 0.0, -0.1),   # 地面以下
        wp.vec3(2.0, 0.0, 0.3),    # 球外，但在球附近
        wp.vec3(0.0, 0.0, 0.0),    # 刚好在地面上
    ], dtype=wp.vec3)
    v = wp.zeros(n, dtype=wp.vec3)
    f = wp.zeros(n, dtype=wp.vec3)

    wp.launch(eval_sphere_collision, dim=n,
              inputs=[x, v, wp.vec3(0.0, 0.0, 0.5), 0.5, 1000.0, 10.0, f])

    f_ground = wp.zeros(n, dtype=wp.vec3)
    wp.launch(eval_ground_collision, dim=n,
              inputs=[x, v, 1000.0, 10.0, f_ground])

    sf = f.numpy()
    gf = f_ground.numpy()
    labels = ["空中", "球内", "地下", "球外", "地面"]
    print("  球体碰撞力：")
    for i, label in enumerate(labels):
        mag = np.linalg.norm(sf[i])
        print(f"    粒子{i}({label}): |f| = {mag:.1f}")

    print("  地面碰撞力：")
    for i, label in enumerate(labels):
        mag = np.linalg.norm(gf[i])
        print(f"    粒子{i}({label}): |f| = {mag:.1f}")
    print()


# ============================================================================
# 算子4：刚体半隐式欧拉积分（含四元数旋转）
# ============================================================================
# 对应 Newton 源码: newton/_src/solvers/solver.py 第 62-106 行
#
# 刚体有 6 个自由度：3 个平移 + 3 个旋转
# 平移：v = v + (F/m + g) * dt, x = x + v * dt
# 旋转：ω = ω + I⁻¹ * (τ - ω × Iω) * dt, q = normalize(q + 0.5*ωq*dt)
#        ↑                 ↑
#        角速度更新         陀螺效应（旋转物体的特殊力矩）

@wp.kernel
def integrate_rigid_bodies(
    body_q: wp.array(dtype=wp.transform),       # 位姿（位置+旋转）
    body_qd: wp.array(dtype=wp.spatial_vector),  # 速度（线速度+角速度）
    body_f: wp.array(dtype=wp.spatial_vector),   # 力（力+力矩）
    mass: wp.array(dtype=float),
    inv_mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.mat33),
    inv_inertia: wp.array(dtype=wp.mat33),
    gravity: wp.vec3,
    dt: float,
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    q = body_q[tid]
    qd = body_qd[tid]
    f_ext = body_f[tid]

    x0 = wp.transform_get_translation(q)
    r0 = wp.transform_get_rotation(q)

    v0 = wp.spatial_top(qd)    # 线速度
    w0 = wp.spatial_bottom(qd)  # 角速度

    f0 = wp.spatial_top(f_ext)   # 力
    t0 = wp.spatial_bottom(f_ext)  # 力矩

    im = inv_mass[tid]

    # 线性部分：牛顿第二定律
    v1 = v0 + (f0 * im + gravity) * dt
    x1 = x0 + v1 * dt

    # 角度部分：欧拉旋转方程（在体坐标系下计算）
    I = inertia[tid]
    I_inv = inv_inertia[tid]

    wb = wp.quat_rotate_inv(r0, w0)                                # 世界→体坐标
    tb = wp.quat_rotate_inv(r0, t0) - wp.cross(wb, I * wb)         # 力矩 - 陀螺效应
    w1 = wp.quat_rotate(r0, wb + I_inv * tb * dt)                  # 新角速度（体→世界）
    r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)     # 四元数积分

    body_q_out[tid] = wp.transform(x1, r1)
    body_qd_out[tid] = wp.spatial_vector(v1, w1)


def demo_rigid_body():
    print("=" * 60)
    print("算子4：刚体积分（含四元数旋转）")
    print("=" * 60)

    q = wp.array([wp.transform(wp.vec3(0.0, 0.0, 5.0), wp.quat_identity())],
                 dtype=wp.transform)
    qd = wp.array([wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 2.0)],
                  dtype=wp.spatial_vector)
    f = wp.array([wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
                 dtype=wp.spatial_vector)

    mass_val = 1.0
    I_val = wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1)
    I_inv = wp.mat33(10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0)

    mass_arr = wp.array([mass_val], dtype=float)
    inv_mass_arr = wp.array([1.0 / mass_val], dtype=float)
    inertia_arr = wp.array([I_val], dtype=wp.mat33)
    inv_inertia_arr = wp.array([I_inv], dtype=wp.mat33)

    q_out = wp.zeros(1, dtype=wp.transform)
    qd_out = wp.zeros(1, dtype=wp.spatial_vector)

    gravity = wp.vec3(0.0, 0.0, -9.81)

    print(f"  初始: 位置=(0,0,5), 线速度=(1,0,0), 角速度=(0,0,2)")

    for _ in range(100):
        wp.launch(integrate_rigid_bodies, dim=1,
                  inputs=[q, qd, f, mass_arr, inv_mass_arr,
                          inertia_arr, inv_inertia_arr, gravity, 0.01],
                  outputs=[q_out, qd_out])
        q, q_out = q_out, q
        qd, qd_out = qd_out, qd

    result_q = q.numpy()[0]
    result_qd = qd.numpy()[0]
    pos = result_q[:3]
    vel_linear = result_qd[:3]

    print(f"  100步后: 位置=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    print(f"           线速度=({vel_linear[0]:.2f}, {vel_linear[1]:.2f}, {vel_linear[2]:.2f})")
    print(f"  X 方向匀速运动: x ≈ {0.0 + 1.0 * 1.0:.1f}")
    print(f"  Z 方向自由落体: z ≈ {5.0 - 0.5 * 9.81 * 1.0:.2f}")
    print()


# ============================================================================
# 算子5：完整的弹簧-粒子仿真（组合所有算子）
# ============================================================================

def demo_full_simulation():
    """组合积分+弹簧+地面碰撞，运行一个完整的弹簧-粒子仿真。"""
    print("=" * 60)
    print("算子5：完整弹簧-粒子仿真（组合所有算子）")
    print("=" * 60)

    # 3个粒子，2根弹簧：0(固定)-1-2，悬挂在 z=5
    n = 3
    x = wp.array([wp.vec3(0.0, 0.0, 5.0),     # 固定点
                   wp.vec3(0.0, 0.0, 4.0),     # 第1个
                   wp.vec3(0.0, 0.0, 3.0)], dtype=wp.vec3)
    v = wp.zeros(n, dtype=wp.vec3)
    inv_mass = wp.array([0.0, 1.0, 1.0], dtype=float)  # 粒子0固定

    indices = wp.array([0, 1, 1, 2], dtype=int)
    rest_length = wp.array([1.0, 1.0], dtype=float)
    ke = wp.array([500.0, 500.0], dtype=float)
    kd = wp.array([10.0, 10.0], dtype=float)

    x_out = wp.zeros(n, dtype=wp.vec3)
    v_out = wp.zeros(n, dtype=wp.vec3)

    gravity = wp.vec3(0.0, 0.0, -9.81)
    dt = 0.001

    print(f"  初始位置: {x.numpy()[:, 2]}")

    for step in range(2000):
        f = wp.zeros(n, dtype=wp.vec3)

        # 1. 弹簧力
        wp.launch(eval_springs, dim=2, inputs=[x, v, indices, rest_length, ke, kd, f])
        # 2. 地面碰撞
        wp.launch(eval_ground_collision, dim=n, inputs=[x, v, 1000.0, 50.0, f])
        # 3. 积分
        wp.launch(integrate_particles, dim=n,
                  inputs=[x, v, f, inv_mass, gravity, dt],
                  outputs=[x_out, v_out])

        x, x_out = x_out, x
        v, v_out = v_out, v

    positions = x.numpy()
    print(f"  2000步后 z: [{positions[0][2]:.3f}, {positions[1][2]:.3f}, {positions[2][2]:.3f}]")
    print(f"  粒子0(固定)应保持 z=5.0, 粒子1~2 应在地面以上且被弹簧拉住")
    assert positions[0][2] == 5.0, "固定粒子不应移动"
    assert positions[1][2] > 0.0, "粒子应在地面以上"
    assert positions[2][2] > 0.0, "粒子应在地面以上"
    print("  验证通过！")
    print()


# ============================================================================
if __name__ == "__main__":
    demo_integration()
    demo_springs()
    demo_collision()
    demo_rigid_body()
    demo_full_simulation()
    print("全部通过！")
