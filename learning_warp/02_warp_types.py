"""
Warp 学习 02：类型系统 —— vec3 / mat33 / quat / transform / spatial_vector
===========================================================================

【运行方式】
    cd /home/hkang/newton_related/newton
    uv run python learning_warp/02_warp_types.py

【学习目标】
- 掌握 Warp 的数学类型：向量、矩阵、四元数、变换、空间向量
- 理解这些类型在物理仿真中的用途
- 每种类型都用 kernel 演示，确认在 GPU 上正确运行
"""

import numpy as np
import warp as wp

wp.init()


# ============================================================================
# 第1节：wp.vec3 —— 三维向量（位置、速度、力）
# ============================================================================

@wp.kernel
def vec3_operations(output: wp.array(dtype=wp.vec3)):
    """演示 vec3 的常用操作。"""
    tid = wp.tid()

    a = wp.vec3(1.0, 2.0, 3.0)
    b = wp.vec3(4.0, 5.0, 6.0)

    if tid == 0:
        output[tid] = a + b                            # 加法
    elif tid == 1:
        output[tid] = a * 2.0                          # 标量乘法
    elif tid == 2:
        output[tid] = wp.vec3(wp.dot(a, b), 0.0, 0.0)  # 点积 = 32
    elif tid == 3:
        output[tid] = wp.cross(a, b)                    # 叉积
    elif tid == 4:
        output[tid] = wp.normalize(a)                   # 归一化
    elif tid == 5:
        output[tid] = wp.vec3(wp.length(a), 0.0, 0.0)  # 长度


def demo_vec3():
    print("=" * 60)
    print("第1节：wp.vec3 —— 三维向量")
    print("=" * 60)

    output = wp.zeros(6, dtype=wp.vec3)
    wp.launch(vec3_operations, dim=6, inputs=[output])
    results = output.numpy()

    labels = ["a + b", "a * 2", "dot(a,b)", "cross(a,b)", "normalize(a)", "length(a)"]
    for label, val in zip(labels, results):
        print(f"  {label:15s} = ({val[0]:.3f}, {val[1]:.3f}, {val[2]:.3f})")
    print()


# ============================================================================
# 第2节：wp.mat33 —— 3×3 矩阵（惯性张量、变形梯度）
# ============================================================================

@wp.kernel
def mat33_operations(output: wp.array(dtype=wp.vec3)):
    """演示 mat33 的用法。"""
    tid = wp.tid()

    # 创建矩阵：按行构造
    m = wp.mat33(
        1.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 3.0,
    )
    v = wp.vec3(1.0, 1.0, 1.0)

    if tid == 0:
        output[tid] = m * v                                 # 矩阵×向量
    elif tid == 1:
        output[tid] = wp.vec3(wp.determinant(m), 0.0, 0.0)  # 行列式 = 6
    elif tid == 2:
        inv_m = wp.inverse(m)
        output[tid] = inv_m * v                              # 逆矩阵×向量


def demo_mat33():
    print("=" * 60)
    print("第2节：wp.mat33 —— 3×3 矩阵")
    print("=" * 60)

    output = wp.zeros(3, dtype=wp.vec3)
    wp.launch(mat33_operations, dim=3, inputs=[output])
    results = output.numpy()

    print(f"  diag(1,2,3) × (1,1,1)     = ({results[0][0]:.1f}, {results[0][1]:.1f}, {results[0][2]:.1f})")
    print(f"  det(diag(1,2,3))           = {results[1][0]:.1f}")
    print(f"  inv(diag(1,2,3)) × (1,1,1) = ({results[2][0]:.3f}, {results[2][1]:.3f}, {results[2][2]:.3f})")
    print()


# ============================================================================
# 第3节：wp.quat —— 四元数（旋转）
# ============================================================================
# 四元数 (w, x, y, z) 用于表示 3D 旋转。
# 比欧拉角没有万向锁问题，比旋转矩阵省内存（4个数 vs 9个数）。
# Newton 中所有刚体的旋转都用四元数表示。

@wp.kernel
def quat_operations(output: wp.array(dtype=wp.vec3)):
    """演示四元数操作。"""
    tid = wp.tid()

    # 绕 Z 轴旋转 90°
    q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)
    v = wp.vec3(1.0, 0.0, 0.0)  # X 方向的单位向量

    if tid == 0:
        output[tid] = wp.quat_rotate(q, v)      # 旋转后应该变成 (0, 1, 0)
    elif tid == 1:
        output[tid] = wp.quat_rotate_inv(q, wp.vec3(0.0, 1.0, 0.0))  # 逆旋转
    elif tid == 2:
        q2 = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)
        q_combined = q * q2  # 两次 90° = 180°
        output[tid] = wp.quat_rotate(q_combined, v)  # 应该变成 (-1, 0, 0)


def demo_quat():
    print("=" * 60)
    print("第3节：wp.quat —— 四元数（旋转）")
    print("=" * 60)

    output = wp.zeros(3, dtype=wp.vec3)
    wp.launch(quat_operations, dim=3, inputs=[output])
    results = output.numpy()

    print(f"  绕 Z 轴转 90°: (1,0,0) → ({results[0][0]:.1f}, {results[0][1]:.1f}, {results[0][2]:.1f})  期望: (0,1,0)")
    print(f"  逆旋转:       (0,1,0) → ({results[1][0]:.1f}, {results[1][1]:.1f}, {results[1][2]:.1f})  期望: (1,0,0)")
    print(f"  两次 90°=180°: (1,0,0) → ({results[2][0]:.1f}, {results[2][1]:.1f}, {results[2][2]:.1f})  期望: (-1,0,0)")
    print()


# ============================================================================
# 第4节：wp.transform —— 位姿（位置 + 旋转）
# ============================================================================
# transform = (vec3 位置, quat 旋转)
# Newton 中每个刚体的位姿就是一个 transform（State.body_q 的类型）。

@wp.kernel
def transform_operations(output: wp.array(dtype=wp.vec3)):
    """演示 transform 操作。"""
    tid = wp.tid()

    # 创建变换：位置 (1,2,3)，绕 Z 轴旋转 90°
    pos = wp.vec3(1.0, 2.0, 3.0)
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)
    xform = wp.transform(pos, rot)

    local_point = wp.vec3(1.0, 0.0, 0.0)

    if tid == 0:
        output[tid] = wp.transform_get_translation(xform)
    elif tid == 1:
        # transform_point: 先旋转再平移（完整变换）
        output[tid] = wp.transform_point(xform, local_point)
    elif tid == 2:
        # transform_vector: 只旋转不平移（方向变换）
        output[tid] = wp.transform_vector(xform, local_point)


def demo_transform():
    print("=" * 60)
    print("第4节：wp.transform —— 位姿（位置+旋转）")
    print("=" * 60)

    output = wp.zeros(3, dtype=wp.vec3)
    wp.launch(transform_operations, dim=3, inputs=[output])
    results = output.numpy()

    print(f"  位置:           ({results[0][0]:.1f}, {results[0][1]:.1f}, {results[0][2]:.1f})")
    print(f"  变换点(1,0,0):  ({results[1][0]:.1f}, {results[1][1]:.1f}, {results[1][2]:.1f})  旋转+平移")
    print(f"  变换向量(1,0,0): ({results[2][0]:.1f}, {results[2][1]:.1f}, {results[2][2]:.1f})  只旋转")
    print()


# ============================================================================
# 第5节：wp.spatial_vector —— 空间向量（刚体力学）
# ============================================================================
# spatial_vector = 6 维向量，前 3 维是线性量，后 3 维是角量。
# Newton 中：
#   body_qd (速度)  = spatial_vector(线速度, 角速度)
#   body_f  (力)    = spatial_vector(力, 力矩)

@wp.kernel
def spatial_operations(output: wp.array(dtype=wp.vec3)):
    """演示空间向量操作。"""
    tid = wp.tid()

    # 创建空间向量：线速度 (1,0,0)，角速度 (0,0,5)
    sv = wp.spatial_vector(wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 5.0))

    if tid == 0:
        output[tid] = wp.spatial_top(sv)     # 线性部分（前3维）
    elif tid == 1:
        output[tid] = wp.spatial_bottom(sv)  # 角量部分（后3维）


def demo_spatial():
    print("=" * 60)
    print("第5节：wp.spatial_vector —— 空间向量")
    print("=" * 60)

    output = wp.zeros(2, dtype=wp.vec3)
    wp.launch(spatial_operations, dim=2, inputs=[output])
    results = output.numpy()

    print(f"  spatial_top(线性):  ({results[0][0]:.1f}, {results[0][1]:.1f}, {results[0][2]:.1f})  = 线速度")
    print(f"  spatial_bottom(角): ({results[1][0]:.1f}, {results[1][1]:.1f}, {results[1][2]:.1f})  = 角速度")
    print()


# ============================================================================
# 第6节：综合示例 —— 刚体坐标变换（Newton 中的核心操作）
# ============================================================================

@wp.func
def world_to_local(body_xform: wp.transform, world_point: wp.vec3) -> wp.vec3:
    """将世界坐标系的点转换到刚体的局部坐标系。"""
    inv_xform = wp.transform_inverse(body_xform)
    return wp.transform_point(inv_xform, world_point)


@wp.func
def local_to_world(body_xform: wp.transform, local_point: wp.vec3) -> wp.vec3:
    """将刚体局部坐标系的点转换到世界坐标系。"""
    return wp.transform_point(body_xform, local_point)


@wp.kernel
def coordinate_transform_demo(body_q: wp.array(dtype=wp.transform),
                              world_points: wp.array(dtype=wp.vec3),
                              local_results: wp.array(dtype=wp.vec3),
                              world_results: wp.array(dtype=wp.vec3)):
    """演示坐标变换。"""
    tid = wp.tid()
    xform = body_q[0]
    wp_point = world_points[tid]

    local_p = world_to_local(xform, wp_point)
    world_p = local_to_world(xform, local_p)

    local_results[tid] = local_p
    world_results[tid] = world_p


def demo_coordinate_transform():
    print("=" * 60)
    print("第6节：刚体坐标变换（Newton 核心操作）")
    print("=" * 60)

    xform = wp.transform(
        wp.vec3(5.0, 0.0, 0.0),
        wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
    )
    body_q = wp.array([xform], dtype=wp.transform)

    world_points = wp.array(
        [wp.vec3(5.0, 0.0, 0.0), wp.vec3(6.0, 0.0, 0.0), wp.vec3(5.0, 1.0, 0.0)],
        dtype=wp.vec3,
    )
    local_results = wp.zeros(3, dtype=wp.vec3)
    world_results = wp.zeros(3, dtype=wp.vec3)

    wp.launch(
        coordinate_transform_demo, dim=3,
        inputs=[body_q, world_points, local_results, world_results],
    )

    lr = local_results.numpy()
    wr = world_results.numpy()

    print("  刚体位置: (5,0,0), 旋转: 绕 Z 轴 90°")
    for i in range(3):
        wp_str = f"({world_points.numpy()[i][0]:.0f},{world_points.numpy()[i][1]:.0f},{world_points.numpy()[i][2]:.0f})"
        lp_str = f"({lr[i][0]:.1f},{lr[i][1]:.1f},{lr[i][2]:.1f})"
        rp_str = f"({wr[i][0]:.1f},{wr[i][1]:.1f},{wr[i][2]:.1f})"
        print(f"  世界{wp_str} → 局部{lp_str} → 还原{rp_str}")
    print()


# ============================================================================
if __name__ == "__main__":
    demo_vec3()
    demo_mat33()
    demo_quat()
    demo_transform()
    demo_spatial()
    demo_coordinate_transform()
    print("全部通过！")
