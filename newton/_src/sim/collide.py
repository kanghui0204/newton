# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from enum import IntEnum

import numpy as np
import warp as wp

from ..geometry.broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit
from ..geometry.broad_phase_sap import BroadPhaseSAP
from ..geometry.collision_core import compute_tight_aabb_from_support
from ..geometry.contact_data import ContactData
from ..geometry.kernels import create_soft_contacts
from ..geometry.narrow_phase import NarrowPhase
from ..geometry.sdf_hydroelastic import SDFHydroelastic, SDFHydroelasticConfig
from ..geometry.support_function import (
    GenericShapeData,
    SupportMapDataProvider,
    pack_mesh_ptr,
)
from ..geometry.types import GeoType
from ..sim.contacts import Contacts
from ..sim.model import Model
from ..sim.state import State


@wp.struct
class ContactWriterData:
    """Contact writer data for collide write_contact function."""

    contact_max: int
    # Body information arrays (for transforming to body-local coordinates)
    body_q: wp.array(dtype=wp.transform)
    shape_body: wp.array(dtype=int)
    shape_contact_margin: wp.array(dtype=float)
    # Output arrays
    contact_count: wp.array(dtype=int)
    out_shape0: wp.array(dtype=int)
    out_shape1: wp.array(dtype=int)
    out_point0: wp.array(dtype=wp.vec3)
    out_point1: wp.array(dtype=wp.vec3)
    out_offset0: wp.array(dtype=wp.vec3)
    out_offset1: wp.array(dtype=wp.vec3)
    out_normal: wp.array(dtype=wp.vec3)
    out_thickness0: wp.array(dtype=float)
    out_thickness1: wp.array(dtype=float)
    out_tids: wp.array(dtype=int)
    # Per-contact shape properties, empty arrays if not enabled.
    # Zero-values indicate that no per-contact shape properties are set for this contact
    out_stiffness: wp.array(dtype=float)
    out_damping: wp.array(dtype=float)
    out_friction: wp.array(dtype=float)


class BroadPhaseMode(IntEnum):
    """Broad phase collision detection mode."""

    NXN = 0
    """All-pairs broad phase with AABB checks (simple, O(N²) but good for small scenes)"""

    SAP = 1
    """Sweep and Prune broad phase with AABB sorting (faster for larger scenes, O(N log N))"""

    EXPLICIT = 2
    """Use precomputed shape pairs (most efficient when pairs are known ahead of time)"""


@wp.func
def write_contact(
    contact_data: ContactData,
    writer_data: ContactWriterData,
    output_index: int,
):
    """
    Write a contact to the output arrays using ContactData and ContactWriterData.

    Args:
        contact_data: ContactData struct containing contact information
        writer_data: ContactWriterData struct containing body info and output arrays
        output_index: If -1, use atomic_add to get the next available index if contact distance is less than margin. If >= 0, use this index directly and skip margin check.
    """
    total_separation_needed = (
        contact_data.radius_eff_a + contact_data.radius_eff_b + contact_data.thickness_a + contact_data.thickness_b
    )

    offset_mag_a = contact_data.radius_eff_a + contact_data.thickness_a
    offset_mag_b = contact_data.radius_eff_b + contact_data.thickness_b

    # Distance calculation matching box_plane_collision
    contact_normal_a_to_b = wp.normalize(contact_data.contact_normal_a_to_b)

    a_contact_world = contact_data.contact_point_center - contact_normal_a_to_b * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_a
    )
    b_contact_world = contact_data.contact_point_center + contact_normal_a_to_b * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_b
    )

    diff = b_contact_world - a_contact_world
    distance = wp.dot(diff, contact_normal_a_to_b)
    d = distance - total_separation_needed

    # Use per-shape contact margins (sum of both shapes, consistent with thickness)
    margin_a = writer_data.shape_contact_margin[contact_data.shape_a]
    margin_b = writer_data.shape_contact_margin[contact_data.shape_b]
    contact_margin = margin_a + margin_b

    index = output_index

    if index < 0:
        # compute index using atomic counter
        if d > contact_margin:
            return
        index = wp.atomic_add(writer_data.contact_count, 0, 1)
        if index >= writer_data.contact_max:
            # Reached buffer limit
            wp.atomic_add(writer_data.contact_count, 0, -1)
            return

    if index >= writer_data.contact_max:
        return

    writer_data.out_shape0[index] = contact_data.shape_a
    writer_data.out_shape1[index] = contact_data.shape_b

    # Get body indices for the shapes
    body0 = writer_data.shape_body[contact_data.shape_a]
    body1 = writer_data.shape_body[contact_data.shape_b]

    # Compute body inverse transforms
    X_bw_a = wp.transform_identity() if body0 == -1 else wp.transform_inverse(writer_data.body_q[body0])
    X_bw_b = wp.transform_identity() if body1 == -1 else wp.transform_inverse(writer_data.body_q[body1])

    # Contact points are stored in body frames
    writer_data.out_point0[index] = wp.transform_point(X_bw_a, a_contact_world)
    writer_data.out_point1[index] = wp.transform_point(X_bw_b, b_contact_world)

    # Match kernels.py convention
    contact_normal = -contact_normal_a_to_b

    # Offsets in body frames
    writer_data.out_offset0[index] = wp.transform_vector(X_bw_a, -offset_mag_a * contact_normal)
    writer_data.out_offset1[index] = wp.transform_vector(X_bw_b, offset_mag_b * contact_normal)

    writer_data.out_normal[index] = contact_normal
    writer_data.out_thickness0[index] = offset_mag_a
    writer_data.out_thickness1[index] = offset_mag_b
    writer_data.out_tids[index] = 0  # tid not available in this context

    # Write stiffness/damping/friction only if per-contact shape properties are enabled
    if writer_data.out_stiffness.shape[0] > 0:
        writer_data.out_stiffness[index] = contact_data.contact_stiffness
        writer_data.out_damping[index] = contact_data.contact_damping
        writer_data.out_friction[index] = contact_data.contact_friction_scale


@wp.kernel
def compute_shape_aabbs(
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_collision_radius: wp.array(dtype=float),
    shape_source_ptr: wp.array(dtype=wp.uint64),
    shape_contact_margin: wp.array(dtype=float),
    # outputs
    aabb_lower: wp.array(dtype=wp.vec3),
    aabb_upper: wp.array(dtype=wp.vec3),
):
    """Compute axis-aligned bounding boxes for each shape in world space.

    Uses support function for most shapes. Infinite planes and meshes use bounding sphere fallback.
    AABBs are enlarged by per-shape contact margin for contact detection.

    Note: Shape thickness is NOT included in AABB expansion - it is applied during narrow phase.
    Therefore, shape_contact_margin should be >= shape_thickness to ensure proper broad phase detection.
    """
    shape_id = wp.tid()

    rigid_id = shape_body[shape_id]
    geo_type = shape_type[shape_id]

    # Compute world transform
    if rigid_id == -1:
        X_ws = shape_transform[shape_id]
    else:
        X_ws = wp.transform_multiply(body_q[rigid_id], shape_transform[shape_id])

    pos = wp.transform_get_translation(X_ws)
    orientation = wp.transform_get_rotation(X_ws)

    # Enlarge AABB by per-shape contact margin for contact detection
    contact_margin = shape_contact_margin[shape_id]
    margin_vec = wp.vec3(contact_margin, contact_margin, contact_margin)

    # Check if this is an infinite plane, mesh, or SDF - use bounding sphere fallback
    scale = shape_scale[shape_id]
    is_infinite_plane = (geo_type == int(GeoType.PLANE)) and (scale[0] == 0.0 and scale[1] == 0.0)
    is_mesh = geo_type == int(GeoType.MESH)
    is_sdf = geo_type == int(GeoType.SDF)

    if is_infinite_plane or is_mesh or is_sdf:
        # Use conservative bounding sphere approach for infinite planes, meshes, and SDFs
        radius = shape_collision_radius[shape_id]
        half_extents = wp.vec3(radius, radius, radius)
        aabb_lower[shape_id] = pos - half_extents - margin_vec
        aabb_upper[shape_id] = pos + half_extents + margin_vec
    else:
        # Use support function to compute tight AABB
        # Create generic shape data
        shape_data = GenericShapeData()
        shape_data.shape_type = geo_type
        shape_data.scale = scale
        shape_data.auxiliary = wp.vec3(0.0, 0.0, 0.0)

        # For CONVEX_MESH, pack the mesh pointer
        if geo_type == int(GeoType.CONVEX_MESH):
            shape_data.auxiliary = pack_mesh_ptr(shape_source_ptr[shape_id])

        data_provider = SupportMapDataProvider()

        # Compute tight AABB using helper function
        aabb_min_world, aabb_max_world = compute_tight_aabb_from_support(shape_data, orientation, pos, data_provider)

        aabb_lower[shape_id] = aabb_min_world - margin_vec
        aabb_upper[shape_id] = aabb_max_world + margin_vec


@wp.kernel
def prepare_geom_data_kernel(
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_thickness: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    # Outputs
    geom_data: wp.array(dtype=wp.vec4),  # scale xyz, thickness w
    geom_transform: wp.array(dtype=wp.transform),  # world space transform
):
    """Prepare geometry data arrays for NarrowPhase API."""
    idx = wp.tid()

    # Pack scale and thickness into geom_data
    scale = shape_scale[idx]
    thickness = shape_thickness[idx]
    geom_data[idx] = wp.vec4(scale[0], scale[1], scale[2], thickness)

    # Compute world space transform
    body_idx = shape_body[idx]
    if body_idx >= 0:
        geom_transform[idx] = wp.transform_multiply(body_q[body_idx], shape_transform[idx])
    else:
        geom_transform[idx] = shape_transform[idx]


def _estimate_rigid_contact_max(model: Model) -> int:
    """
    Estimate the maximum number of rigid contacts for the collision pipeline.

    Uses a linear neighbor-budget estimate assuming each non-plane shape contacts
    at most ``MAX_NEIGHBORS_PER_SHAPE`` others (spatial locality).  The non-plane
    term is additive across independent worlds so a single-pool computation is
    correct.  The plane term (each plane vs all non-planes in its world) would be
    quadratic if computed globally, so it is evaluated per world when metadata is
    available.

    When precomputed contact pairs are available their count is used as an
    alternative tighter bound (``min`` of heuristic and pair-based estimate).

    Args:
        model: The simulation model.

    Returns:
        Estimated maximum number of rigid contacts.
    """
    if not hasattr(model, "shape_type") or model.shape_type is None:
        return 1000  # Fallback

    shape_types = model.shape_type.numpy()

    # Primitive pairs (GJK/MPR) produce up to 5 manifold contacts.
    # Mesh-involved pairs (SDF + contact reduction) typically retain ~40.
    PRIMITIVE_CPP = 5
    MESH_CPP = 40
    MAX_NEIGHBORS_PER_SHAPE = 20

    mesh_mask = shape_types == int(GeoType.MESH)
    plane_mask = shape_types == int(GeoType.PLANE)
    non_plane_mask = ~plane_mask
    num_meshes = int(np.count_nonzero(mesh_mask))
    num_non_planes = int(np.count_nonzero(non_plane_mask))
    num_primitives = num_non_planes - num_meshes
    num_planes = int(np.count_nonzero(plane_mask))

    # Weighted contacts from non-plane shape types.
    # Each shape's neighbor pairs are weighted by its type's contacts-per-pair.
    # Divide by 2 to avoid double-counting pairs.
    non_plane_contacts = (
        num_primitives * MAX_NEIGHBORS_PER_SHAPE * PRIMITIVE_CPP + num_meshes * MAX_NEIGHBORS_PER_SHAPE * MESH_CPP
    ) // 2

    # Weighted average contacts-per-pair based on the scene's shape mix.
    avg_cpp = (
        (num_primitives * PRIMITIVE_CPP + num_meshes * MESH_CPP) // max(num_non_planes, 1) if num_non_planes > 0 else 0
    )

    # Plane contacts: each plane contacts all non-plane shapes *in its world*.
    # The naive global formula (num_planes * num_non_planes) is O(worlds²) when
    # both counts grow with the number of worlds.  Use per-world counts instead.
    plane_contacts = 0
    if num_planes > 0 and num_non_planes > 0:
        has_world_info = (
            hasattr(model, "shape_world")
            and model.shape_world is not None
            and hasattr(model, "num_worlds")
            and model.num_worlds > 0
        )
        shape_world = model.shape_world.numpy() if has_world_info else None

        if shape_world is not None and len(shape_world) == len(shape_types):
            global_mask = shape_world == -1
            local_mask = ~global_mask
            n_worlds = model.num_worlds

            global_planes = int(np.count_nonzero(global_mask & plane_mask))
            global_non_planes = int(np.count_nonzero(global_mask & non_plane_mask))

            local_plane_counts = np.bincount(shape_world[local_mask & plane_mask], minlength=n_worlds)[:n_worlds]
            local_non_plane_counts = np.bincount(shape_world[local_mask & non_plane_mask], minlength=n_worlds)[
                :n_worlds
            ]

            per_world_planes = local_plane_counts + global_planes
            per_world_non_planes = local_non_plane_counts + global_non_planes

            # Global-global pairs appear in every world slice; keep one copy.
            plane_pair_count = int(np.sum(per_world_planes * per_world_non_planes))
            if n_worlds > 1:
                plane_pair_count -= (n_worlds - 1) * global_planes * global_non_planes
            plane_contacts = plane_pair_count * avg_cpp
        else:
            # Fallback: exact type-weighted sum (correct for single-world models).
            plane_contacts = num_planes * (num_primitives * PRIMITIVE_CPP + num_meshes * MESH_CPP)

    total_contacts = non_plane_contacts + plane_contacts

    # When precomputed contact pairs are available, use as a tighter bound.
    if hasattr(model, "shape_contact_pair_count") and model.shape_contact_pair_count > 0:
        weighted_cpp = max(avg_cpp, PRIMITIVE_CPP)
        pair_contacts = int(model.shape_contact_pair_count) * weighted_cpp
        total_contacts = min(total_contacts, pair_contacts)

    # Ensure minimum allocation
    return max(1000, total_contacts)


class CollisionPipeline:
    """碰撞检测管线：从"哪些物体可能碰撞"到"精确接触点计算"的完整流程。

    Full-featured collision pipeline with GJK/MPR narrow phase and pluggable broad phase.

    碰撞检测分三步：
        1. 宽相（Broad Phase）：快速排除不可能碰撞的物体对（用 AABB 包围盒）
        2. 窄相（Narrow Phase）：对宽相筛选出的候选对做精确碰撞检测（GJK/MPR 算法）
        3. 软接触（Soft Contact）：粒子与形状之间的 SDF 距离查询（用于布料/可微仿真）

    Key features:
        - GJK/MPR algorithms for convex-convex collision detection
        - Multiple broad phase modes: NXN (all-pairs), SAP (sweep-and-prune), EXPLICIT (precomputed pairs)
        - Mesh-mesh collision via SDF with contact reduction
        - Optional hydroelastic contact model for compliant surfaces
    """

    def __init__(
        self,
        model: Model,                                                        # 仿真模型
        *,
        reduce_contacts: bool = True,                                        # 是否做接触归约
        rigid_contact_max: int | None = None,                                # 刚体接触最大数量（GPU 缓冲区大小）
        shape_pairs_filtered: wp.array(dtype=wp.vec2i) | None = None,        # EXPLICIT 模式的预计算碰撞对
        soft_contact_max: int | None = None,                                 # 软接触最大数量
        soft_contact_margin: float = 0.01,                                   # 软接触感知范围（米）
        requires_grad: bool | None = None,                                   # 是否支持梯度（True=只算可微的软接触）
        broad_phase_mode: BroadPhaseMode = BroadPhaseMode.EXPLICIT,          # 宽相模式：NXN/SAP/EXPLICIT
        sap_sort_type=None,                                                  # SAP 排序算法
        sdf_hydroelastic_config: SDFHydroelasticConfig | None = None,        # SDF 弹性接触配置（高级）
    ):
        """Initialize the CollisionPipeline.
        初始化碰撞检测管线。

        Args:
            model (Model): The simulation model.
                仿真模型（包含所有形状、刚体、粒子信息）。

            reduce_contacts (bool, optional): Whether to reduce contacts for mesh-mesh collisions. Defaults to True.
                是否对网格碰撞做接触归约。网格碰撞可能产生大量接触点，
                归约后只保留最重要的几个，减少求解器的计算量。

            rigid_contact_max (int | None, optional): Maximum number of rigid contacts to allocate.
                If None, estimated based on broad phase mode:
                - EXPLICIT: len(shape_pairs_filtered) * 10 contacts
                - NXN/SAP: shape_count * 20 contacts (assumes ~20 contacts per shape)
                For better memory efficiency, use rigid_contact_max computed from actual collision pairs.
                整个场景所有刚体接触点的 GPU 缓冲区总大小（不是单个物体的）。
                GPU 不能动态分配内存，必须预先开好固定大小的数组。
                碰撞检测时按顺序写入接触点，超出上限的接触点会被丢弃（导致穿透）。
                设太小 → 接触丢失，物体穿透；设太大 → 浪费显存。
                None 时自动估算，一般不需要手动设置。

            soft_contact_max (int | None, optional): Maximum number of soft contacts to allocate.
                If None, computed as shape_count * particle_count.
                整个场景所有粒子-形状软接触点的 GPU 缓冲区总大小。
                默认 shape_count × particle_count（最坏情况：每个粒子碰每个形状），
                通常远远用不完。布料/粒子仿真中会用到。

            soft_contact_margin (float, optional): Margin for soft contact generation. Defaults to 0.01.
                软接触的感知范围（米）。粒子在此距离内才会与形状产生接触。
                普通仿真: 0.01（省计算）
                可微仿真: 设大（如 10.0），让梯度信号传得更远。

            requires_grad (bool | None, optional): Whether to enable gradient computation. If None, uses model.requires_grad.
                控制碰撞检测算哪些类型（不是"梯度开不开"，而是"哪些碰撞类型参与计算"）。
                False: 两种碰撞都算（rigid + soft），碰撞最完整，用于普通仿真。
                True:  跳过 rigid contact，只算 soft contact，用于可微仿真。
                       原因：GJK/MPR 窄相 kernel 编译时 enable_backward=False，
                       没有反向代码，如果被 Tape 录进去 backward() 会出错。
                       所以 requires_grad=True 时直接跳过，只走可微的 soft contact。
                None:  跟随 model.requires_grad。

            broad_phase_mode (BroadPhaseMode, optional): Broad phase mode for collision detection.
                - BroadPhaseMode.NXN: Use all-pairs AABB broad phase (O(N²), good for small scenes)
                - BroadPhaseMode.SAP: Use sweep-and-prune AABB broad phase (O(N log N), better for larger scenes)
                - BroadPhaseMode.EXPLICIT: Use precomputed shape pairs (most efficient when pairs known)
                Defaults to BroadPhaseMode.EXPLICIT.
                宽相检测模式，三种模式：
                EXPLICIT: 预计算碰撞对，最高效，适合碰撞体不增减的场景。
                NXN: 全对比较 O(N²)，适合形状少（<100）的场景。
                SAP: 扫描排序剪枝 O(N log N)，适合形状多（>100）的场景。

            shape_pairs_filtered (wp.array | None, optional): Precomputed shape pairs for EXPLICIT mode.
                When broad_phase_mode is BroadPhaseMode.EXPLICIT, uses model.shape_contact_pairs if not provided. For NXN/SAP modes, ignored.
                EXPLICIT 模式下，告诉引擎"哪些形状对需要检测碰撞"。
                每个元素是 vec2i(shape_a, shape_b)，例如 [(0,1), (0,2), (1,2)]。
                不需要检测的对（如同一机器人上的相邻零件）会被排除。
                一般不需要手动传——finalize() 时 Newton 自动算好存在
                model.shape_contact_pairs 里，CollisionPipeline 会自动读取。

            sap_sort_type (SAPSortType | None, optional): Sorting algorithm for SAP broad phase.
                Only used when broad_phase_mode is BroadPhaseMode.SAP. Options: SEGMENTED or TILE.
                If None, uses default (SEGMENTED).
                SAP 宽相内部对 AABB 排序时用的算法。
                SEGMENTED（分段排序，默认）或 TILE（Tile 排序）。
                只在 SAP 模式下有效，几乎不需要改。

            sdf_hydroelastic_config (SDFHydroelasticConfig | None, optional): Configuration for SDF hydroelastic collision handling. Defaults to None.
                高级功能：柔性表面碰撞模型。默认 None = 不启用。

                【普通碰撞（默认）—— 两个面碰了记几个点？】

                  两个盒子面贴面时：
                    理论上整个面都在接触，有无穷多个接触点。
                    实际上 GJK/MPR 算法只返回 1 个接触点（最深穿透点）：

                      ┌──────────┐
                      │  盒子A    │
                      └─────●────┘  ← 只记录 1 个点
                      ┌──────────┐
                      │  盒子B    │
                      └──────────┘

                    但 1 个点不够！盒子会像陀螺一样绕接触点打转。
                    需要至少 4 个点才能稳定支撑一个面。

                    为什么 1 个点会打转？
                      重力在盒子中心（向下），支撑力在接触点（向上），
                      两者不在同一条线上 → 产生力矩 → 盒子绕接触点旋转。
                      就像铅笔立在桌上，只有底部一点支撑，碰一下就倒。

                    Newton 的解决方案：contact reduction（接触归约）
                    把 1 个接触点"展开"成 4 个分散在接触面角上的点：

                      ┌──────────┐
                      │  盒子A    │
                      └─●──●──●─●┘  ← 归约后 4 个点，像桌子 4 条腿
                      ┌──────────┐      不管重力在哪，4 点总能平衡
                      │  盒子B    │
                      └──────────┘

                    "归约"根据场景不同，可以是：
                      少→多：GJK 只给 1 个点 → 展开成 4 个代表点
                      多→少：网格碰撞产生几百个点 → 精简到有代表性的几个
                    核心目的：得到数量合理、分布均匀的接触点集合。
                    这就是 reduce_contacts=True 的作用。
                    计算量小，适合大多数刚体仿真。

                【Hydroelastic（柔性表面）—— 记多少个点？】

                  不是记"几个点"，而是在接触面上生成网格，每个网格节点一个采样点：

                      ┌──────────┐
                      │  盒子A    │
                      └──────────┘
                      ╔══════════╗  ← 接触面被网格化
                      ║●──●──●──●║
                      ║│  │  │  │║    几十到几百个点
                      ║●──●──●──●║    每个点有自己的压力
                      ╚══════════╝
                      ┌──────────┐
                      │  盒子B    │
                      └──────────┘

                  Hydroelastic 的计算流程：
                    1. 用两个物体的 SDF 计算重叠区域
                    2. 在重叠区域生成接触面网格
                    3. 在每个网格节点上计算压力
                    → 接触力 = 所有节点的压力之和

                  类比：
                    普通碰撞    = 手指戳桌子（1 个接触点，力集中）
                    Hydroelastic = 手掌按桌子（一整个面，力分散）

                【对比总结】

                                      普通碰撞              Hydroelastic
                  接触点数量           1 个（归约后约 4 个）   几十~几百个
                  力的分布             集中在SDFHydroelasticConfig上
                  计算量               小                      大
                  稳定性               归约后还行              天然稳定
                  适用场景             大多数刚体仿真          轮胎/柔性夹爪/精确接触力
                  Newton 中设置        默认（None）            传入 SDFHydroelasticConfig

                【None 的含义】
                  None 不是"没有接触点"！是"不用 Hydroelastic，用普通碰撞"。
                  普通碰撞（GJK/MPR + 归约）仍然正常工作，接触点一个不少。

                【SDFHydroelasticConfig 的所有配置项】
                  源码位置: newton/_src/geometry/sdf_hydroelastic.py 第 102-133 行

                  SDFHydroelasticConfig(
                    reduce_contacts=True,      # 是否归约接触点（True=精简,False=全部保留）
                    buffer_mult_broad=1,       # 宽相缓冲区倍率（溢出报警时增大）
                    buffer_mult_iso=1,         # 等值面提取缓冲区倍率（溢出时增大）
                    buffer_mult_contact=1,     # 接触点缓冲区倍率（溢出时增大）
                    grid_size=256*8*128,       # 网格大小（性能调优用）
                    output_contact_surface=False,  # 是否输出接触面顶点（用于可视化调试）
                    normal_matching=True,      # 归约后旋转法线，使合力方向一致
                    anchor_contact=False,      # 在压力中心添加锚点接触（保持力矩平衡）
                    moment_matching=False,     # 缩放摩擦系数匹配力矩（需 anchor_contact=True）
                    margin_contact_area=1e-2,  # 非穿透边缘接触的面积
                  )
                  大多数情况用默认值就行，只有溢出报警时才需要增大 buffer_mult_* 。
        """
        # ─── 第1步：读取模型基本信息 ───
        shape_count = model.shape_count      # 场景中碰撞形状的数量
        particle_count = model.particle_count  # 粒子数量（布料/软体的顶点）
        device = model.device                # GPU 设备（如 "cuda:0"）

        # ─── 第2步：估算接触缓冲区大小 ───
        # GPU 需要预分配固定大小的缓冲区存放碰撞结果
        if rigid_contact_max is None:
            rigid_contact_max = _estimate_rigid_contact_max(model)
        self.rigid_contact_max = rigid_contact_max
        if requires_grad is None:
            requires_grad = model.requires_grad

        # ─── 第3步：准备 EXPLICIT 模式的碰撞对 ───
        # EXPLICIT 模式在 finalize() 时就预计算好了哪些形状对需要检测
        if shape_pairs_filtered is None and broad_phase_mode == BroadPhaseMode.EXPLICIT:
            shape_pairs_filtered = getattr(model, "shape_contact_pairs", None)

        # ─── 第4步：初始化 SDF 弹性接触（高级功能，通常为 None）───
        sdf_hydroelastic = SDFHydroelastic._from_model(model, config=sdf_hydroelastic_config, writer_func=write_contact)

        # ─── 第5步：检测是否有网格形状（网格碰撞需要特殊处理）───
        has_meshes = False
        if hasattr(model, "shape_type") and model.shape_type is not None:
            shape_types = model.shape_type.numpy()
            has_meshes = bool((shape_types == int(GeoType.MESH)).any())

        shape_world = getattr(model, "shape_world", None)  # 每个形状属于哪个世界
        shape_flags = getattr(model, "shape_flags", None)  # 形状的碰撞标志

        self.model = model
        self.shape_count = shape_count
        self.broad_phase_mode = broad_phase_mode
        self.device = device
        self.reduce_contacts = reduce_contacts
        # 最大可能的碰撞对数 = C(N, 2) = N*(N-1)/2
        self.shape_pairs_max = (shape_count * (shape_count - 1)) // 2

        # ─── 第6步：构建碰撞排除列表（NXN/SAP 模式用）───
        # 某些形状对不应该碰撞（比如同一个关节链上相邻的两个连杆）
        shape_pairs_excluded = None
        if broad_phase_mode in (BroadPhaseMode.NXN, BroadPhaseMode.SAP) and hasattr(
            model, "shape_collision_filter_pairs"
        ):
            filters = model.shape_collision_filter_pairs
            if filters:
                sorted_pairs = sorted(filters)
                shape_pairs_excluded = wp.array(
                    np.array(sorted_pairs),
                    dtype=wp.vec2i,
                    device=model.device,
                )

        self.shape_pairs_excluded = shape_pairs_excluded
        self.shape_pairs_excluded_count = shape_pairs_excluded.shape[0] if shape_pairs_excluded is not None else 0

        # ─── 第7步：初始化宽相检测器（三选一）───
        #
        # 宽相的作用：快速过滤掉不可能碰撞的形状对
        # 方法：比较 AABB（轴对齐包围盒），不重叠就不可能碰撞
        #
        #   NXN:      遍历所有对                → O(N²)，简单但慢
        #   SAP:      沿轴排序后扫描             → O(N log N)，大场景更快
        #   EXPLICIT: 直接用预计算好的碰撞对列表  → O(K)，K 是碰撞对数，最快
        if self.broad_phase_mode == BroadPhaseMode.NXN:
            if shape_world is None:
                raise ValueError("shape_world must be provided when using BroadPhaseMode.NXN")
            self.nxn_broadphase = BroadPhaseAllPairs(shape_world, shape_flags=shape_flags, device=device)
            self.sap_broadphase = None
            self.explicit_broadphase = None
            self.shape_pairs_filtered = None
        elif self.broad_phase_mode == BroadPhaseMode.SAP:
            if shape_world is None:
                raise ValueError("shape_world must be provided when using BroadPhaseMode.SAP")
            self.sap_broadphase = BroadPhaseSAP(
                shape_world,
                shape_flags=shape_flags,
                sort_type=sap_sort_type,
                device=device,
            )
            self.nxn_broadphase = None
            self.explicit_broadphase = None
            self.shape_pairs_filtered = None
        else:  # BroadPhaseMode.EXPLICIT
            if shape_pairs_filtered is None:
                raise ValueError("shape_pairs_filtered must be provided when using EXPLICIT mode")
            self.explicit_broadphase = BroadPhaseExplicit()
            self.nxn_broadphase = None
            self.sap_broadphase = None
            self.shape_pairs_filtered = shape_pairs_filtered
            self.shape_pairs_max = len(shape_pairs_filtered)

        # ─── 第8步：预分配 GPU 缓冲区 ───
        # 宽相结果：哪些形状对通过了 AABB 重叠测试
        # AABB 缓冲区：每个形状的包围盒下界/上界
        with wp.ScopedDevice(device):
            self.broad_phase_pair_count = wp.zeros(1, dtype=wp.int32, device=device)
            self.broad_phase_shape_pairs = wp.zeros(self.shape_pairs_max, dtype=wp.vec2i, device=device)
            self.shape_aabb_lower = wp.zeros(shape_count, dtype=wp.vec3, device=device)
            self.shape_aabb_upper = wp.zeros(shape_count, dtype=wp.vec3, device=device)

        # ─── 第9步：初始化窄相检测器 ───
        # 窄相对宽相筛选出的候选对做精确碰撞检测（GJK/MPR 算法）
        # 输出：精确的接触点位置、法线、穿透深度
        self.narrow_phase = NarrowPhase(
            max_candidate_pairs=self.shape_pairs_max,
            max_triangle_pairs=1000000,   # 网格碰撞的三角形对数上限
            reduce_contacts=self.reduce_contacts,
            device=device,
            shape_aabb_lower=self.shape_aabb_lower,
            shape_aabb_upper=self.shape_aabb_upper,
            contact_writer_warp_func=write_contact,
            sdf_hydroelastic=sdf_hydroelastic,
            has_meshes=has_meshes,
        )
        self.sdf_hydroelastic = self.narrow_phase.sdf_hydroelastic

        # ─── 第10步：窄相的几何数据缓冲区 ───
        with wp.ScopedDevice(device):
            self.geom_data = wp.zeros(shape_count, dtype=wp.vec4, device=device)       # 几何参数（半径等）
            self.geom_transform = wp.zeros(shape_count, dtype=wp.transform, device=device)  # 世界空间变换

        # ─── 第11步：软接触参数 ───
        if soft_contact_max is None:
            soft_contact_max = shape_count * particle_count
        self.soft_contact_margin = soft_contact_margin
        self.soft_contact_max = soft_contact_max
        self.requires_grad = requires_grad

    def contacts(self) -> Contacts:
        """
        Allocate and return a new :class:`Contacts` object for this pipeline.

        Returns:
            Contacts: A newly allocated contacts buffer sized for this pipeline.
        """
        contacts = Contacts(
            self.rigid_contact_max,
            self.soft_contact_max,
            requires_grad=self.requires_grad,
            device=self.model.device,
            per_contact_shape_properties=self.narrow_phase.sdf_hydroelastic is not None,
            requested_attributes=self.model.get_requested_contact_attributes(),
        )

        # attach custom attributes with assignment==CONTACT
        self.model._add_custom_attributes(contacts, Model.AttributeAssignment.CONTACT, requires_grad=self.requires_grad)
        return contacts

    def collide(
        self,
        state: State,
        contacts: Contacts,
        *,
        soft_contact_margin: float | None = None,
    ):
        """
        Run the collision pipeline using NarrowPhase.

        Args:
            state: The current simulation state.
            contacts: The contacts buffer to populate (will be cleared first).
            soft_contact_margin: Margin for soft contact generation. If None, uses the value from construction.

        """

        contacts.clear()
        # TODO: validate contacts dimensions & compatibility

        # Clear counters
        self.broad_phase_pair_count.zero_()

        model = self.model
        # update any additional parameters
        soft_contact_margin = soft_contact_margin if soft_contact_margin is not None else self.soft_contact_margin

        # When requires_grad, skip rigid contact path so the tape does not record narrow phase
        # kernels (they have enable_backward=False). Only soft contacts are differentiable.
        if not self.requires_grad:
            # Compute AABBs for all shapes (already expanded by per-shape contact margins)
            wp.launch(
                kernel=compute_shape_aabbs,
                dim=model.shape_count,
                inputs=[
                    state.body_q,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_type,
                    model.shape_scale,
                    model.shape_collision_radius,
                    model.shape_source_ptr,
                    model.shape_contact_margin,
                ],
                outputs=[
                    self.shape_aabb_lower,
                    self.shape_aabb_upper,
                ],
                device=self.device,
            )

            # Run broad phase (AABBs are already expanded by contact margins, so pass None)
            if self.broad_phase_mode == BroadPhaseMode.NXN:
                self.nxn_broadphase.launch(
                    self.shape_aabb_lower,
                    self.shape_aabb_upper,
                    None,  # AABBs are pre-expanded, no additional margin needed
                    model.shape_collision_group,
                    model.shape_world,
                    model.shape_count,
                    self.broad_phase_shape_pairs,
                    self.broad_phase_pair_count,
                    device=self.device,
                    filter_pairs=self.shape_pairs_excluded,
                    num_filter_pairs=self.shape_pairs_excluded_count,
                )
            elif self.broad_phase_mode == BroadPhaseMode.SAP:
                self.sap_broadphase.launch(
                    self.shape_aabb_lower,
                    self.shape_aabb_upper,
                    None,  # AABBs are pre-expanded, no additional margin needed
                    model.shape_collision_group,
                    model.shape_world,
                    model.shape_count,
                    self.broad_phase_shape_pairs,
                    self.broad_phase_pair_count,
                    device=self.device,
                    filter_pairs=self.shape_pairs_excluded,
                    num_filter_pairs=self.shape_pairs_excluded_count,
                )
            else:  # BroadPhaseMode.EXPLICIT
                self.explicit_broadphase.launch(
                    self.shape_aabb_lower,
                    self.shape_aabb_upper,
                    None,  # AABBs are pre-expanded, no additional margin needed
                    self.shape_pairs_filtered,
                    len(self.shape_pairs_filtered),
                    self.broad_phase_shape_pairs,
                    self.broad_phase_pair_count,
                    device=self.device,
                )

            # Prepare geometry data arrays for NarrowPhase API
            wp.launch(
                kernel=prepare_geom_data_kernel,
                dim=model.shape_count,
                inputs=[
                    model.shape_transform,
                    model.shape_body,
                    model.shape_type,
                    model.shape_scale,
                    model.shape_thickness,
                    state.body_q,
                ],
                outputs=[
                    self.geom_data,
                    self.geom_transform,
                ],
                device=self.device,
            )

            # Create ContactWriterData struct for custom contact writing
            writer_data = ContactWriterData()
            writer_data.contact_max = contacts.rigid_contact_max
            writer_data.body_q = state.body_q
            writer_data.shape_body = model.shape_body
            writer_data.shape_contact_margin = model.shape_contact_margin
            writer_data.contact_count = contacts.rigid_contact_count
            writer_data.out_shape0 = contacts.rigid_contact_shape0
            writer_data.out_shape1 = contacts.rigid_contact_shape1
            writer_data.out_point0 = contacts.rigid_contact_point0
            writer_data.out_point1 = contacts.rigid_contact_point1
            writer_data.out_offset0 = contacts.rigid_contact_offset0
            writer_data.out_offset1 = contacts.rigid_contact_offset1
            writer_data.out_normal = contacts.rigid_contact_normal
            writer_data.out_thickness0 = contacts.rigid_contact_thickness0
            writer_data.out_thickness1 = contacts.rigid_contact_thickness1
            writer_data.out_tids = contacts.rigid_contact_tids

            writer_data.out_stiffness = contacts.rigid_contact_stiffness
            writer_data.out_damping = contacts.rigid_contact_damping
            writer_data.out_friction = contacts.rigid_contact_friction

            # Run narrow phase with custom contact writer (writes directly to Contacts format)
            self.narrow_phase.launch_custom_write(
                candidate_pair=self.broad_phase_shape_pairs,
                num_candidate_pair=self.broad_phase_pair_count,
                shape_types=model.shape_type,
                shape_data=self.geom_data,
                shape_transform=self.geom_transform,
                shape_source=model.shape_source_ptr,
                shape_sdf_data=model.shape_sdf_data,
                shape_contact_margin=model.shape_contact_margin,
                shape_collision_radius=model.shape_collision_radius,
                shape_flags=model.shape_flags,
                shape_local_aabb_lower=model.shape_local_aabb_lower,
                shape_local_aabb_upper=model.shape_local_aabb_upper,
                shape_voxel_resolution=model.shape_voxel_resolution,
                writer_data=writer_data,
                device=self.device,
            )

        # Generate soft contacts for particles and shapes
        particle_count = len(state.particle_q) if state.particle_q else 0
        if state.particle_q and model.shape_count > 0:
            wp.launch(
                kernel=create_soft_contacts,
                dim=particle_count * model.shape_count,
                inputs=[
                    state.particle_q,
                    model.particle_radius,
                    model.particle_flags,
                    model.particle_world,
                    state.body_q,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_type,
                    model.shape_scale,
                    model.shape_source_ptr,
                    model.shape_world,
                    soft_contact_margin,
                    self.soft_contact_max,
                    model.shape_count,
                    model.shape_flags,
                ],
                outputs=[
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    contacts.soft_contact_tids,
                ],
                device=self.device,
            )

    def get_hydro_contact_surface(self):
        """Get hydroelastic contact surface data for visualization, if available.

        Returns:
            HydroelasticContactSurfaceData if sdf_hydroelastic is configured, None otherwise.
        """
        if self.sdf_hydroelastic is not None:
            return self.sdf_hydroelastic.get_hydro_contact_surface()
        return None

    def set_output_contact_surface(self, enabled: bool) -> None:
        """Enable or disable contact surface visualization.

        Note: When ``output_contact_surface=True`` in the config, the kernel always
        writes debug surface data. This method is provided for API compatibility but
        the actual display is controlled by the viewer's ``show_hydro_contact_surface`` flag.

        Args:
            enabled: If True, visualization is enabled (viewer will display the data).
                     If False, visualization is disabled (viewer will hide the data).
        """
        if self.sdf_hydroelastic is not None:
            self.sdf_hydroelastic.set_output_contact_surface(enabled)
