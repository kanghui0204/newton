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

"""Implementation of the Newton model class."""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import warp as wp

from ..core.types import Devicelike
from .contacts import Contacts
from .control import Control
from .state import State


class Model:
    """
    Represents the static (non-time-varying) definition of a simulation model in Newton.

    The Model class encapsulates all geometry, constraints, and parameters that describe a physical system
    for simulation. It is designed to be constructed via the ModelBuilder, which handles the correct
    initialization and population of all fields.

    Key Features:
        - Stores all static data for simulation: particles, rigid bodies, joints, shapes, soft/rigid elements, etc.
        - Supports grouping of entities by world using world indices (e.g., `particle_world`, `body_world`, etc.).
          - Index -1: global entities shared across all worlds.
          - Indices 0, 1, 2, ...: world-specific entities.
        - Grouping enables:
          - Collision detection optimization (e.g., separating worlds)
          - Visualization (e.g., spatially separating worlds)
          - Parallel processing of independent worlds

    Note:
        It is strongly recommended to use the :class:`ModelBuilder` to construct a Model.
        Direct instantiation and manual population of Model fields is possible but discouraged.
        直接实例化并手动填充 Model 字段是可行的，但不推荐。
    """

    class AttributeAssignment(IntEnum):
        """Enumeration of attribute assignment categories.

        Defines which component of the simulation system owns and manages specific attributes.
        This categorization determines where custom attributes are attached during simulation
        object creation (Model, State, Control, or Contacts).
        """

        MODEL = 0
        """Model attributes are attached to the :class:`~newton.Model` object."""
        STATE = 1
        """State attributes are attached to the :class:`~newton.State` object."""
        CONTROL = 2
        """Control attributes are attached to the :class:`~newton.Control` object."""
        CONTACT = 3
        """Contact attributes are attached to the :class:`~newton.Contacts` object."""

    class AttributeFrequency(IntEnum):
        """Enumeration of attribute frequency categories.

        Defines the dimensional structure and indexing pattern for custom attributes.
        This determines how many elements an attribute array should have and how it
        should be indexed in relation to the model's entities such as joints, bodies, shapes, etc.
        """

        ONCE = 0
        """Attribute frequency is a single value."""
        JOINT = 1
        """Attribute frequency follows the number of joints (see :attr:`~newton.Model.joint_count`)."""
        JOINT_DOF = 2
        """Attribute frequency follows the number of joint degrees of freedom (see :attr:`~newton.Model.joint_dof_count`)."""
        JOINT_COORD = 3
        """Attribute frequency follows the number of joint positional coordinates (see :attr:`~newton.Model.joint_coord_count`)."""
        JOINT_CONSTRAINT = 4
        """Attribute frequency follows the number of joint constraints (see :attr:`~newton.Model.joint_constraint_count`)."""
        BODY = 5
        """Attribute frequency follows the number of bodies (see :attr:`~newton.Model.body_count`)."""
        SHAPE = 6
        """Attribute frequency follows the number of shapes (see :attr:`~newton.Model.shape_count`)."""
        ARTICULATION = 7
        """Attribute frequency follows the number of articulations (see :attr:`~newton.Model.articulation_count`)."""
        EQUALITY_CONSTRAINT = 8
        """Attribute frequency follows the number of equality constraints (see :attr:`~newton.Model.equality_constraint_count`)."""
        PARTICLE = 9
        """Attribute frequency follows the number of particles (see :attr:`~newton.Model.particle_count`)."""
        EDGE = 10
        """Attribute frequency follows the number of edges (see :attr:`~newton.Model.edge_count`)."""
        TRIANGLE = 11
        """Attribute frequency follows the number of triangles (see :attr:`~newton.Model.tri_count`)."""
        TETRAHEDRON = 12
        """Attribute frequency follows the number of tetrahedra (see :attr:`~newton.Model.tet_count`)."""
        SPRING = 13
        """Attribute frequency follows the number of springs (see :attr:`~newton.Model.spring_count`)."""
        CONSTRAINT_MIMIC = 14
        """Attribute frequency follows the number of mimic constraints (see :attr:`~newton.Model.constraint_mimic_count`)."""
        WORLD = 15
        """Attribute frequency follows the number of worlds (see :attr:`~newton.Model.num_worlds`)."""

    class AttributeNamespace:
        """
        A container for namespaced custom attributes.

        Custom attributes are stored as regular instance attributes on this object,
        allowing hierarchical organization of related properties.
        """

        def __init__(self, name: str):
            """Initialize the namespace container.

            Args:
                name: The name of the namespace
            """
            self._name = name

        def __repr__(self):
            """Return a string representation showing the namespace and its attributes."""
            # List all public attributes (not starting with _)
            attrs = [k for k in self.__dict__ if not k.startswith("_")]
            return f"AttributeNamespace('{self._name}', attributes={attrs})"

    def __init__(self, device: Devicelike | None = None):
        """
        Initialize a Model object.
        初始化 Model 对象，分配并设置所有仿真所需的静态数据结构（粒子、刚体、关节、形状等）。

        Args:
            device (wp.Device, optional): Device on which the Model's data will be allocated.
        """
        self.requires_grad = False
        """Whether the model was finalized (see :meth:`ModelBuilder.finalize`) with gradient computation enabled."""
        self.num_worlds = 0
        """Number of worlds added to the ModelBuilder."""

        # =====================================================================
        # 粒子数据（布料/软体的顶点，由 add_particle/add_cloth_grid/add_soft_grid 创建）
        # =====================================================================
        # 粒子是可变形体（布料、软体）的基本元素。
        # 每个粒子有位置、速度、质量、碰撞半径。
        # 粒子之间通过弹簧（XPBD）或三角形/四面体 FEM（SemiImplicit/VBD）连接。
        # inv_mass=0 的粒子是固定的（如布料的固定边 fix_left=True）。
        #
        # 求解器使用方式:
        #   SolverSemiImplicit: particle_q → eval_spring/eval_triangle → particle_f → integrate
        #   SolverXPBD:         particle_q → solve_springs/solve_tetrahedra → 直接修正 particle_q
        # =====================================================================
        self.particle_q = None
        """Particle positions, shape [particle_count, 3], float.
        粒子位置（vec3 数组）。finalize 时从 builder.particle_q 列表转换而来。
        model.state() 会把这个值复制到 State.particle_q 作为初始位置。"""
        self.particle_qd = None
        """Particle velocities, shape [particle_count, 3], float.
        粒子速度。同上，复制到 State.particle_qd。"""
        self.particle_mass = None
        """Particle mass, shape [particle_count], float.
        粒子质量。0 表示固定粒子（不参与动力学，如布料的固定边）。
        布料中由 add_cloth_grid 的 mass 参数按面积加权分配。"""
        self.particle_inv_mass = None
        """Particle inverse mass, shape [particle_count], float.
        粒子质量的倒数。0 = 固定粒子（质量无穷大，不会被力移动）。
        积分时用 inv_mass 而不是 mass，避免除法：a = f * inv_mass。"""
        self.particle_radius = None
        """Particle radius, shape [particle_count], float.
        粒子碰撞半径。用于 create_soft_contacts 中的 SDF 碰撞检测。
        也决定 Viewer 中粒子小球的显示大小。"""
        self.particle_max_radius = 0.0
        """Maximum particle radius (useful for HashGrid construction).
        所有粒子中最大的半径，用于 HashGrid 的格子大小设置。"""
        self.particle_ke = 1.0e3
        """Particle normal contact stiffness (used by :class:`~newton.solvers.SolverSemiImplicit`).
        粒子-粒子接触的法向弹性刚度（只有 SemiImplicit 求解器用）。"""
        self.particle_kd = 1.0e2
        """Particle normal contact damping (used by :class:`~newton.solvers.SolverSemiImplicit`).
        粒子-粒子接触的法向阻尼。"""
        self.particle_kf = 1.0e2
        """Particle friction force stiffness (used by :class:`~newton.solvers.SolverSemiImplicit`).
        粒子-粒子接触的摩擦力刚度。"""
        self.particle_mu = 0.5
        """Particle friction coefficient.
        粒子摩擦系数。"""
        self.particle_cohesion = 0.0
        """Particle cohesion strength.
        粒子间粘聚力（让粒子"粘"在一起，用于模拟湿沙等）。"""
        self.particle_adhesion = 0.0
        """Particle adhesion strength.
        粒子-形状粘附力（让粒子"粘"到碰撞体表面）。"""
        self.particle_grid: wp.HashGrid | None = None
        """HashGrid instance for accelerated simulation of particle interactions.
        空间哈希网格，用于加速粒子-粒子邻域查询。
        格子大小 = 2 * particle_max_radius，只查询相邻格子内的粒子。"""
        self.particle_flags: wp.array | None = None
        """Particle enabled state, shape [particle_count], int.
        粒子标志位（ACTIVE=1 表示参与仿真，0 表示禁用/休眠）。"""
        self.particle_max_velocity: float = 1e5
        """Maximum particle velocity (to prevent instability).
        粒子最大速度限制。超过此值会被截断，防止数值不稳定导致"爆炸"。"""
        self.particle_world: wp.array | None = None
        """World index for each particle, shape [particle_count], int. -1 for global. 粒子所属世界编号（-1=全局共享）。"""
        self.particle_world_start = None
        """Start index of the first particle per world, shape [num_worlds + 2], int.

        The entries at indices `0` to `num_worlds - 1` store the start index of the particles belonging to that world.
        The second-last element (accessible via index `-2`) stores the start index of the global particles (i.e. with
        world index `-1`) added to the end of the model, and the last element stores the total particle count.

        The number of particles in a given world `w` can be computed as:
            `num_particles_in_world = particle_world_start[w + 1] - particle_world_start[w]`.

        The total number of global particles can be computed as:
            `num_global_particles = particle_world_start[-1] - particle_world_start[-2] + particle_world_start[0]`.
        """

        # =====================================================================
        # 碰撞形状数据（由 add_shape_sphere/box/capsule/mesh 等创建）
        # =====================================================================
        # Shape 是碰撞几何体，附加在 Body 上。
        # Body 是物理实体（有质量、惯性），Shape 是它的"外壳"（用于碰撞检测）。
        # 一个 Body 可以有多个 Shape（如机器人的腿由胶囊+球组合）。
        # shape_body=-1 的形状是固定到世界的（如地面、固定障碍物）。
        #
        # 碰撞检测流程中 Shape 数据的使用:
        #   1. compute_shape_aabbs() 读 shape_transform + body_q → 算世界空间 AABB
        #   2. 宽相用 AABB 找候选碰撞对
        #   3. 窄相读 shape_type + shape_scale + shape_source_ptr → 精确碰撞
        #   4. 接触力计算读 shape_material_* → 算弹性/阻尼/摩擦力
        # =====================================================================
        self.shape_key = []
        """List of keys for each shape. 形状的标签名（调试用，如 "ground", "left_wheel"）。"""
        self.shape_transform = None
        """Rigid shape transforms, shape [shape_count, 7], float.
        形状在所属 body 局部坐标系中的变换（transform = 位置+旋转）。
        碰撞检测时：世界变换 = body_q[shape_body[i]] × shape_transform[i]。"""
        self.shape_body = None
        """Rigid shape body index, shape [shape_count], int.
        形状附加到哪个 body。-1 = 固定到世界（如地面）。
        碰撞检测需要知道形状属于哪个 body 来计算世界坐标。"""
        self.shape_flags = None
        """Rigid shape flags, shape [shape_count], int.
        形状标志位（是否参与碰撞、是否参与粒子碰撞等）。"""
        self.body_shapes = {}
        """Mapping from body index to list of attached shape indices.
        从 body ID 到其所有 shape ID 列表的映射。"""

        # Shape material properties
        # 形状材质参数（碰撞力计算时使用）
        self.shape_material_ke = None
        """Shape contact elastic stiffness, shape [shape_count], float.
        接触弹性刚度（碰撞时的"硬度"，越大接触力越强，穿透越小）。"""
        self.shape_material_kd = None
        """Shape contact damping stiffness, shape [shape_count], float."""
        self.shape_material_kf = None
        """Shape contact friction stiffness, shape [shape_count], float."""
        self.shape_material_ka = None
        """Shape contact adhesion distance, shape [shape_count], float."""
        self.shape_material_mu = None
        """Shape coefficient of friction, shape [shape_count], float."""
        self.shape_material_restitution = None
        """Shape coefficient of restitution, shape [shape_count], float."""
        self.shape_material_torsional_friction = None
        """Shape torsional friction coefficient (resistance to spinning at contact point), shape [shape_count], float."""
        self.shape_material_rolling_friction = None
        """Shape rolling friction coefficient (resistance to rolling motion), shape [shape_count], float."""
        self.shape_material_k_hydro = None
        """Shape hydroelastic stiffness coefficient, shape [shape_count], float."""
        self.shape_contact_margin = None
        """Shape contact margin for collision detection, shape [shape_count], float."""

        # Shape geometry properties
        self.shape_type = None
        """Shape geometry type, shape [shape_count], int32."""
        self.shape_is_solid = None
        """Whether shape is solid or hollow, shape [shape_count], bool."""
        self.shape_thickness = None
        """Shape thickness, shape [shape_count], float."""
        self.shape_source = []
        """List of source geometry objects (e.g., :class:`~newton.Mesh`, :class:`~newton.SDF`) used for rendering and broadphase, shape [shape_count]."""
        self.shape_source_ptr = None
        """Geometry source pointer to be used inside the Warp kernels which can be generated by finalizing the geometry objects, see for example :meth:`newton.Mesh.finalize`, shape [shape_count], uint64."""
        self.shape_scale = None
        """Shape 3D scale, shape [shape_count, 3], float."""
        self.shape_filter = None
        """Shape filter group, shape [shape_count], int."""

        self.shape_collision_group = None
        """Collision group of each shape, shape [shape_count], int. Array populated during finalization."""
        self.shape_collision_filter_pairs: set[tuple[int, int]] = set()
        """Pairs of shape indices (s1, s2) that should not collide. Pairs are in canonical order: s1 < s2."""
        self.shape_collision_radius = None
        """Collision radius for bounding sphere broadphase, shape [shape_count], float. Not supported by :class:`~newton.solvers.SolverMuJoCo`."""
        self.shape_contact_pairs = None
        """Pairs of shape indices that may collide, shape [contact_pair_count, 2], int."""
        self.shape_contact_pair_count = 0
        """Number of shape contact pairs."""
        self.shape_world = None
        """World index for each shape, shape [shape_count], int. -1 for global."""
        self.shape_world_start = None
        """Start index of the first shape per world, shape [num_worlds + 2], int.

        The entries at indices `0` to `num_worlds - 1` store the start index of the shapes belonging to that world.
        The second-last element (accessible via index `-2`) stores the start index of the global shapes (i.e. with
        world index `-1`) added to the end of the model, and the last element stores the total shape count.

        The number of shapes in a given world `w` can be computed as:
            `num_shapes_in_world = shape_world_start[w + 1] - shape_world_start[w]`.

        The total number of global shapes can be computed as:
            `num_global_shapes = shape_world_start[-1] - shape_world_start[-2] + shape_world_start[0]`.
        """

        # Mesh SDF storage
        self.shape_sdf_data = None
        """Array of SDFData structs for mesh shapes, shape [shape_count]. Contains sparse and coarse SDF pointers, extents, and voxel sizes. Empty array if there are no colliding meshes."""
        self.shape_sdf_volume = []
        """List of sparse SDF volume references for mesh shapes, shape [shape_count]. None for non-mesh shapes. Empty if there are no colliding meshes. Kept for reference counting."""
        self.shape_sdf_coarse_volume = []
        """List of coarse SDF volume references for mesh shapes, shape [shape_count]. None for non-mesh shapes. Empty if there are no colliding meshes. Kept for reference counting."""

        # Local AABB and voxel grid for contact reduction
        # Note: These are stored in Model (not Contacts) because they are static geometry properties
        # computed once during finalization, not per-frame contact data.
        self.shape_local_aabb_lower = None
        """Local-space AABB lower bound for each shape, shape [shape_count, 3], float.
        Computed from base geometry only (excludes thickness - thickness is added during contact
        margin calculations). Used for voxel-based contact reduction."""
        self.shape_local_aabb_upper = None
        """Local-space AABB upper bound for each shape, shape [shape_count, 3], float.
        Computed from base geometry only (excludes thickness - thickness is added during contact
        margin calculations). Used for voxel-based contact reduction."""
        self.shape_voxel_resolution = None
        """Voxel grid resolution (nx, ny, nz) for each shape, shape [shape_count, 3], int. Used for voxel-based contact reduction."""

        # =====================================================================
        # 弹簧数据（XPBD 求解器用，由 add_cloth_grid(add_springs=True) 创建）
        # =====================================================================
        # 弹簧连接两个粒子，约束它们之间的距离保持在 rest_length。
        # XPBD 求解器逐个弹簧检查"拉长了没"，拉长了就修正位置。
        # SemiImplicit/VBD 不用弹簧，用三角形 FEM 力代替。
        # =====================================================================
        self.spring_indices = None
        """Particle spring indices, shape [spring_count*2], int.
        弹簧端点索引，每根弹簧 2 个粒子 ID：[i0,j0, i1,j1, ...]。"""
        self.spring_rest_length = None
        """Particle spring rest length, shape [spring_count], float.
        弹簧的原始长度。当前长度 ≠ 原始长度时产生恢复力。"""
        self.spring_stiffness = None
        """Particle spring stiffness, shape [spring_count], float.
        弹簧刚度 ke。力 = ke × (当前长度 - 原始长度)。"""
        self.spring_damping = None
        """Particle spring damping, shape [spring_count], float.
        弹簧阻尼 kd。阻止弹簧剧烈振荡。"""
        self.spring_control = None
        """Particle spring activation, shape [spring_count], float.
        弹簧激活量（用于肌肉驱动）。"""
        self.spring_constraint_lambdas = None
        """Lagrange multipliers for spring constraints (internal use).
        XPBD 内部的拉格朗日乘数（约束求解的中间量）。"""

        # =====================================================================
        # 三角形数据（布料 FEM，由 add_cloth_grid 创建）
        # =====================================================================
        # 三角形是布料的 2D 有限元。
        # 每个三角形有 3 个顶点（粒子），存储原始形状（pose）用于计算变形梯度 F。
        # SemiImplicit 和 VBD 用 tri_materials 里的 ke/ka/kd 算 FEM 力。
        # XPBD 不直接用三角形数据（它用弹簧代替）。
        # =====================================================================
        self.tri_indices = None
        """Triangle element indices, shape [tri_count*3], int.
        三角形顶点索引：[v0,v1,v2, v0,v1,v2, ...]。"""
        self.tri_poses = None
        """Triangle element rest pose, shape [tri_count, 2, 2], float.
        三角形原始形状矩阵的逆 (Dm_inv)。用于计算变形梯度 F = Ds × Dm_inv。"""
        self.tri_activations = None
        """Triangle element activations, shape [tri_count], float.
        三角形激活力（肌肉驱动的布料用）。"""
        self.tri_materials = None
        """Triangle element materials, shape [tri_count, 5], float.
        三角形材质参数 [ke, ka, kd, drag, lift]。
        ke=膜刚度, ka=面积保持, kd=阻尼, drag=空气阻力, lift=升力。"""
        self.tri_areas = None
        """Triangle element rest areas, shape [tri_count], float.
        三角形原始面积，用于面积保持约束。"""

        # =====================================================================
        # 弯曲边数据（布料弯曲约束，由 add_cloth_grid 自动创建）
        # =====================================================================
        # 边连接两个共享边的三角形，约束它们之间的二面角。
        # 没有弯曲约束的布料完全没有弯曲抵抗力（像纸一样随意折叠）。
        # 有弯曲约束的布料会抵抗弯折（像硬纸板）。
        # =====================================================================
        self.edge_indices = None
        """Bending edge indices, shape [edge_count*4], int, each row is [o0, o1, v1, v2], where v1, v2 are on the edge.
        弯曲边索引：[o0, o1, v1, v2]。o0,o1 是两个三角形的"对面顶点"，v1,v2 在共享边上。"""
        self.edge_rest_angle = None
        """Bending edge rest angle, shape [edge_count], float.
        边的原始二面角。当前角度偏离原始角度时产生弯曲恢复力。"""
        self.edge_rest_length = None
        """Bending edge rest length, shape [edge_count], float.
        共享边的原始长度。"""
        self.edge_bending_properties = None
        """Bending edge stiffness and damping, shape [edge_count, 2], float.
        弯曲刚度和阻尼 [ke, kd]。ke 越大布料越"硬"（不容易弯折）。"""
        self.edge_constraint_lambdas = None
        """Lagrange multipliers for edge constraints (internal use).
        XPBD 内部使用。"""

        # =====================================================================
        # 四面体数据（软体 3D FEM，由 add_soft_grid 创建）
        # =====================================================================
        # 四面体是软体的 3D 有限元（类似三角形是布料的 2D 有限元）。
        # 每个四面体有 4 个顶点，存储原始形状用于计算 3D 变形梯度。
        # 材质模型通常是 Neo-Hookean（超弹性）。
        # =====================================================================
        self.tet_indices = None
        """Tetrahedral element indices, shape [tet_count*4], int.
        四面体顶点索引：[v0,v1,v2,v3, ...]。"""
        self.tet_poses = None
        """Tetrahedral rest poses, shape [tet_count, 3, 3], float.
        四面体原始形状矩阵的逆 (Dm_inv)。F = Ds × Dm_inv。"""
        self.tet_activations = None
        """Tetrahedral volumetric activations, shape [tet_count], float.
        四面体激活力（人工肌肉驱动的软体用）。"""
        self.tet_materials = None
        """Tetrahedral elastic parameters in form :math:`k_{mu}, k_{lambda}, k_{damp}`, shape [tet_count, 3].
        四面体材质 [k_mu, k_lambda, k_damp]。k_mu=剪切模量, k_lambda=体积模量, k_damp=阻尼。"""

        self.muscle_start = None
        """Start index of the first muscle point per muscle, shape [muscle_count], int."""
        self.muscle_params = None
        """Muscle parameters, shape [muscle_count, 5], float."""
        self.muscle_bodies = None
        """Body indices of the muscle waypoints, int."""
        self.muscle_points = None
        """Local body offset of the muscle waypoints, float."""
        self.muscle_activations = None
        """Muscle activations, shape [muscle_count], float."""

        # =====================================================================
        # 刚体数据（由 add_body/add_link 创建）
        # =====================================================================
        # 刚体是不变形的物理实体（盒子、球、机器人连杆等）。
        # 每个刚体有 7 个位姿参数（位置3 + 四元数旋转4）和 6 个速度参数（线3 + 角3）。
        # body_q 和 body_qd 是初始值，model.state() 会复制它们到 State 中。
        # 仿真过程中只修改 State 中的 body_q/body_qd，Model 中的不变。
        #
        # inv_mass=0 的刚体是"运动学体"（kinematic body），不受力影响，只能通过
        # 直接设置 body_q 来移动（如传送带、抓手的固定部分）。
        #
        # 积分时使用 inv_mass 和 inv_inertia 而不是 mass 和 inertia，
        # 避免除法运算：加速度 = 力 × inv_mass。
        # =====================================================================
        self.body_q = None
        """Rigid body poses for state initialization, shape [body_count, 7], float.
        刚体初始位姿。transform = (位置 vec3, 旋转 quat)，共 7 个数。
        model.state() 会把这个值复制到 State.body_q 作为仿真起点。"""
        self.body_qd = None
        """Rigid body velocities for state initialization, shape [body_count, 6], float.
        刚体初始速度。spatial_vector = (线速度 vec3, 角速度 vec3)，共 6 个数。"""
        self.body_com = None
        """Rigid body center of mass (in local frame), shape [body_count, 3], float.
        刚体质心在局部坐标系中的偏移。质心不一定在几何中心
        （比如 L 形物体的质心在 L 的弯角处）。
        积分时所有力矩都相对于质心计算。"""
        self.body_inertia = None
        """Rigid body inertia tensor (relative to COM), shape [body_count, 3, 3], float.
        刚体惯性张量（3×3 矩阵，相对于质心，在体坐标系下）。
        描述物体质量如何分布——决定物体绕不同轴旋转的"阻力"。
        球的惯性张量是对角的（各方向一样），长杆的惯性张量差异很大。"""
        self.body_inv_inertia = None
        """Rigid body inverse inertia tensor (relative to COM), shape [body_count, 3, 3], float.
        惯性张量的逆。积分时用：角加速度 = inv_inertia × 力矩。"""
        self.body_mass = None
        """Rigid body mass, shape [body_count], float.
        刚体质量（kg）。由 shape 的密度 × 体积自动计算（finalize 时）。"""
        self.body_inv_mass = None
        """Rigid body inverse mass, shape [body_count], float.
        刚体质量的倒数。0 = 固定体/运动学体（质量无穷大，不受力影响）。"""
        self.body_key = []
        """Rigid body keys, shape [body_count], str."""
        self.body_world = None
        """World index for each body, shape [body_count], int. Global entities have index -1."""
        self.body_world_start = None
        """Start index of the first body per world, shape [num_worlds + 2], int.

        The entries at indices `0` to `num_worlds - 1` store the start index of the bodies belonging to that world.
        The second-last element (accessible via index `-2`) stores the start index of the global bodies (i.e. with
        world index `-1`) added to the end of the model, and the last element stores the total body count.

        The number of bodies in a given world `w` can be computed as:
            `num_bodies_in_world = body_world_start[w + 1] - body_world_start[w]`.

        The total number of global bodies can be computed as:
            `num_global_bodies = body_world_start[-1] - body_world_start[-2] + body_world_start[0]`.
        """

        # =====================================================================
        # 关节数据（由 add_joint_revolute/prismatic/ball/free/fixed 创建）
        # =====================================================================
        # 关节连接两个刚体，限制它们之间的相对运动。
        #
        # 【关节坐标的索引系统】（理解这个很重要！）
        #   所有关节的坐标拼接在一个扁平数组里：
        #
        #   例: 3 个关节 [旋转(1坐标), 球(4坐标), 自由(7坐标)]
        #     joint_q = [θ₀, qx,qy,qz,qw, x,y,z,qx,qy,qz,qw]  ← 共 12 个数
        #     joint_q_start = [0, 1, 5, 12]  ← 关节 i 从 joint_q_start[i] 开始
        #
        #   同理 joint_qd 存速度（自由度数可能和坐标数不同，如球关节 4 坐标 3 自由度）：
        #     joint_qd_start = [0, 1, 4, 10]
        #
        # 【关节在 Featherstone 求解器中的使用】
        #   Featherstone 直接在关节坐标空间积分（不用 body_q），
        #   然后通过正运动学(FK) joint_q → body_q 恢复刚体位姿。
        #
        # 【关节在 SemiImplicit/XPBD 求解器中的使用】
        #   这些求解器在 body_q 空间积分，用关节力（弹簧+阻尼）约束关节运动。
        # =====================================================================
        self.joint_q = None
        """Generalized joint positions for state initialization, shape [joint_coord_count], float.
        关节广义坐标（角度/位移），所有关节按顺序拼接成一个扁平数组。
        用 joint_q_start[i] 找到关节 i 的起始位置。"""
        self.joint_qd = None
        """Generalized joint velocities for state initialization, shape [joint_dof_count], float.
        关节广义速度，所有关节按顺序拼接。用 joint_qd_start[i] 找到关节 i 的起始位置。"""
        self.joint_f = None
        """Generalized joint forces for state initialization, shape [joint_dof_count], float.
        关节力/力矩的初始值。"""
        self.joint_target_pos = None
        """Generalized joint position targets, shape [joint_dof_count], float.
        关节目标位置。PD 控制器用：力 = ke*(target - current) - kd*velocity。"""
        self.joint_target_vel = None
        """Generalized joint velocity targets, shape [joint_dof_count], float.
        关节目标速度。"""
        self.joint_type = None
        """Joint type, shape [joint_count], int.
        关节类型编号（REVOLUTE=0, PRISMATIC=1, BALL=2, FREE=3, FIXED=4, ...）。
        每种类型有不同的自由度数和坐标数。"""
        self.joint_articulation = None
        """Joint articulation index (-1 if not in any articulation), shape [joint_count], int.
        关节所属的 articulation（关节链）编号。-1 = 不属于任何关节链。"""
        self.joint_parent = None
        """Joint parent body indices, shape [joint_count], int.
        关节的父体 ID。-1 = 世界（固定参考系）。
        物理引擎保证 parent_xform 的点和 child_xform 的点始终重合。"""
        self.joint_child = None
        """Joint child body indices, shape [joint_count], int.
        关节的子体 ID。子体通过关节"挂"在父体上。"""
        self.joint_ancestor = None
        """Maps from joint index to the index of the joint that has the current joint parent body as child (-1 if no such joint ancestor exists), shape [joint_count], int."""
        self.joint_X_p = None
        """Joint transform in parent frame, shape [joint_count, 7], float. 关节在父体上的安装位置。"""
        self.joint_X_c = None
        """Joint mass frame in child frame, shape [joint_count, 7], float. 关节在子体上的安装位置。"""
        self.joint_axis = None
        """Joint axis in child frame, shape [joint_dof_count, 3], float."""
        self.joint_armature = None
        """Armature for each joint axis (used by :class:`~newton.solvers.SolverMuJoCo` and :class:`~newton.solvers.SolverFeatherstone`), shape [joint_dof_count], float."""
        self.joint_act_mode = None
        """Actuator mode per DOF, see :class:`newton.ActuatorMode`. Shape [joint_dof_count], dtype int32."""
        self.joint_target_ke = None
        """Joint stiffness, shape [joint_dof_count], float."""
        self.joint_target_kd = None
        """Joint damping, shape [joint_dof_count], float."""
        self.joint_effort_limit = None
        """Joint effort (force/torque) limits, shape [joint_dof_count], float."""
        self.joint_velocity_limit = None
        """Joint velocity limits, shape [joint_dof_count], float."""
        self.joint_friction = None
        """Joint friction coefficient, shape [joint_dof_count], float."""
        self.joint_dof_dim = None
        """Number of linear and angular dofs per joint, shape [joint_count, 2], int."""
        self.joint_enabled = None
        """Controls which joint is simulated (bodies become disconnected if False, only supported by :class:`~newton.solvers.SolverXPBD` and :class:`~newton.solvers.SolverSemiImplicit`), shape [joint_count], bool."""
        self.joint_limit_lower = None
        """Joint lower position limits, shape [joint_dof_count], float."""
        self.joint_limit_upper = None
        """Joint upper position limits, shape [joint_dof_count], float."""
        self.joint_limit_ke = None
        """Joint position limit stiffness (used by :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverFeatherstone`), shape [joint_dof_count], float."""
        self.joint_limit_kd = None
        """Joint position limit damping (used by :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverFeatherstone`), shape [joint_dof_count], float."""
        self.joint_twist_lower = None
        """Joint lower twist limit, shape [joint_count], float."""
        self.joint_twist_upper = None
        """Joint upper twist limit, shape [joint_count], float."""
        self.joint_q_start = None
        """Start index of the first position coordinate per joint (last value is a sentinel for dimension queries), shape [joint_count + 1], int."""
        self.joint_qd_start = None
        """Start index of the first velocity coordinate per joint (last value is a sentinel for dimension queries), shape [joint_count + 1], int."""
        self.joint_key = []
        """Joint keys, shape [joint_count], str."""
        self.joint_world = None
        """World index for each joint, shape [joint_count], int. -1 for global."""
        self.joint_world_start = None
        """Start index of the first joint per world, shape [num_worlds + 2], int.

        The entries at indices `0` to `num_worlds - 1` store the start index of the joints belonging to that world.
        The second-last element (accessible via index `-2`) stores the start index of the global joints (i.e. with
        world index `-1`) added to the end of the model, and the last element stores the total joint count.

        The number of joints in a given world `w` can be computed as:
            `num_joints_in_world = joint_world_start[w + 1] - joint_world_start[w]`.

        The total number of global joints can be computed as:
            `num_global_joints = joint_world_start[-1] - joint_world_start[-2] + joint_world_start[0]`.
        """
        self.joint_dof_world_start = None
        """Start index of the first joint degree of freedom per world, shape [num_worlds + 2], int.

        The entries at indices `0` to `num_worlds - 1` store the start index of the joint DOFs belonging to that world.
        The second-last element (accessible via index `-2`) stores the start index of the global joint DOFs (i.e. with
        world index `-1`) added to the end of the model, and the last element stores the total joint DOF count.

        The number of joint DOFs in a given world `w` can be computed as:
            `num_joint_dofs_in_world = joint_dof_world_start[w + 1] - joint_dof_world_start[w]`.

        The total number of global joint DOFs can be computed as:
            `num_global_joint_dofs = joint_dof_world_start[-1] - joint_dof_world_start[-2] + joint_dof_world_start[0]`.
        """
        self.joint_coord_world_start = None
        """Start index of the first joint coordinate per world, shape [num_worlds + 2], int.

        The entries at indices `0` to `num_worlds - 1` store the start index of the joint coordinates belonging to that world.
        The second-last element (accessible via index `-2`) stores the start index of the global joint coordinates (i.e. with
        world index `-1`) added to the end of the model, and the last element stores the total joint coordinate count.

        The number of joint coordinates in a given world `w` can be computed as:
            `num_joint_coords_in_world = joint_coord_world_start[w + 1] - joint_coord_world_start[w]`.

        The total number of global joint coordinates can be computed as:
            `num_global_joint_coords = joint_coord_world_start[-1] - joint_coord_world_start[-2] + joint_coord_world_start[0]`.
        """
        self.joint_constraint_world_start = None
        """Start index of the first joint constraint per world, shape [num_worlds + 2], int.

        The entries at indices `0` to `num_worlds - 1` store the start index of the joint constraints belonging to that world.
        The second-last element (accessible via index `-2`) stores the start index of the global joint constraints (i.e. with
        world index `-1`) added to the end of the model, and the last element stores the total joint constraint count.

        The number of joint constraints in a given world `w` can be computed as:
            `num_joint_constraints_in_world = joint_constraint_world_start[w + 1] - joint_constraint_world_start[w]`.

        The total number of global joint constraints can be computed as:
            `num_global_joint_constraints = joint_constraint_world_start[-1] - joint_constraint_world_start[-2] + joint_constraint_world_start[0]`.
        """

        self.articulation_start = None
        """Articulation start index, shape [articulation_count], int."""
        self.articulation_key = []
        """Articulation keys, shape [articulation_count], str."""
        self.articulation_world = None
        """World index for each articulation, shape [articulation_count], int. -1 for global."""
        self.articulation_world_start = None
        """Start index of the first articulation per world, shape [num_worlds + 2], int.

        The entries at indices `0` to `num_worlds - 1` store the start index of the articulations belonging to that world.
        The second-last element (accessible via index `-2`) stores the start index of the global articulations (i.e. with
        world index `-1`) added to the end of the model, and the last element stores the total articulation count.

        The number of articulations in a given world `w` can be computed as:
            `num_articulations_in_world = articulation_world_start[w + 1] - articulation_world_start[w]`.

        The total number of global articulations can be computed as:
            `num_global_articulations = articulation_world_start[-1] - articulation_world_start[-2] + articulation_world_start[0]`.
        """
        self.max_joints_per_articulation = 0
        """Maximum number of joints in any articulation (used for IK kernel dimensioning)."""
        self.max_dofs_per_articulation = 0
        """Maximum number of degrees of freedom in any articulation (used for Jacobian/mass matrix computation)."""

        self.soft_contact_ke = 1.0e3
        """Stiffness of soft contacts (used by :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverFeatherstone`)."""
        self.soft_contact_kd = 10.0
        """Damping of soft contacts (used by :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverFeatherstone`)."""
        self.soft_contact_kf = 1.0e3
        """Stiffness of friction force in soft contacts (used by :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverFeatherstone`)."""
        self.soft_contact_mu = 0.5
        """Friction coefficient of soft contacts."""
        self.soft_contact_restitution = 0.0
        """Restitution coefficient of soft contacts (used by :class:`SolverXPBD`)."""

        self.rigid_contact_max = 0
        """Number of potential contact points between rigid bodies."""

        self.up_axis = 2
        """Up axis: 0 for x, 1 for y, 2 for z."""
        self.gravity = None
        """Gravity vector, shape [1], dtype vec3."""

        self.equality_constraint_type = None
        """Type of equality constraint, shape [equality_constraint_count], int."""
        self.equality_constraint_body1 = None
        """First body index, shape [equality_constraint_count], int."""
        self.equality_constraint_body2 = None
        """Second body index, shape [equality_constraint_count], int."""
        self.equality_constraint_anchor = None
        """Anchor point on first body, shape [equality_constraint_count, 3], float."""
        self.equality_constraint_torquescale = None
        """Torque scale, shape [equality_constraint_count], float."""
        self.equality_constraint_relpose = None
        """Relative pose, shape [equality_constraint_count, 7], float."""
        self.equality_constraint_joint1 = None
        """First joint index, shape [equality_constraint_count], int."""
        self.equality_constraint_joint2 = None
        """Second joint index, shape [equality_constraint_count], int."""
        self.equality_constraint_polycoef = None
        """Polynomial coefficients, shape [equality_constraint_count, 2], float."""
        self.equality_constraint_key = []
        """Constraint name/key, shape [equality_constraint_count], str."""
        self.equality_constraint_enabled = None
        """Whether constraint is active, shape [equality_constraint_count], bool."""
        self.equality_constraint_world = None
        """World index for each constraint, shape [equality_constraint_count], int."""
        self.equality_constraint_world_start = None
        """Start index of the first equality constraint per world, shape [num_worlds + 2], int.

        The entries at indices `0` to `num_worlds - 1` store the start index of the equality constraints belonging to that world.
        The second-last element (accessible via index `-2`) stores the start index of the global equality constraints (i.e. with
        world index `-1`) added to the end of the model, and the last element stores the total equality constraint count.

        The number of equality constraints in a given world `w` can be computed as:
            `num_equality_constraints_in_world = equality_constraint_world_start[w + 1] - equality_constraint_world_start[w]`.

        The total number of global equality constraints can be computed as:
            `num_global_equality_constraints = equality_constraint_world_start[-1] - equality_constraint_world_start[-2] + equality_constraint_world_start[0]`.
        """

        self.constraint_mimic_joint0 = None
        """Follower joint index (``joint0 = coef0 + coef1 * joint1``), shape [constraint_mimic_count], int."""
        self.constraint_mimic_joint1 = None
        """Leader joint index (``joint0 = coef0 + coef1 * joint1``), shape [constraint_mimic_count], int."""
        self.constraint_mimic_coef0 = None
        """Offset coefficient (coef0) for the mimic constraint (``joint0 = coef0 + coef1 * joint1``), shape [constraint_mimic_count], float."""
        self.constraint_mimic_coef1 = None
        """Scale coefficient (coef1) for the mimic constraint (``joint0 = coef0 + coef1 * joint1``), shape [constraint_mimic_count], float."""
        self.constraint_mimic_enabled = None
        """Whether constraint is active, shape [constraint_mimic_count], bool."""
        self.constraint_mimic_key = []
        """Constraint name/key, shape [constraint_mimic_count], str."""
        self.constraint_mimic_world = None
        """World index for each constraint, shape [constraint_mimic_count], int."""

        self.particle_count = 0
        """Total number of particles in the system."""
        self.body_count = 0
        """Total number of bodies in the system."""
        self.shape_count = 0
        """Total number of shapes in the system."""
        self.joint_count = 0
        """Total number of joints in the system."""
        self.tri_count = 0
        """Total number of triangles in the system."""
        self.tet_count = 0
        """Total number of tetrahedra in the system."""
        self.edge_count = 0
        """Total number of edges in the system."""
        self.spring_count = 0
        """Total number of springs in the system."""
        self.muscle_count = 0
        """Total number of muscles in the system."""
        self.articulation_count = 0
        """Total number of articulations in the system."""
        self.joint_dof_count = 0
        """Total number of velocity degrees of freedom of all joints. Equals the number of joint axes."""
        self.joint_coord_count = 0
        """Total number of position degrees of freedom of all joints."""
        self.joint_constraint_count = 0
        """Total number of joint constraints of all joints."""
        self.equality_constraint_count = 0
        """Total number of equality constraints in the system."""
        self.constraint_mimic_count = 0
        """Total number of mimic constraints in the system."""

        # indices of particles sharing the same color
        self.particle_color_groups = []
        """Coloring of all particles for Gauss-Seidel iteration (see :class:`~newton.solvers.SolverVBD`). Each array contains indices of particles sharing the same color."""
        self.particle_colors = None
        """Color assignment for every particle."""

        self.body_color_groups = []
        """Coloring of all rigid bodies for Gauss-Seidel iteration (see :class:`~newton.solvers.SolverVBD`). Each array contains indices of bodies sharing the same color."""
        self.body_colors = None
        """Color assignment for every rigid body."""

        self.device = wp.get_device(device)
        """Device on which the Model was allocated."""

        self.attribute_frequency: dict[str, Model.AttributeFrequency | str] = {}
        """Classifies each attribute using Model.AttributeFrequency enum values (per body, per joint, per DOF, etc.)
        or custom frequencies for custom entity types (e.g., ``"mujoco:pair"``)."""

        self.custom_frequency_counts: dict[str, int] = {}
        """Counts for custom frequencies (e.g., ``{"mujoco:pair": 5}``). Set during finalize()."""

        self.attribute_assignment: dict[str, Model.AttributeAssignment] = {}
        """Assignment for custom attributes using Model.AttributeAssignment enum values.
        If an attribute is not in this dictionary, it is assumed to be a Model attribute (assignment=Model.AttributeAssignment.MODEL)."""

        self._requested_state_attributes: set[str] = set()
        self._collision_pipeline: CollisionPipeline | None = None
        # cached collision pipeline
        self._requested_contact_attributes: set[str] = set()

        # attributes per body
        self.attribute_frequency["body_q"] = Model.AttributeFrequency.BODY
        self.attribute_frequency["body_qd"] = Model.AttributeFrequency.BODY
        self.attribute_frequency["body_com"] = Model.AttributeFrequency.BODY
        self.attribute_frequency["body_inertia"] = Model.AttributeFrequency.BODY
        self.attribute_frequency["body_inv_inertia"] = Model.AttributeFrequency.BODY
        self.attribute_frequency["body_mass"] = Model.AttributeFrequency.BODY
        self.attribute_frequency["body_inv_mass"] = Model.AttributeFrequency.BODY
        self.attribute_frequency["body_f"] = Model.AttributeFrequency.BODY

        # attributes per joint
        self.attribute_frequency["joint_type"] = Model.AttributeFrequency.JOINT
        self.attribute_frequency["joint_parent"] = Model.AttributeFrequency.JOINT
        self.attribute_frequency["joint_child"] = Model.AttributeFrequency.JOINT
        self.attribute_frequency["joint_ancestor"] = Model.AttributeFrequency.JOINT
        self.attribute_frequency["joint_articulation"] = Model.AttributeFrequency.JOINT
        self.attribute_frequency["joint_X_p"] = Model.AttributeFrequency.JOINT
        self.attribute_frequency["joint_X_c"] = Model.AttributeFrequency.JOINT
        self.attribute_frequency["joint_dof_dim"] = Model.AttributeFrequency.JOINT
        self.attribute_frequency["joint_enabled"] = Model.AttributeFrequency.JOINT
        self.attribute_frequency["joint_twist_lower"] = Model.AttributeFrequency.JOINT
        self.attribute_frequency["joint_twist_upper"] = Model.AttributeFrequency.JOINT

        # attributes per joint coord
        self.attribute_frequency["joint_q"] = Model.AttributeFrequency.JOINT_COORD

        # attributes per joint dof
        self.attribute_frequency["joint_qd"] = Model.AttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_f"] = Model.AttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_armature"] = Model.AttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_target_pos"] = Model.AttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_target_vel"] = Model.AttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_axis"] = Model.AttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_act_mode"] = Model.AttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_target_ke"] = Model.AttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_target_kd"] = Model.AttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_limit_lower"] = Model.AttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_limit_upper"] = Model.AttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_limit_ke"] = Model.AttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_limit_kd"] = Model.AttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_effort_limit"] = Model.AttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_friction"] = Model.AttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_velocity_limit"] = Model.AttributeFrequency.JOINT_DOF

        # attributes per shape
        self.attribute_frequency["shape_transform"] = Model.AttributeFrequency.SHAPE
        self.attribute_frequency["shape_body"] = Model.AttributeFrequency.SHAPE
        self.attribute_frequency["shape_flags"] = Model.AttributeFrequency.SHAPE
        self.attribute_frequency["shape_material_ke"] = Model.AttributeFrequency.SHAPE
        self.attribute_frequency["shape_material_kd"] = Model.AttributeFrequency.SHAPE
        self.attribute_frequency["shape_material_kf"] = Model.AttributeFrequency.SHAPE
        self.attribute_frequency["shape_material_ka"] = Model.AttributeFrequency.SHAPE
        self.attribute_frequency["shape_material_mu"] = Model.AttributeFrequency.SHAPE
        self.attribute_frequency["shape_material_restitution"] = Model.AttributeFrequency.SHAPE
        self.attribute_frequency["shape_material_torsional_friction"] = Model.AttributeFrequency.SHAPE
        self.attribute_frequency["shape_material_rolling_friction"] = Model.AttributeFrequency.SHAPE
        self.attribute_frequency["shape_material_k_hydro"] = Model.AttributeFrequency.SHAPE
        self.attribute_frequency["shape_contact_margin"] = Model.AttributeFrequency.SHAPE
        self.attribute_frequency["shape_type"] = Model.AttributeFrequency.SHAPE
        self.attribute_frequency["shape_is_solid"] = Model.AttributeFrequency.SHAPE
        self.attribute_frequency["shape_thickness"] = Model.AttributeFrequency.SHAPE
        self.attribute_frequency["shape_source_ptr"] = Model.AttributeFrequency.SHAPE
        self.attribute_frequency["shape_scale"] = Model.AttributeFrequency.SHAPE
        self.attribute_frequency["shape_filter"] = Model.AttributeFrequency.SHAPE

    def state(self, requires_grad: bool | None = None) -> State:
        """
        Create and return a new :class:`State` object for this model.
        创建并返回一个新的 State 对象，从 Model 中复制初始位姿和速度。

        The returned state is initialized with the initial configuration from the model description.

        Args:
            requires_grad (bool, optional): Whether the state variables should have `requires_grad` enabled.
                If None, uses the model's :attr:`requires_grad` setting.

        Returns:
            State: The state object
        """

        requested = self.get_requested_state_attributes()

        s = State()
        if requires_grad is None:
            requires_grad = self.requires_grad

        # particles
        if self.particle_count:
            s.particle_q = wp.clone(self.particle_q, requires_grad=requires_grad)
            s.particle_qd = wp.clone(self.particle_qd, requires_grad=requires_grad)
            s.particle_f = wp.zeros_like(self.particle_qd, requires_grad=requires_grad)

        # rigid bodies
        if self.body_count:
            s.body_q = wp.clone(self.body_q, requires_grad=requires_grad)
            s.body_qd = wp.clone(self.body_qd, requires_grad=requires_grad)
            s.body_f = wp.zeros_like(self.body_qd, requires_grad=requires_grad)

        # joints
        if self.joint_count:
            s.joint_q = wp.clone(self.joint_q, requires_grad=requires_grad)
            s.joint_qd = wp.clone(self.joint_qd, requires_grad=requires_grad)

        if "body_qdd" in requested:
            s.body_qdd = wp.zeros_like(self.body_qd, requires_grad=requires_grad)

        if "body_parent_f" in requested:
            s.body_parent_f = wp.zeros_like(self.body_qd, requires_grad=requires_grad)

        # attach custom attributes with assignment==STATE
        self._add_custom_attributes(s, Model.AttributeAssignment.STATE, requires_grad=requires_grad)

        return s

    def control(self, requires_grad: bool | None = None, clone_variables: bool = True) -> Control:
        """
        Create and return a new :class:`Control` object for this model.
        创建并返回一个新的 Control 对象。

        The returned control object is initialized with the control inputs from the model description.

        Args:
            requires_grad (bool, optional): Whether the control variables should have `requires_grad` enabled.
                If None, uses the model's :attr:`requires_grad` setting.
            clone_variables (bool): If True, clone the control input arrays; if False, use references.

        Returns:
            Control: The initialized control object.
        """
        c = Control()
        if requires_grad is None:
            requires_grad = self.requires_grad
        if clone_variables:
            if self.joint_count:
                c.joint_target_pos = wp.clone(self.joint_target_pos, requires_grad=requires_grad)
                c.joint_target_vel = wp.clone(self.joint_target_vel, requires_grad=requires_grad)
                c.joint_f = wp.clone(self.joint_f, requires_grad=requires_grad)
            if self.tri_count:
                c.tri_activations = wp.clone(self.tri_activations, requires_grad=requires_grad)
            if self.tet_count:
                c.tet_activations = wp.clone(self.tet_activations, requires_grad=requires_grad)
            if self.muscle_count:
                c.muscle_activations = wp.clone(self.muscle_activations, requires_grad=requires_grad)
        else:
            c.joint_target_pos = self.joint_target_pos
            c.joint_target_vel = self.joint_target_vel
            c.joint_f = self.joint_f
            c.tri_activations = self.tri_activations
            c.tet_activations = self.tet_activations
            c.muscle_activations = self.muscle_activations
        # attach custom attributes with assignment==CONTROL
        self._add_custom_attributes(
            c, Model.AttributeAssignment.CONTROL, requires_grad=requires_grad, clone_arrays=clone_variables
        )
        return c

    def set_gravity(
        self,
        gravity: tuple[float, float, float] | list | wp.vec3 | np.ndarray,
        world: int | None = None,
    ) -> None:
        """
        Set gravity for runtime modification.

        Args:
            gravity: Gravity vector (3,) or per-world array (num_worlds, 3).
            world: If provided, set gravity only for this world.

        Note:
            Call ``solver.notify_model_changed(SolverNotifyFlags.MODEL_PROPERTIES)`` after.

            Global entities (particles/bodies not assigned to a specific world) use
            gravity from world 0.
        """
        gravity_np = np.asarray(gravity, dtype=np.float32)

        if world is not None:
            if gravity_np.shape != (3,):
                raise ValueError("Expected single gravity vector (3,) when world is specified")
            if world < 0 or world >= self.num_worlds:
                raise IndexError(f"world {world} out of range [0, {self.num_worlds})")
            current = self.gravity.numpy()
            current[world] = gravity_np
            self.gravity.assign(current)
        elif gravity_np.ndim == 1:
            self.gravity.fill_(gravity_np)
        else:
            if len(gravity_np) != self.num_worlds:
                raise ValueError(f"Expected {self.num_worlds} gravity vectors, got {len(gravity_np)}")
            self.gravity.assign(gravity_np)

    def _init_collision_pipeline(self):
        """
        Initialize a :class:`CollisionPipeline` for this model.

        This method creates a default collision pipeline for the model. The pipeline is cached on
        the model for subsequent use by :meth:`collide`.

        """
        from .collide import BroadPhaseMode, CollisionPipeline  # noqa: PLC0415

        self._collision_pipeline = CollisionPipeline(self, broad_phase_mode=BroadPhaseMode.EXPLICIT)

    def contacts(
        self: Model,
    ) -> Contacts:
        """
        Create and return a :class:`Contacts` object for this model.

        This method initializes a collision pipeline with default arguments (when not already
        cached) and allocates a contacts buffer suitable for storing collision detection results.
        Call :meth:`collide` to run the collision detection and populate the contacts object.

        Note:
            Rigid contact margins are controlled per-shape via :attr:`Model.shape_contact_margin`, which is populated
            from ``ShapeConfig.contact_margin`` during model building. If a shape doesn't specify a contact margin,
            it defaults to ``builder.rigid_contact_margin``. To adjust contact margins, set them before calling
            :meth:`ModelBuilder.finalize`.

        Returns:
            Contacts: The contact object containing collision information.
        """
        if self._collision_pipeline is None:
            self._init_collision_pipeline()

        contacts = self._collision_pipeline.contacts()
        return contacts

    def collide(
        self,
        state: State,
        contacts: Contacts,
    ):
        """
        Generate contact points for the particles and rigid bodies in the model using the default collision
        pipeline.

        Args:
            state (State): The current simulation state.
            contacts (Contacts): The contacts buffer to populate (will be cleared first).
        """

        if self._collision_pipeline is None:
            raise ValueError(
                "Model does not have a collision pipeline. Call model.contacts() "
                "or use your collision pipeline directly: CollisionPipeline.collide(state, contacts)."
            )

        self._collision_pipeline.collide(state, contacts)

    def request_state_attributes(self, *attributes: str) -> None:
        """
        Request that specific state attributes be allocated when creating a State object.

        See :ref:`extended_state_attributes` for details and usage.

        Args:
            *attributes: Variable number of attribute names (strings).
        """
        State.validate_extended_attributes(attributes)
        self._requested_state_attributes.update(attributes)

    def request_contact_attributes(self, *attributes: str) -> None:
        """
        Request that specific contact attributes be allocated when creating a Contacts object.

        Args:
            *attributes: Variable number of attribute names (strings).
        """
        Contacts.validate_extended_attributes(attributes)
        self._requested_contact_attributes.update(attributes)

    def get_requested_contact_attributes(self) -> set[str]:
        """
        Get the set of requested contact attribute names.

        Returns:
            set[str]: The set of requested contact attributes.
        """
        return self._requested_contact_attributes

    def _add_custom_attributes(
        self,
        destination: object,
        assignment: Model.AttributeAssignment,
        requires_grad: bool = False,
        clone_arrays: bool = True,
    ) -> None:
        """
        Add custom attributes of a specific assignment type to a destination object.

        Args:
            destination: The object to add attributes to (State, Control, or Contacts)
            assignment: The assignment type to filter attributes by
            requires_grad: Whether cloned arrays should have requires_grad enabled
            clone_arrays: Whether to clone wp.arrays (True) or use references (False)
        """
        for full_name, _freq in self.attribute_frequency.items():
            if self.attribute_assignment.get(full_name, Model.AttributeAssignment.MODEL) != assignment:
                continue

            # Parse namespace from full_name (format: "namespace:attr_name" or "attr_name")
            if ":" in full_name:
                namespace, attr_name = full_name.split(":", 1)
                # Get source from namespaced location on model
                ns_obj = getattr(self, namespace, None)
                if ns_obj is None:
                    raise AttributeError(f"Namespace '{namespace}' does not exist on the model")
                src = getattr(ns_obj, attr_name, None)
                if src is None:
                    raise AttributeError(
                        f"Attribute '{namespace}.{attr_name}' is registered but does not exist on the model"
                    )
                # Create namespace on destination if it doesn't exist
                if not hasattr(destination, namespace):
                    setattr(destination, namespace, Model.AttributeNamespace(namespace))
                dest = getattr(destination, namespace)
            else:
                # Non-namespaced attribute - add directly to destination
                attr_name = full_name
                src = getattr(self, attr_name, None)
                if src is None:
                    raise AttributeError(
                        f"Attribute '{attr_name}' is registered in attribute_frequency but does not exist on the model"
                    )
                dest = destination

            # Add attribute to the determined destination (either destination or dest_ns)
            if isinstance(src, wp.array):
                if clone_arrays:
                    setattr(dest, attr_name, wp.clone(src, requires_grad=requires_grad))
                else:
                    setattr(dest, attr_name, src)
            else:
                setattr(dest, attr_name, src)

    def add_attribute(
        self,
        name: str,
        attrib: wp.array | list,
        frequency: Model.AttributeFrequency | str,
        assignment: Model.AttributeAssignment | None = None,
        namespace: str | None = None,
    ):
        """
        Add a custom attribute to the model.

        Args:
            name (str): Name of the attribute.
            attrib (wp.array | list): The array to add as an attribute. Can be a wp.array for
                numeric types or a list for string attributes.
            frequency (Model.AttributeFrequency | str): The frequency of the attribute.
                Can be a Model.AttributeFrequency enum value or a string for custom frequencies.
            assignment (Model.AttributeAssignment, optional): The assignment category using Model.AttributeAssignment enum.
                Determines which object will hold the attribute.
            namespace (str, optional): Namespace for the attribute.
                If None, attribute is added directly to the assignment object (e.g., model.attr_name).
                If specified, attribute is added to a namespace object (e.g., model.namespace_name.attr_name).

        Raises:
            AttributeError: If the attribute already exists or is on the wrong device.
        """
        if isinstance(attrib, wp.array) and attrib.device != self.device:
            raise AttributeError(f"Attribute '{name}' device mismatch (model={self.device}, got={attrib.device})")

        # Handle namespaced attributes
        if namespace:
            # Create namespace object if it doesn't exist
            if not hasattr(self, namespace):
                setattr(self, namespace, Model.AttributeNamespace(namespace))

            ns_obj = getattr(self, namespace)
            if hasattr(ns_obj, name):
                raise AttributeError(f"Attribute already exists: {namespace}.{name}")

            setattr(ns_obj, name, attrib)
            full_name = f"{namespace}:{name}"
        else:
            # Add directly to model
            if hasattr(self, name):
                raise AttributeError(f"Attribute already exists: {name}")
            setattr(self, name, attrib)
            full_name = name

        self.attribute_frequency[full_name] = frequency
        if assignment is not None:
            self.attribute_assignment[full_name] = assignment

    def get_attribute_frequency(self, name: str) -> Model.AttributeFrequency | str:
        """
        Get the frequency of an attribute.

        Args:
            name (str): Name of the attribute.

        Returns:
            Model.AttributeFrequency | str: The frequency of the attribute.
                Either a Model.AttributeFrequency enum value or a string for custom frequencies.

        Raises:
            KeyError: If the attribute frequency is not known.
        """
        frequency = self.attribute_frequency.get(name)
        if frequency is None:
            raise KeyError(f"Attribute frequency of '{name}' is not known")
        return frequency

    def get_custom_frequency_count(self, frequency: str) -> int:
        """
        Get the count for a custom frequency.

        Args:
            frequency (str): The custom frequency (e.g., ``"mujoco:pair"``).

        Returns:
            int: The count of elements with this frequency.

        Raises:
            KeyError: If the frequency is not known.
        """
        if frequency not in self.custom_frequency_counts:
            raise KeyError(f"Custom frequency '{frequency}' is not known")
        return self.custom_frequency_counts[frequency]

    def get_requested_state_attributes(self) -> list[str]:
        """
        Get the list of requested state attribute names that have been requested on the model.

        See :ref:`extended_state_attributes` for details.

        Returns:
            list[str]: The list of requested state attributes.
        """
        attributes = []

        if self.particle_count:
            attributes.extend(
                (
                    "particle_q",
                    "particle_qd",
                    "particle_f",
                )
            )
        if self.body_count:
            attributes.extend(
                (
                    "body_q",
                    "body_qd",
                    "body_f",
                )
            )
        if self.joint_count:
            attributes.extend(("joint_q", "joint_qd"))

        attributes.extend(self._requested_state_attributes.difference(attributes))
        return attributes
