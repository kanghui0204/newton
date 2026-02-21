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

from enum import IntEnum


# Particle flags
# 粒子标志位（用位运算组合多个标志）
class ParticleFlags(IntEnum):
    """Flags for particle properties.
    粒子属性标志。用于控制粒子是否参与仿真。

    使用方式:
        # 在 GPU kernel 中检查粒子是否活跃
        if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
            return  # 跳过非活跃粒子

        # 在 Python 中禁用某个粒子
        flags = model.particle_flags.numpy()
        flags[particle_id] = 0  # 移除 ACTIVE 标志 → 粒子不参与仿真
        model.particle_flags.assign(flags)
    """

    ACTIVE = 1 << 0
    """Indicates that the particle is active.
    粒子是否活跃（参与仿真）。0 = 休眠/禁用，不参与积分和碰撞。"""


# Shape flags
# 形状标志位（用位运算组合多个标志）
class ShapeFlags(IntEnum):
    """Flags for shape properties.
    形状属性标志。控制形状的可见性、碰撞行为等。

    标志可以组合使用（位运算 OR）:
        flags = ShapeFlags.VISIBLE | ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES
        = 0b00111 = 7  → 可见、参与刚体碰撞、参与粒子碰撞

    在 ShapeConfig 中通过布尔属性设置，不需要手动操作位运算：
        cfg = ShapeConfig(is_visible=True, has_shape_collision=True, has_particle_collision=True)
    """

    VISIBLE = 1 << 0
    """Indicates that the shape is visible.
    是否在 Viewer 中可见。bit 0 = 0b00001。"""

    COLLIDE_SHAPES = 1 << 1
    """Indicates that the shape collides with other shapes.
    是否参与刚体-刚体碰撞（rigid contact，GJK/MPR 窄相）。bit 1 = 0b00010。"""

    COLLIDE_PARTICLES = 1 << 2
    """Indicates that the shape collides with particles.
    是否参与粒子-形状碰撞（soft contact，SDF 距离查询）。bit 2 = 0b00100。"""

    SITE = 1 << 3
    """Indicates that the shape is a site (non-colliding reference point).
    是否是"站点"——不参与碰撞的参考点，用于传感器安装、目标位置标记等。
    设为 SITE 时自动禁用 COLLIDE_SHAPES 和 COLLIDE_PARTICLES。bit 3 = 0b01000。"""

    HYDROELASTIC = 1 << 4
    """Indicates that the shape uses hydroelastic collision."""


__all__ = [
    "ParticleFlags",
    "ShapeFlags",
]
