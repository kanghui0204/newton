"""
Newton 参考文档 12：关节系统、坐标数/约束数、Articulation 完全指南
====================================================================

本文件整合了关于关节系统的所有知识，包括：
- 坐标数和约束数的定义和源码位置
- joint_q 数组的布局
- parent=-1 + parent_xform 的含义
- add_articulation 的作用和源码位置
- test_body_state 的定义位置
- 导入文件(USD/URDF)自动创建articulation的源码位置

================================================================================
一、坐标数和约束数的定义
================================================================================

【源码位置】
    newton/_src/sim/joints.py → JointType 枚举类
    第 49 行: dof_count() 方法 → 返回 (自由度数, 坐标数)
    第 81 行: constraint_count() 方法 → 返回 约束数

【定义】

一个自由刚体在3D空间中有 6 个自由度（3平移+3旋转）。
关节就是"限制其中一些自由度"：

  自由度 (DoF)   = 关节允许的运动维度
  约束数         = 关节锁死的运动维度
  自由度 + 约束数 = 6（总是等于6！）

  坐标数 (coords) = 用几个 float 来描述当前关节状态
    通常坐标数 = 自由度
    但 BALL(四元数4个值描述3个旋转自由度)
    和 FREE(7个值描述6个自由度) 是过参数化

【完整关节类型表】

    类型        自由度  坐标数  约束数  joint_q 存什么
    ──────────────────────────────────────────────────────────
    REVOLUTE     1      1      5      [θ] 旋转角度(弧度)
    PRISMATIC    1      1      5      [d] 滑动距离(米)
    BALL         3      4      3      [qx,qy,qz,qw] 四元数
    FIXED        0      0      6      (无) 完全锁死
    FREE         6      7      0      [x,y,z,qx,qy,qz,qw]
    D6         1-6    1-6    可变     取决于启用的轴数
    DISTANCE     6      7      0      [x,y,z,qx,qy,qz,qw]
    CABLE        2      2      4      [stretch, bend]

    CABLE 的2个自由度：
    ① stretch(拉伸)：缆线沿轴向伸缩
    ② bend(弯曲)：缆线可以弯曲
    其余4个方向被约束（2个横向平移+2个旋转）
    专门为电缆/软管/绳索设计的弹性连接元素

【约束的物理含义】

    以 REVOLUTE (铰链) 为例，5个约束：
    ① 连接点x坐标一致（不能x方向分离）
    ② 连接点y坐标一致
    ③ 连接点z坐标一致
    ④ 子体不能绕非旋转轴方向1旋转
    ⑤ 子体不能绕非旋转轴方向2旋转
    剩下1个自由度：绕axis旋转

    以 FIXED (焊接) 为例，6个约束：
    ①②③ 三个平移方向全锁死
    ④⑤⑥ 三个旋转方向全锁死
    → 0自由度，完全焊死

【源码中的计算逻辑】

    # newton/_src/sim/joints.py 第96行
    cts_count = 6 - num_axes  # 默认：约束数 = 6 - 轴数
    if self == JointType.BALL:
        cts_count = 3           # BALL 特殊：3个平移约束
    elif self == JointType.FREE or self == JointType.DISTANCE:
        cts_count = 0           # FREE/DISTANCE：无约束
    elif self == JointType.FIXED:
        cts_count = 6           # FIXED：6个约束全锁

================================================================================
二、joint_q 数组的布局和 joint_q[-1] 的含义
================================================================================

joint_q 是一个展平的 float 列表(builder阶段)或数组(finalize后)。
不同关节的坐标依次排列，FIXED关节不占位置。

【例子：03_basic_joints 中的布局】

    添加顺序：
    j_fixed_rev  = add_joint_fixed(...)      # FIXED: 坐标数=0
    j_revolute   = add_joint_revolute(...)   # REVOLUTE: 坐标数=1
    j_fixed_pri  = add_joint_fixed(...)      # FIXED: 坐标数=0
    j_prismatic  = add_joint_prismatic(...)  # PRISMATIC: 坐标数=1
    j_fixed_ball = add_joint_fixed(...)      # FIXED: 坐标数=0
    j_ball       = add_joint_ball(...)       # BALL: 坐标数=4

    joint_q = [θ_revolute, d_prismatic, qx, qy, qz, qw]
               ↑            ↑           ↑──────────────↑
            索引0         索引1          索引2,3,4,5
            joint_q[-6]   joint_q[-5]   joint_q[-4:]

    注意：FIXED关节不在joint_q中占位置！

    builder.joint_q[-1] = wp.pi * 0.5
    → 设置的是 j_revolute 的角度（列表最后一个元素）
    → 但实际上在03_basic_joints中，-1指向的是BALL的qw分量
    → 具体取决于代码中最后添加的是哪个关节

    builder.joint_q[-4:] = wp.quat_rpy(0.5, 0.6, 0.7)
    → 设置最后4个元素 = BALL关节的四元数

【找某个关节在joint_q中的位置】

    finalize后:
    model.joint_q_start[i]   → 第i个关节在joint_q中的起始索引
    model.joint_qd_start[i]  → 第i个关节在joint_qd中的起始索引

================================================================================
三、parent=-1 + parent_xform 的含义
================================================================================

    parent=-1:
      父体是"世界"——绝对固定的背景参考系
      世界本身不是一个body，它永远不动

    parent_xform.p 的含义取决于 parent 是谁：
      parent=-1 (世界): p 就是世界坐标系中的绝对位置
      parent=some_body: p 是那个body局部坐标系中的相对位置

    child_xform.p:
      始终是子体局部坐标系中的相对位置

    物理引擎的核心约束：
      parent_xform 和 child_xform 指定的两个点在世界中始终重合

    例子:
      add_joint_fixed(parent=-1, child=anchor,
          parent_xform=transform(p=(0,0,5), ...),
          child_xform=transform(p=(0,0,0), ...))
      → anchor的原点被钉在世界坐标(0,0,5)处

================================================================================
四、add_articulation 的作用
================================================================================

【功能】把多个关节声明为一个关节链

【源码位置】
    newton/_src/sim/builder.py 第 1317 行: def add_articulation()
    newton/_src/sim/builder.py 第 749 行: self.joint_articulation (每个joint属于哪个articulation)
    newton/_src/sim/builder.py 第 751 行: self.articulation_start (每个articulation的起始joint)

【内部做了什么】(builder.py 第1403行)

    # 记录这个articulation的第一个joint索引
    self.articulation_start.append(sorted_joints[0])
    self.articulation_key.append(key)
    self.articulation_world.append(self.current_world)

    # 标记所有属于这个articulation的joint
    for joint_idx in joints:
        self.joint_articulation[joint_idx] = articulation_idx

【有什么好处】

    1. eval_fk 按正确拓扑顺序计算：
       先算根关节的body位姿，再算子关节的body位姿
       没有articulation信息就不知道先算哪个

    2. Featherstone/MuJoCo 求解器依赖链结构：
       递推算法需要从根到叶遍历整条链

    3. joint_q 中同一条链的坐标连续排列：
       方便整体读写一个机器人的所有关节坐标

    4. replicate() 时整条链作为一个单元复制：
       确保关节拓扑在每个世界中一致

【不加会怎样】
    仿真能跑，但求解器不知道关节之间的层级关系。
    对于简单场景（一两个关节）影响不大，
    对于复杂机器人（30+关节的树形结构）可能导致错误或低效。

================================================================================
五、从 USD/URDF/MJCF 导入时自动创建 Articulation
================================================================================

【URDF 导入】
源码位置: newton/_src/utils/import_urdf.py 第 839-849 行

    # Create articulation from all collected joints
    articulation_key = urdf_root.attrib.get("name")
    builder._finalize_imported_articulation(
        joint_indices=joint_indices,
        parent_body=parent_body,
        articulation_key=articulation_key,
    )

    → add_urdf() 解析完所有关节后，自动调用
      _finalize_imported_articulation() 创建 articulation
    → 你不需要手动调用 add_articulation

【USD 导入】
源码位置: newton/_src/utils/import_usd.py

    USD导入同样在内部自动处理 articulation 的创建。
    它从 USD 的 PhysicsArticulationRootAPI 中识别关节链结构，
    自动调用 builder._finalize_imported_articulation()

【总结】
    手动构建 → 需要自己调 add_articulation([j0, j1, ...])
    add_urdf() → 自动创建 articulation（从URDF的name属性取key）
    add_usd()  → 自动创建 articulation（从USD的ArticulationRoot取key）
    add_mjcf() → 自动创建 articulation

================================================================================
六、test_body_state 的定义位置
================================================================================

【源码位置】
    newton/examples/__init__.py 第 38 行: def test_body_state()

【功能】
    对指定body的位姿和速度执行 lambda 谓词检测。
    如果任何body不满足条件，抛出 ValueError。

【签名】
    test_body_state(
        model: newton.Model,       # 模型
        state: newton.State,       # 当前状态
        test_name: str,            # 测试名称（用于错误信息）
        test_fn: (q, qd) -> bool,  # 检测函数
        indices: list[int],        # 要测试的body索引列表
    )

    其中:
    q  = body_q[i]  : wp.transform (7维: pos + quaternion)
    qd = body_qd[i] : wp.spatial_vector (6维: linear_vel + angular_vel)

【用法示例】(03_basic_joints_annotated.py)

    # 检查旋转关节只绕X轴运动
    newton.examples.test_body_state(
        self.model, self.state_0,
        "revolute motion in plane",
        lambda q, qd: wp.length(abs(wp.cross(
            wp.spatial_bottom(qd),   # 角速度
            wp.vec3(1.0, 0.0, 0.0)   # X轴
        ))) < 1e-5,                   # 角速度与X轴平行 → 叉积≈0
        indices=[model.body_key.index("b_rev")],
    )

    它不是在 03 文件中定义的，而是从 newton.examples 模块导入使用的。

【类似函数】
    test_particle_state(): 同文件第 116 行
    检测粒子的位置和速度，签名：
    test_fn: (q: vec3, qd: vec3) -> bool
"""
