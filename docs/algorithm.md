# 球化算法与坐标约定

## 参考资料与实现边界

查阅日期：2026-11-03。

1. [IsaacLab cuRobo planner](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_mimic/isaaclab_mimic/motion_planners/curobo/curobo_planner.py)：附着物通过 `attach_objects_to_robot(..., sphere_fit_type=SphereFitType.SAMPLE_SURFACE)` 请求表面球化。机器人本身可加载预制的碰撞球文件。这不是 IsaacLab 内部实现了一套通用自动体积球化的证据。
2. [Isaac Sim Lula Robot Description and XRDF Editor](https://docs.isaacsim.omniverse.nvidia.com/latest/manipulators/manipulators_robot_description_editor.html)：按 link 编辑球、从闭合三角 mesh 自动生成指定数量的球、增加半径偏移。文档说明自动生成并非适用于全部 mesh。
3. [Lula Python API](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/py/source/deprecated/isaacsim.robot_motion.lula/docs/index.html)：公开 `create_collision_sphere_generator(vertices, triangles)`、`generate_spheres(num_spheres, radius_offset)`，以及 medial-axis 候选球的查询接口。内部候选生成与优化细节并未在这份 API 中给出。
4. [用户指定知乎回答](https://www.zhihu.com/question/2096010351653435323/answer/2097772625942655119)：读取失败，内容待核实。没有将其他搜索结果当作这条回答的内容。

本仓库独立实现 CPU 拟合代码，没有复制 NVIDIA 源码。`voxel` 采用体积近似与候选球选择的通用思路；不等同于 Lula 内部算法，也不是精确 medial axis 提取。`surface` 借鉴其用途并采用本项目的最远点采样实现。`lula` 后端直接调用公开 API；不在缺失依赖时悄悄切换算法。

## 坐标与单位

对于 mesh 顶点 `v`，先应用 URDF 的对角缩放矩阵 `S`，再应用 geometry 的 origin：

```text
v_link = T_link_geometry @ S @ v_mesh
```

`rpy` 使用固定轴 X/Y/Z 的 roll/pitch/yaw，对应 `Rz(yaw) @ Ry(pitch) @ Rx(roll)`。同一个 link 的多个 collision 或 visual 都先转入 link 坐标系，再合并拟合。

所有导出的球心都为 `center_link`，不将 joint origin 烘焙进球心。预览时才计算：

```text
center_world = T_world_link(q=0) @ center_link
```

预览的 joint 位置按零处理，没有对 mimic offset、关节限位或运动进行求解。URDF 导出保留输入的 joint 定义，因此目标机器人系统仍可正常处理其运动学。

URDF 采用米和弧度。独立 mesh 不推测单位，毫米资产需 `--scale 0.001`。独立场景先展开节点变换与重复实例，再应用单位缩放。

## `voxel`：体素距离场候选球

1. 设体素边长为显式 `voxel_size`，否则为最长包围盒边长除以 `resolution`。
2. 根据包围盒与边界裕量估算网格数量，在体素化前检查 `max_grid_cells`。
3. 用 trimesh 体素化三角表面，并填充封闭壳的内部。用空白边界包围栅格后，计算欧氏距离变换，避免实心长方体的栅格边界影响 EDT。
4. 对每个占据体素中心构造半径 `max(EDT - 0.5, 0.5) * pitch` 的候选球。这个半径来自离散栅格，不是精确到原始三角表面的有符号距离。
5. 混合局部距离极大值和确定性随机体素作为候选中心，限制候选数量。向半径加上 `padding`。
6. 在最多 `samples` 个占据体素中心上进行贪心集合覆盖：每次选能覆盖最多尚未覆盖样本的候选球，使用惰性堆更新收益。球数不超过指定预算；没有额外收益时停止。
7. 用全部占据体素中心计算 `voxel_center_coverage`，再用独立表面样本评估间隙。

该方法允许球重叠，不做凸包替代，也不将尚未覆盖区域隐藏在一个笼统的成功标记里。提高球数、padding 或分辨率可以改变结果，覆盖率不保证随所有参数单调变化。候选数与评价样本数有限，小部件可能需要更高采样预算或手动调整。

体素填充将闭合壳视为实体，完全封闭的内部空腔可能被填实；相邻薄壁、很小的孔和窄间隙也可能因分辨率而消失。网格单元数限制约束栅格规模，不是整个进程内存的严格上限：输入 mesh、三角细分、候选覆盖索引同样占用内存。

## `surface`：固定半径表面球

1. 按三角面积加权采样；三角形内采用均匀重心坐标采样。
2. 用最远点策略从候选样本中挑选球心，改善空间分布。
3. 每个球半径固定为 `surface_radius + padding`。

适合开口或薄片资产，也可近似规划中的附着物。它覆盖的是离散表面邻域，不承诺填充物体内部。采样器只使用局部 NumPy Generator，不修改调用方的全局随机状态。

## `cover`：完整三角表面的包络

1. 对面数较少的 mesh 进行有限细分，保持原三角形的几何表面，改善长三角形导致的粗包络。
2. 对三角形重心使用最远点初始化，再按面积加权迭代更新簇中心。
3. 将每个完整三角形分配给一个中心。
4. 每个球半径取所属全部三角形顶点到中心距离的最大值，向外舍入并加 padding。

因为球是凸集，一个球包含三角形的全部顶点，就包含该完整三角形。因此这是边界表面的构造性覆盖，不依赖采样率。球并集未必包含整个实体内部，也可能显著突出原 mesh。零面积面不作为有效几何处理。

## `lula`：原生后端

使用 float64 的 `(3, 1)` 顶点和 int32 的 `(3, 1)` 面索引调用公开构造函数，将 `count` 与 `padding` 分别传给 `num_spheres` 与 `radius_offset`。原生结果也转换为本项目的 link 局部 `Sphere`，走相同的报告与导出流程。

依赖需要由 Isaac Sim 环境提供。当前测试只验证适配器的参数和数据结构约定，不替代对实际原生运行时的测试。

## 输出和质量语义

球并集的覆盖间隙按以下公式计算：

```text
gap(p) = min_i (norm(p - center_i) - radius_i)
```

`gap <= 0` 表示在球并集内。报告中的间隙为 `max(gap, 0)`，所以已覆盖样本记为零。

外部点的正间隙是到球并集的精确距离；内部的负值用于判断包含关系，不宣称是到重叠球并集外边界的精确有符号距离。

表面评价使用与拟合不同的随机种子。报告不提供未实现的精确外扩体积、精确球并集体积或连续实体覆盖证明。手动修改球后保留原始报告但显式设置 `quality_metrics_stale`，CLI 不再把其旧值展示成当前质量。

结果 JSON 的 `schema_version=1`、`units=metre`、`sphere_frame=link_local` 是接口契约。cuRobo 与 Lula 导出只生成对应格式的碰撞球片段。URDF 导出复制原结构后替换所选 link 的 collision；保留的 mesh 引用按新位置重写，不复制原始资产或纹理。所有输出先完整序列化，再逐文件原子替换；这不是跨多个文件的事务。
