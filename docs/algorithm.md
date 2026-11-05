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

以下首先描述默认 `voxelizer=subdivide`。v0.3 另有显式选择的原 mesh 距离准备方式，见本节后文；两者不承诺相同的占据中心或球结果。

1. 设体素边长为显式 `voxel_size`，否则为最长包围盒边长除以 `resolution`。
2. 根据包围盒与边界裕量估算网格数量，在体素化前检查 `max_grid_cells`。
3. 按原有的整三角中点细分与格点舍入规则离散表面，并填充封闭壳的内部。任意一条边过长时，三条边一起二分，生成四个子三角形；这保持 Trimesh 4.12 时的规则，不随 Trimesh 5 的新细分默认值改变。v0.3.1 分批深度优先处理，直接写入占据栅格，不保留完整细分网格。用空白边界包围栅格后计算欧氏距离变换，避免实心长方体的栅格边界影响 EDT。
4. 对每个占据体素中心构造半径 `max(EDT - 0.5, 0.5) * pitch` 的候选球。这个半径来自离散栅格，不是精确到原始三角表面的有符号距离。
5. 混合局部距离极大值和确定性随机体素作为候选中心，限制候选数量。向半径加上 `padding`。
6. 在最多 `samples` 个占据体素中心上进行贪心集合覆盖：每次选能覆盖最多尚未覆盖样本的候选球，使用惰性堆更新收益。球数不超过指定预算；没有额外收益时停止。
7. 用全部占据体素中心计算 `voxel_center_coverage`，再用独立表面样本评估间隙。

该方法允许球重叠，不做凸包替代，也不将尚未覆盖区域隐藏在一个笼统的成功标记里。提高球数、padding 或分辨率可以改变结果，覆盖率不保证随所有参数单调变化。候选数与评价样本数有限，小部件可能需要更高采样预算或手动调整。

### 可选 `open3d` / `warp` 距离准备

新后端直接查询原三角 mesh，在有界规则格点中仅保留数值判断在内部的中心，候选半径取原 mesh 的内部距离减数值裕量，不再用 EDT 减半格的近似。Open3D 查询在 CPU；Warp 可在 CPU 或 CUDA。后续候选选择、refined 和报告沿用明确的阶段接口。

默认 `single` 拒绝多壳歧义；显式 `union-components` 对各闭合壳分别采样，再按实体并集合并内部中心，采用各实体内部 clearance 的最大值作为并集半径下界。该解释会填掉嵌套壳表示的空腔，不恢复原始 URDF 部件分组。

每个组件必须产生内部格点。多组件还保留各组件最深内部代表点，先覆盖它们再花费剩余球预算；预算不足或优化后丢掉代表点时明确失败。只保留代表点，不等于完整组件覆盖。

距离查询只枚举各组件包围盒附近的格点，并在边界扩展额外格层以容纳坐标舍入。组件外部不贡献内部 clearance，因此分散组件可跳过大量查询；相交、嵌套组件仍按原并集规则合并。

查询几何采用居中、归一化的 float32。格点先按最终球心的同一物理坐标重建，半径裕量同时考虑归一化数值精度与世界坐标 ULP，避免大平移下查询点与导出中心发生不一致。距离和符号仍是数值近似，没有自交检测或形式化误差证书。完整限制和独立质量/速度对比见 [voxelizer-performance.md](voxelizer-performance.md)。

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

## `cover-tight`：缩紧已有表面分组

保留 `cover` 的三角形分组，针对每簇点集优化包围球中心。先平移并按点集尺度归一化，再对有限支持集求解有约束的数值最小包围球；发现不在候选球内的点时加入支持集继续求解。每次都回到原始坐标，用全部点重新计算半径并向外舍入，只接受比原中心包围半径更小的结果。优化预算耗尽或失败时使用已经认证的原包围球。

因此，在相同三角分组下，每球半径不会增加（数值舍入除外），完整三角表面覆盖保留；这不意味着新球一定是旧球的子集，也不保证整个球并集的外扩体积单调下降。它不是精确最小包围球求解器或完整 VSSA 复现。

## `cover-opt`：重新分组的三角表面包络

从 `cover-tight` 的结果开始，最多执行两轮：按三角形三个顶点到球心的最大距离，
将每个完整三角形分配给代价最小的球心，再对各组全部顶点调用同一包围球缩紧器。
距离计算按三角形和球心分块，不构造全量点球距离矩阵；并列时保留较早的球心。
空组被移除，球数不会增加。最终半径向外舍入并加 padding。

只接受包含 padding 的 `sum(radius**3)` 严格下降的候选，否则保留上一轮并停止。
比较时用共同的半径尺度归一化，避免立方计算溢出或下溢。这个目标与球体积之和成正比，
包含重叠部分的重复计算，不能据此保证球并集体积或外扩体积单调下降。

每个完整三角形仍由同一个球包含，保留三角表面包络。重新分组可能改变球数、顺序和单球半径；
内部覆盖可能降低。它没有实现 VSSA 的原网格外部体积积分，效果与额外耗时见
[独立测量](cover-optimization.md)。

## `slab`：单轴平行切片的完整截面包络

1. 从 PCA 的候选轴中选投影跨度最大的轴，按投影范围均匀分成最多 `count` 个闭 slab。
2. 针对每个与 slab 相交的原三角形，保留区间内的原顶点和三条边与两个边界平面的全部交点。即使 slab 内没有原顶点，穿过它的三角形也不会被跳过。相邻 slab 使用同一组分界面。
3. 一个 slab 内所有裁剪多边形的顶点共同构造一个包围球，使用 `tighten_ball` 缩紧，并对全部点重新认证。
4. 半径加入用户 padding，以及基于模型尺度和坐标 ULP 的数值裕量。

**几何推导。** 设有界闭实体为 `K`，输入曲面为 `M`，且 `boundary(K)` 包含于 `M`。如果每个 slab 的球包含 `M ∩ slab`，则它也包含 `K ∩ slab`：对任意内部点 `p` 取一条垂直分割轴、通过 `p` 的直线。包含 `p` 的实体区间两端 `q1, q2` 均为原实体边界点，且整条直线的轴向投影相同，因此二者属于同一个 slab，也都在它的球内。由球的凸性，`p` 在球内。边界点直接由表面包络得到。

这个结论不要求实体凸或横截面连通，也适用于完整闭合部件的并集；但单个截面上的多个分支和空腔可能被同一个球过度包络。它依赖单轴 slab 横向无限延展，不能直接推广为任意三维网格单元中的原表面覆盖。

代码报告 `bounded_solid_coverage_conditional` 与具体假设；`is_watertight` 不代表已经排除自交或确定实体解释。数值实现带裁剪容差和半径裕量，没有区间算术或精确谓词，因此不把几何推导称为已形式化验证的浮点证明。URDF 球和圆柱先离散成三角 mesh，保证范围也以该三角几何为准。

## `refined`：固定预算的中心与半径优化

采用未加用户 padding 的体素球初始化。归一化坐标后，用 CPU NumPy 解析梯度或 CUDA PyTorch 自动微分执行相同 float64 Adam 更新：

```text
g(p) = min_i(norm(p - c_i) - r_i)
L = 8 * mean_surface(relu(g + margin)^2)
  + 1 * mean_inside(relu(g + margin)^2)
  + outside_weight * mean_outside(min(g, 0)^2)
  + 0.002 * mean_i(r_i^3)
margin = 0.15 * voxel_pitch / normalization_scale
```

每组独立归一化。中心限制在原 mesh 包围盒内，半径显式保持为正；固定学习率为归一化坐标的 0.006，默认 120 步。输出选择训练目标最低的迭代状态，再转回物理单位，只在此时添加用户 padding。该有限样本目标不是 MorphIt 原作者实现，也不直接等同于连续体积积分。

默认 `refine_reference=voxel` 使用填充栅格内部点，外部点来自距离占据体素 1.5～4 格的窄带；栅格边界与 mesh 边界之间有偏差，因此不能以训练损失下降宣称原 mesh 的 IoU 提升。`refine_reference=mesh` 使用原始 `mesh.contains` 给近表面扰动和包围盒样本分类，减少栅格标签偏差，但需要 rtree、单一闭合表面和无自交假设。多壳不能未经说明使用整体奇偶内部。

训练种子为用户 seed 加固定偏移，表面报告和最终比较样本使用其他种子。CPU/CUDA 在本地测试中同目标、同数值精度，夹紧距离的梯度也保持一致；一般硬件与库版本下不承诺全局逐位确定性。训练矩阵最多 400 万个点球对，超过时显式要求降低样本数或球数。这个预算不是进程总内存的严格上限。

## 并行执行边界

代码按阶段组织：`fit.py` 负责准备、方法选择和报告，`surface_fit.py` 与 `voxel_fit.py` 实现表面和体素拟合；`geometry.py` 提供共享采样与距离函数。`refine.py` 和 `quality.py` 直接使用这些基础函数，不反向依赖拟合调度。原有 `assetspheres.fit.sample_surface/sphere_gaps` 导入路径保留。

CPU 的 `fit_asset(..., workers=N, executor='thread'|'process')` 按独立 link 分发，结果按输入顺序汇总。线程共享进程并避免 mesh 序列化；进程使用 spawn，包含额外导入、传输与上下文成本。默认仍为一个 worker，不隐式占满 CPU。

CUDA 当前覆盖候选球贪心收益和点球距离，以及 `refined` 的优化；体素化和 EDT 仍在 CPU。贪心外层有顺序依赖，GPU 并行的是候选和点维度。内存预算包括计划工作矩阵及归约临时张量，不包括线性输入输出、框架运行时和其他进程。显式 CUDA 请求在不可用时报告错误，不自动切回 CPU。

当前没有实现 CPU 准备队列与 GPU 执行的异步重叠。更完整的流水线应先缓存 mesh/BVH，再由有限数量 CPU worker 生产数据、单个 GPU 执行器复用上下文；是否值得实现由端到端 profile 决定。实际计时和进一步路线见 [performance.md](performance.md)。

## 输出和质量语义

球并集的覆盖间隙按以下公式计算：

```text
gap(p) = min_i (norm(p - center_i) - radius_i)
```

`gap <= 0` 表示在球并集内。报告中的间隙为 `max(gap, 0)`，所以已覆盖样本记为零。

外部点的正间隙是到球并集的精确距离；内部的负值用于判断包含关系，不宣称是到重叠球并集外边界的精确有符号距离。

表面评价使用与拟合不同的随机种子。报告不提供未实现的精确外扩体积、精确球并集体积或连续实体覆盖证明。手动修改球后保留原始报告但显式设置 `quality_metrics_stale`，CLI 不再把其旧值展示成当前质量。

结果 JSON 的 `schema_version=1`、`units=metre`、`sphere_frame=link_local` 是接口契约。cuRobo 与 Lula 导出只生成对应格式的碰撞球片段。URDF 导出复制原结构后替换所选 link 的 collision；保留的 mesh 和 URDF material 纹理引用按新位置重写，不复制原始资产或纹理。所有输出先完整序列化，再逐文件原子替换；这不是跨多个文件的事务。GLB 以每 link 一个共享单位球网格加节点位姿/尺度表达实例，应通过场景图遍历全部球。
