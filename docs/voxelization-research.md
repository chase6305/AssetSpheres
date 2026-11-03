# 体素准备阶段的替代路线实验

日期：2026-11-03。本文件讨论准备阶段，不宣称完整球拟合已经获得相同比例的加速。仅新增独立研究脚本，没有修改默认后端。

## 测量方式

脚本为 [benchmarks/explore_voxelization.py](../benchmarks/explore_voxelization.py)。原始逐次计时、网格和反例位于本地 `outputs/explore-voxelization.json`。

使用同一锚点 `point = integer_index × pitch`、最长轴 resolution 48 的规则格点，包围盒外多留一格；不同方法的 mask 都在这些相同位置比较。Open3D 0.19.0、trimesh 4.12.2、NumPy 1.26.4，单查询线程，重复 3 次取中位数。trimesh 参考查询在本机使用 `trimesh.ray.ray_pyembree` / embreex 加速，不能直接外推到只有纯 Python 三角射线的环境。

Open3D 的 scene 准备时间包含顶点、查询点的归一化和 Float32 转换，以及一次强制执行的单点距离查询，避免把延迟的 BVH 提交藏进预热阶段。后续分别计 occupancy 与 signed-distance 查询。方法的完整准备时间为 scene 准备加对应查询，不包含 Open3D 的进程内首次 import，也不包含 sphere fitting 和导出。

本机首次 Open3D import 约 0.96 秒；小资产一次性处理时，这个成本可能超过其后续查询收益。工作站并未隔离其他任务，特别是 subdivide 的大三角形细分有较明显的运行间波动。

## 同规则格点上的时间

单位为 ms，查询线程数为 1。Open3D 的两列完整准备均含 BVH 准备和数据转换；SDF 查询是在已建立场景上计算，不需要再建另一份 BVH。

| 输入 | 格点数 | subdivide + fill | ray + fill | Open3D occupancy 完整准备 | Open3D SDF 完整准备 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Box | 31,059 | 39.6 | 3.07 | 4.90 | 19.0 |
| Cylinder | 22,491 | 1,673 | 1.44 | 4.27 | 14.0 |
| Torus | 39,015 | 78.2 | 5.88 | 10.0 | 56.6 |
| 2 mm 薄闭盒，位于格层之间 | 5,916 | 38.6 | 3.15 | 1.10 | 3.66 |
| 分离的小部件 | 29,900 | 36.5 | 2.65 | 4.62 | 11.1 |
| 嵌套空腔 | 81,141 | 64.6 | 5.89 | 16.9 | 82.4 |
| 相交的两个闭盒 | 33,201 | 116 | 3.72 | 7.89 | 21.0 |

圆柱的 scene/BVH 准备中位数仅约 0.53 ms，occupancy 查询约 3.74 ms。这里长三角形不需要递归细分，因此对这一类输入非常有吸引力。Torus 已有较密三角面，SDF 的最近表面查询成本较高，收益较小。

`ray + fill` 虽然最快，却不是可直接采用的正确替换，见下述反例。

## 四种不同的采样语义

1. **subdivide + fill**：把三角表面采样并舍入到体素中心，随后填充封闭孔洞。占据中心可以位于原 mesh 外；完全封闭的内部空腔也可能被填实。
2. **严格中心占据**：只保留判定在原实体内部的格点。它保留了内部的几何定义，但不能保证小于 pitch 的结构恰好拥有一个内部格点。
3. **SDF 表面带**：保留 `signed_distance <= sqrt(3) × pitch / 2` 的中心，包含内部及半格对角线宽的外部带。若距离精确，每个与表面相交的体素，其中心到表面不超过半格对角线；但数值实现仍需容差，而且这会增加外部占据。
4. **Z 射线命中点 voxelization**：trimesh 当前 `ray` 实现沿 Z 采集命中位置后离散化，不等于把每条射线在实体内部的区间全部填满。后续 `fill()` 也不能自动补全所有缺少侧壁的结构。

本轮 Box 和 Cylinder 的 SDF 表面带与 subdivide 的总占据格数碰巧一致，分别为 22,491 与 11,809；Torus 则是 13,532 对 12,596。即使部分形状相同，也不能把两种构造视为普遍等价。

## 反例与正确性边界

### 快射线体素化会漏掉内部

对 Box 的 17,625 个非边界内部格点，`ray + fill` 没覆盖任何一个；对 Cylinder 的 9,071 个非边界内部格点也全部遗漏。Torus 内部格点覆盖约 19%。这些数字来自当前 trimesh 版本和本机 Embree 路径，说明其快时间不是完整实体占据的同任务加速。

### 中心占据可能完全看不到薄壁和小部件

在 pitch 6.25 mm 的规则格层之间放置一个闭合 2 mm 薄盒：Open3D 连续几何查询本身正确，独立随机点分类没有分歧，但规则格点的内部占据数恰好是 **0**。Subdivide 和 SDF 表面带各保留 2,646 个格点。不能把“非边界格点零误判”理解为细薄实体已经被表示出来。

类似地，一个与主体分离的 3 mm 小闭盒没有任何内部格点。对其中心所在体素的探针，严格中心 mask 为 false，而 subdivide 和 SDF 表面带为 true。本轮只以已知实心盒中心作诊断；任意 mesh 的质心或包围盒中心可能落在外部，不能据此普遍证明某部件存在或缺失。

### 空腔与相交部件需要明确实体语义

嵌套闭盒采用明确的奇偶壳语义时，Open3D occupancy 与独立随机点参考在非边界处无分歧，中心探针正确识别空腔；subdivide 的填充把空腔中心也标成占据。

相交双盒的目标是两个实体的并集，不能用整份三角 soup 的奇偶内外替代。在本轮测试中，整体 Open3D sign 对并集参考产生 3,315 个非边界格点漏判和 644 个误判；10,000 个独立随机点有 1,329 个分歧。该输入违反其官方前提，这不是可忽略的 Float32 舍入误差。

[Open3D 官方 API](https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html)要求占据与 SDF 的内外语义明确、mesh 闭合且不同 mesh 不相交；查询使用 Float32，`nsamples` 必须为奇数，增加射线数可缓解穿过边或顶点时的误判。本轮使用 3 条射线，并在统计时排除绝对距离小于 `max(pitch×1e-4, longest_extent×2e-6)` 的边界格点；排除数量记录在 JSON 中，不把这些位置算作已验证正确。

### Open3D 的 CUDA 可用性不能混同于 RaycastingScene 的设备支持

本机 `open3d.core.cuda.is_available()` 为 true，但 [RaycastingScene 的设备契约](https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html)仅包含 CPU 和 SYCL。当前 SYCL 不可用，因此本轮全部 raycasting 在 `CPU:0` 上完成，没有测试或声称 Open3D CUDA SDF。

## 可落地的后端接口建议

建议返回有明确字段的准备对象，而非一个意义不明的 `solid` 数组：

```text
pitch_m, origin_m, grid_shape
strict_interior_mask
surface_band_mask
signed_distance_m
solid_interpretation: single / explicit-parity / original-part-union
boundary_tolerance_m
missing_feature_diagnostics
```

传给 Open3D 前应平移到 mesh 局部中心，并按最长边缩放，再转换 Float32；返回距离时还原物理单位。不要让世界坐标的大平移吞掉毫米级细节。查询点可分块；网格单元数应在分配数组前限制。

对单一闭合无自交表面，可提供显式选择的 occupancy 或 SDF 准备后端；先保留当前默认，独立比较完整拟合的漏覆盖和外扩。对于原 mesh 距离半径，可以从内部格点取 `-signed_distance`，这样还可能避免 EDT 的栅格边界偏差，但其 Float32 查询和候选采样误差仍须报告。

若 `strict_interior_mask` 为空或小组件没有有效候选，不应静默删除几何：可明确要求更高分辨率，或切换到保留表面的 cover/slab 路线。SDF 表面带是另一种显式采样解释，不是严格内部采样的无误差修复。

URDF 多 collision 的并集需要保留原部件分组，再分别确定每个部件的实体解释后取并集。直接按所有连通闭壳做 OR 会填实某些部件内部的嵌套空腔，因此不能作为默认的通用多壳修复。

## 复现与范围

```bash
# 需要当前环境可导入 open3d 和一个 trimesh 可用的 contains/ray 依赖
OMP_NUM_THREADS=1 .venv/bin/python benchmarks/explore_voxelization.py \
  --repeats 3 --threads 1 --output outputs/explore-voxelization.json
```

每个方法使用相同格点，另有种子 99173 的 10,000 个独立随机点，仅用于查询正确性检查。没有据此调拟合参数，也没有将准备阶段时间称为完整球化时间。最终是否接入默认路径，还需要父流程对生成球、特征遗漏和下游误碰撞进行验证。[Open3D 距离教程](https://www.open3d.org/docs/release/tutorial/geometry/distance_queries.html)也明确区分无符号最近距离与需要闭合实体语义的有符号/占据查询。
