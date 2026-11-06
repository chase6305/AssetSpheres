# Python API

常用流程是 `load_asset → FitConfig → fit_asset → write_outputs`。以下示例从仓库根目录运行，基础流程只需 `pip install -e .`；不要求 ROS、PyTorch、Open3D 或 Warp。

## 先跑通一个资产

```python
from pathlib import Path

from assetspheres import FitConfig, fit_asset, load_asset
from assetspheres.export import write_outputs

input_path = Path("examples/two_link.urdf")
asset = load_asset(input_path)
configuration = FitConfig(method="cover-tight", count=16, padding=0.002)
result = fit_asset(asset, configuration)
output_paths = write_outputs(asset, result, "outputs/api-demo")

for output_path in output_paths:
    print(output_path)
```

这里选择 `cover-tight` 演示三角表面包络，`count=16` 表示**每个 link 最多 16 个球**，`padding=0.002` 表示半径增加 **2 mm**。这不是整台机器人的总球数，也不要求每个 link 用满预算。表面包络不自动保证实体内部覆盖；其他方法的用途见 [算法说明](algorithm.md)。

默认输出 JSON、cuRobo 球片段、Lula 球片段、球碰撞 URDF 和 GLB。再次写入同一目录时，已有同名文件会触发 `ValueError`；确认需要替换后，可显式调用 `write_outputs(..., force=True)`。

`force=True` 也不能覆盖源资产及其 URDF mesh/纹理引用。长任务可先调用 `assetspheres.export.check_output_paths(asset, directory, force=False, include_glb=True)` 预检，返回目标路径列表；它不创建目录或文件，`write_outputs` 写入前仍会再次检查。

## 读取资产与选择几何

`load_asset` 返回 `Asset`，其中 `asset.links` 是 `dict[str, LinkGeometry]`，每份 `LinkGeometry.mesh` 都已经转换到所属 link 的局部坐标系。

```python
asset = load_asset(
    "examples/two_link.urdf",
    geometry="collision",
    link_names=["base", "arm"],
)

for link_name, link_geometry in asset.links.items():
    print(link_name, link_geometry.geometry_source, link_geometry.mesh.extents)
```

| 参数 | 含义 |
| --- | --- |
| `geometry="auto"` | 按 link 优先读取 collision；该 link 没有 collision 时才读取 visual |
| `geometry="collision"` / `"visual"` | 明确选择来源，不混合两种来源 |
| `link_names=[...]` | 只加载指定 link 的几何；仍保留 URDF 结构和装配变换供导出使用 |
| `package_roots={"包名": "目录"}` | 为 `package://包名/...` 指定该包的根目录 |
| `scale=0.001` | 独立 mesh 以毫米建模时转换成米；不用于整体缩放 URDF |

输入路径相对于 Python 进程的当前工作目录解析；URDF 内的相对 mesh 路径相对于 **URDF 所在目录**解析。`package_roots` 的值是 package 自身的目录，不是它的 `urdf/` 子目录。Xacro 需先展开成 URDF。

URDF 采用米和弧度，并应用各 mesh 的 `scale`、各 geometry 的 `origin xyz/rpy`。球和圆柱等解析几何先三角化，因此后续几何保证针对参与拟合的三角 mesh。

同一次加载中，相同 `filename` 的重复 mesh 引用只解析一次，各实例仍独立应用缩放和位姿。缓存不跨 `load_asset` 调用保留，也不为只引用一次的 mesh 留下额外副本。

## 查看球与报告

沿用第一段的 `result`：

```python
arm_spheres = result.links["arm"]
first_sphere = arm_spheres[0]

print(first_sphere.center)  # tuple[float, float, float]，link 局部坐标，米
print(first_sphere.radius)  # float，米
print(first_sphere.to_dict())  # {"center": [...], "radius": ...}

arm_report = result.reports["arm"]
print("实际球数:", arm_report["sphere_count"])
print("表面样本覆盖:", arm_report["surface_coverage"])
print("p95 正间隙，米:", arm_report["surface_gap_p95_m"])
```

每个球是有字段名的 `Sphere`，不用记忆一个裸数组的哪一列是半径。球心不包含 joint origin；`result.transforms[link_name]` 是该 link 在 root 坐标系中的零位齐次位姿矩阵，可将局部球心转换到 root 坐标，不应再次烘焙进导出的局部球心。

`Sphere` 会复制球心并保存为不可变的原生浮点数。配置的长度与权重、资产缩放及 `FitResult.from_dict` 已验证的浮点字段也会规范化，支持 NumPy 实数标量；球数、采样数等整数字段仍要求 Python `int`。自定义报告字段需调用方提供可 JSON 序列化的值。

`FitResult` 的常用字段如下：

| 字段 | Python 类型 / 用途 |
| --- | --- |
| `links` | `dict[str, list[Sphere]]`，实际处理的 link 及其球 |
| `reports` | `dict[str, dict]`，每个 link 的方法、预算、实际球数、质量和适用边界 |
| `config` | 全局 `FitConfig`；每 link 覆盖后的预算看其报告中的 `requested_count` |
| `source` | 原始资产路径 |
| `transforms` | `dict[str, list[list[float]]]`，link 在 root 下的 4×4 零位位姿矩阵 |
| `asset_options` | 重新加载资产需要的几何来源、单位缩放和 package 映射 |

没有几何或未被选择的 link 可能不在 `result.links` 中，导出的 URDF 仍会保留原结构。使用手动编辑后的报告前，先检查 `report.get("quality_metrics_stale", False)`；旧采样质量不能当作修改后的结果。

## 控制预算、采样与并行

```python
parallel_result = fit_asset(
    asset,
    FitConfig(method="cover-tight", count=16, samples=3000, seed=7),
    link_counts={"base": 12, "arm": 24},
    workers=4,
    executor="thread",
)

for link_name, spheres in parallel_result.links.items():
    print(link_name, len(spheres))
```

`link_counts` 只覆盖对应 link 的球数上限；键必须是已加载的 link。`workers=1` 为默认串行执行，线程和进程均保持结果的 link 顺序；小资产不一定因增加 worker 而变快。`executor="process"` 使用 spawn，需要在正常的 `.py` 脚本中调用，并将入口放在 `if __name__ == "__main__":` 下。

进程模式对普通 Trimesh 只发送顶点、面索引和 metadata，源视觉与自定义属性留在父进程。
自定义网格子类沿用原对象传输路径。完整生命周期成本与适用范围见[拟合流水线实测](fitting-pipeline.md)。

| 配置 | 默认值 | 作用 |
| --- | ---: | --- |
| `count` | 32 | 每 link 的球数上限 |
| `padding` | 0.0 | 增加到候选或输出球半径的非负裕量，米 |
| `samples` | 6000 | 表面评价样本数；也影响 surface 候选池和 voxel/refined 初始覆盖的采样目标 |
| `seed` | 0 | 拟合采样种子，不修改 NumPy 全局随机状态 |
| `resolution` | 48 | 体素边长按最长包围盒边 / resolution 计算 |
| `voxel_size` | `None` | 显式体素边长，米；设置后优先于 resolution |
| `refine_samples` | 2048 | refined 每组训练点数上限：表面、内部、外部各一组 |
| `iterations` | 120 | refined 优化迭代次数 |
| `outside_weight` | 10.0 | refined 对外部误覆盖的损失权重，不能保证增大后覆盖率也提高 |

`samples` 和 `refine_samples` 不同：前者用于已有拟合/评价流程，后者只控制 refined 的训练数据规模。覆盖率是有限样本统计；`sphere_volume_sum_m3` 是含重叠重复计数的各球体积之和，不是球并集体积。

`FitConfig` 不可原地修改。需要派生配置时使用 `dataclasses.replace(configuration, count=32)`，而不是修改 `configuration.count`。

## 优化后修补三角表面

闭合 mesh 可用 `FitConfig(method="refined-cover")` 组合优化与表面修补；球数不增加，
padding 只添加一次。沿用已有资产：

```python
repaired_result = fit_asset(asset, FitConfig(method="refined-cover", count=24, padding=0.003))
```

已有球模型也可单独调用 `assetspheres.surface_cover.repair_surface_cover(mesh, spheres)`，
返回新球列表和修补诊断，支持开口三角 mesh。它不会移动球心或缩小半径，原 mesh 和球列表不变。
相对原模型，外扩可能增加；表面包络不等于实体包络。该函数不重新评价调用方保留的旧质量报告。
详细实验见 [修补说明](refined-cover.md)。

## 单独拟合一个 mesh

已有 `trimesh.Trimesh` 时可以直接调用 `fit_mesh`：

```python
import trimesh

from assetspheres import FitConfig, fit_mesh

triangle_mesh = trimesh.creation.cylinder(radius=0.05, height=0.3, sections=32)
spheres, report = fit_mesh(
    triangle_mesh,
    FitConfig(method="slab", count=16, padding=0.002),
)

print("球数:", len(spheres))
print("表面覆盖:", report["surface_coverage"])
```

返回值是 `(list[Sphere], dict)`，不包含 URDF 或导出路径。此时坐标系就是传入 mesh 的坐标系，单位由调用方统一为米；函数不会猜测单位，也不会修改输入 mesh。若还需要标准文件导出，可通过 `load_asset("part.stl", scale=0.001)` 读取独立 mesh，再使用完整资产流程。

## `device` 与 `voxelizer` 分别选择什么

`voxelizer` 选择体积样本和候选半径的准备方式，仅影响 `voxel` / `refined`；`device` 选择后续张量步骤和 refined 优化的执行设备。二者不是同一个开关。

| `voxelizer` | 几何准备 | 可选依赖 | `device="cuda"` 时 |
| --- | --- | --- | --- |
| `subdivide`（默认） | CPU 三角细分、壳层填充、EDT，保留已有行为 | 无 | 后续候选选择、点球距离和优化可用 CUDA；几何准备仍在 CPU |
| `open3d` | 原 mesh 的内部中心采样和距离查询，CPU | `.[open3d]` | Open3D 查询仍在 CPU，后续张量步骤可用 CUDA |
| `warp` | 原 mesh 的内部中心采样和距离查询，CPU 或 CUDA | `.[warp]` | Warp 几何查询及后续张量步骤可用 CUDA；张量步骤另需兼容的 PyTorch |

基础示例不要求这些可选依赖。安装相应 extra 后，可替换配置，例如：

```python
open3d_configuration = FitConfig(method="voxel", voxelizer="open3d", device="cpu")
warp_cpu_configuration = FitConfig(method="voxel", voxelizer="warp", device="cpu")
warp_cuda_configuration = FitConfig(method="voxel", voxelizer="warp", device="cuda")
```

`device="cuda"` 目前只用于 `voxel` / `refined`，并要求可运行 CUDA 的 PyTorch；明确请求后不会因设备不可用而静默切回 CPU。CUDA 资产拟合使用一个 GPU 执行者，保持 `workers=1`。Warp CPU 查询本身不需要 PyTorch。

Open3D / Warp 与默认填充栅格的语义不同：前者在原三角 mesh 的内部网格中心采样并查询距离，不保证得到与 subdivide 相同的球；极薄组件没有内部样本时会显式报错。查询使用归一化的 float32 几何和数值裕量，不能当作精确 SDF 或覆盖证明。

### 显式选择多闭合壳的含义

`voxel_solid_policy` 用于 Open3D / Warp 的内部查询；默认 subdivide 保留旧填充行为。

- `"single"`（默认）：要求一个闭合表面体；遇到多个壳时拒绝猜测其空腔/并集含义。
- `"union-components"`：每个闭合连通壳都当作实体，再取并集。相交组件可如此解释，但嵌套空腔会被填实。

```python
union_configuration = FitConfig(
    method="voxel",
    voxelizer="open3d",
    voxel_solid_policy="union-components",
    count=24,
)
```

这些组件是 mesh 的连通壳；不能把这一策略当成任意自交或空腔资产的自动修复。非闭合 mesh 的体积拟合默认报错，只有显式设置 `non_watertight="surface"` 或 `"cover"` 才会切换方法；也可直接选择表面方法。

距离后端会检查每个组件至少有内部格点，并把最深内部代表点加入候选和必需覆盖目标。预算不足或优化后丢掉代表点会报错；这保证的是代表点不被忽略，不是完整组件包络。位于大坐标偏移处的资产还会检查格距与坐标精度，并对半径考虑坐标 ULP；必要时应先把几何移近原点。

## 用原 mesh 公平比较多个球模型

先安装 `pip install -e '.[quality]'`，然后让所有候选模型共用独立评价样本和同一评价域：

```python
import trimesh

from assetspheres import FitConfig, fit_mesh
from assetspheres.quality import compare_sphere_models

reference_mesh = trimesh.creation.cylinder(radius=0.05, height=0.3, sections=32)
voxel_spheres, _ = fit_mesh(reference_mesh, FitConfig(method="voxel", count=16))
slab_spheres, _ = fit_mesh(reference_mesh, FitConfig(method="slab", count=16))

quality_by_method = compare_sphere_models(
    reference_mesh,
    {"voxel": voxel_spheres, "slab": slab_spheres},
    surface_samples=3000,
    volume_samples=10000,
    seed=88019,
)

for method_name, metrics in quality_by_method.items():
    print(method_name, metrics["surface_coverage"], metrics["volume_iou"])
```

所有球必须与 `reference_mesh` 在同一坐标系中，使用相同预算和 padding 比较才有意义。评价域包含原 mesh 和所有比较模型的全部球，避免漏掉包围盒之外的外扩。

- `volume_coverage`：原实体被球并集覆盖的比例。
- `outside_volume_over_mesh`：球并集外扩体积 / 原 mesh 体积，可超过 100%。
- `sphere_precision`：球并集中位于原实体内的比例。
- `volume_iou`：原实体与球并集的交并比。

体积指标是 Monte Carlo 估计，结合 `validation_interior_points` 查看有效内部样本量。开口 mesh 只报告表面指标；多壳在默认 `solid_policy="single"` 下跳过体积指标。评价函数另可显式选择 `solid_policy="parity"` 或 `"union-components"`，与体素准备的策略参数分开设置；前者解释为奇偶壳，后者会填实嵌套空腔。自交检测尚未实现，不能仅凭闭合标记确认实体语义。

## 保存、重载与错误处理

```python
from assetspheres.export import load_result_asset, read_result

saved_result = read_result("outputs/api-demo/spheres.json")
source_asset = load_result_asset(saved_result)
print(saved_result.links.keys())
```

`result.to_dict()` 返回可序列化的 schema version 1 数据；`FitResult.from_dict(data)` 可还原对象。已有公共入口、参数名及旧结果读取接口保持兼容；旧配置未包含新增可选字段时使用其默认值。

读取会检查球列表、数值、刚性零位变换、报告与资产选项；无效结构和非有限 JSON 数字会触发 `ValueError`。`Sphere` 拥有不可变的球心 tuple，不保留调用方可修改的列表或 NumPy 数组。

导出的 URDF 会按新目录重写保留的 mesh 和 URDF material 纹理相对路径，**不复制 mesh 或纹理**。移动结果时需维护原资产位置；原资产已移动可通过 `load_result_asset(saved_result, path="新位置")` 和 `package_roots=...` 覆盖来源。两份规划器 YAML 是球配置片段，仍需与目标规划器的运动学和关节配置组合。

入口脚本可以按需捕获 `ValueError`、`OSError` 和 `ImportError`，保留原错误信息。常见处理方式如下：

| 问题 | 处理 |
| --- | --- |
| 找不到源文件 / package mesh | 检查工作目录、URDF 相对路径与 `package_roots` |
| 球数覆盖的 link 未加载 | 检查 `link_names` 与 `link_counts` 的键 |
| 体素预算超限 | 增大 `voxel_size` 或降低 `resolution` |
| 距离后端漏采到薄组件 | 降低体素边长，或改用 subdivide / 表面方法，不静默丢掉该组件 |
| 输出同名文件已存在 | 换目录，或明确传 `force=True` |
| 缺少可选后端 / CUDA 不可用 | 安装对应 extra 和兼容运行时，或显式选择 CPU 路线 |

CLI 使用方式见 [cli.md](cli.md)，算法与性能取舍见 [performance.md](performance.md)。
