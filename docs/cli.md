# 命令行接口

CLI 以“先检查资产，再选择算法，最后查看或导出结果”为常用顺序。所有球心使用所属 link 的局部坐标系，长度单位为米；独立 mesh 的毫米单位需要显式缩放。

## 最短流程

```bash
# 查看各算法的用途、覆盖范围和运行环境
assetspheres methods

# 无需拟合即可检查 link、几何来源、闭合性与尺寸
assetspheres inspect examples/two_link.urdf

# 省略 -o 时，输出到当前目录下 outputs/two_link/
assetspheres fit examples/two_link.urdf --method slab

assetspheres report outputs/two_link/spheres.json
assetspheres view outputs/two_link/spheres.json
```

拟合或重新导出成功后，会打印结果目录和已正确处理路径空格的 `report` / `view` 命令。已有输出需要显式 `--force`，默认不会覆盖。

`fit` 在拟合前预检已有输出。`--force` 仍不能覆盖源文件及 URDF 引用的 mesh/纹理。查看器自动取景，`Frame model` 可重新对准模型；勾选 `Only selected link` 时只对准该 link。

## 命令概览

| 命令 | 输入 | 用途 |
| --- | --- | --- |
| `methods` | 无 | 比较算法、闭合 mesh 要求、支持设备和依赖 |
| `inspect INPUT` | URDF 或 mesh | 读取资产几何，检查 link、三角形数量、闭合性与尺寸 |
| `fit INPUT` | URDF 或 mesh | 生成碰撞球、质量报告、URDF 与预览 |
| `report RESULT` | `spheres.json` | 显示已保存的质量报告 |
| `view RESULT` | `spheres.json` | 启动可选的 Viser 查看与编辑界面 |
| `export RESULT -o DIRECTORY` | 原始或编辑后的结果 JSON | 重新生成规划器球片段、URDF 和预览 |

每个命令均支持 `--help`。`fit --help` 将参数分成输入几何、输出文件、拟合算法、CPU/CUDA、体素化和实验优化六组，并显示默认值。

## 检查输入

```bash
assetspheres inspect robot.urdf --geometry collision
assetspheres inspect robot.urdf --link forearm --link wrist
assetspheres inspect robot.urdf \
  --package-root robot_description=/path/to/robot_description
assetspheres inspect part.stl --scale 0.001
```

`inspect` 不运行球化，也不生成输出文件。它显示参与加载的几何 link；空 link 或被 `--link` 排除的 link 不计入 `link_count`，完整运动学树的 link 数另外记录在 JSON 的 `tree_link_count`。`body_count` 是连通表面体数量，不等于 URDF collision 部件数量。闭合性是三角网格拓扑指标，不能据此自动认定没有自交或空腔解释歧义。

默认 `--geometry auto` 优先 collision，没有 collision 时才用 visual；也可明确选择 `collision` 或 `visual`。独立 mesh 的 `--scale` 在加载时应用，检查和拟合必须使用相同单位设置。

## 配置球化

```bash
assetspheres fit robot.urdf --method voxel --count 24 \
  --link-count wrist=12 --padding 0.002 -o outputs/robot

assetspheres fit part.stl --method cover-tight --scale 0.001 -o outputs/part

assetspheres fit robot.urdf --method cover-opt --count 24 --padding 0.003

assetspheres fit robot.urdf --method slab --workers 4 --executor thread

assetspheres fit robot.urdf --method refined --device cuda \
  --iterations 120 --refine-samples 2048 --outside-weight 10
```

`--count` 是每个 link 的球数上限；`--link-count LINK=N` 可重复，为某个 link 单独覆盖。`--padding` 是最终球半径的米制增量。`--link`、`--package-root` 同样可以重复，重复定义同一个键会明确报错。

默认 `--device cpu --workers 1`。线程/进程适用于不同 link 的 CPU 拟合；CUDA 路径由一个 GPU 执行者处理。小任务不一定因并行而变快，完整性能取舍见 [performance.md](performance.md)。

`cover-opt` 在 `cover-tight` 后最多执行两轮完整三角形重分组与包围球缩紧，只接受最终
球体积之和更小的结果。它支持开口 mesh，仅在 CPU 运行；表面包络不代表实体内部也被覆盖。
实测和复现命令见 [优化对比](cover-optimization.md)。

体素化相关参数集中在同一组：

- `--resolution`：最长轴的体素数量，默认 48。
- `--voxel-size`：显式米制体素边长，覆盖 `--resolution`。
- `--voxelizer`：默认 `subdivide` 保留原 CPU 路径；`open3d` 使用可选 CPU 几何查询运行时，`warp` 使用可选 CPU/CUDA 查询运行时。具体可用性由已安装依赖和运行时决定。
- `--voxel-solid-policy`：新几何查询后端的实体解释，默认 `single`；`union-components` 将各闭合组件视为实体再取并集，会填掉嵌套壳表示的空腔。它不会自动恢复原 URDF collision 的部件分组。

这些选项不会自动替用户更换算法。开放 mesh 的体积后端默认报错，可修复资产，或显式指定 `--non-watertight surface` / `cover` 回退。

新距离后端会为每个有内部样本的组件保留一个最深代表点。球预算不能覆盖这些代表点，或 refined 优化把某个代表点丢掉时，会报出可行动错误；它不等于整个组件的覆盖证明。

Open3D / Warp 是可选的原 mesh 查询后端，以网格中心的占据状态构造体素；原 `subdivide` 则离散表面后填充。两类流程不保证生成相同的占据点或球模型，尤其在薄壁、小部件与空腔边界附近。薄到没有内部网格中心时会明确报错，应调整分辨率或显式选择另一种拟合方法，不会自动把结果改成表面球。

## JSON 与脚本调用

```bash
assetspheres methods --json
assetspheres inspect robot.urdf --json > asset-info.json
assetspheres report outputs/robot/spheres.json --json > quality.json
```

以上命令的标准输出只包含 JSON，不附带人类阅读用表格。`methods` 描述的是支持能力，不会因为 CUDA 被列出就表示这台机器已安装可用运行时。`report` 读取已保存的测量值，不重新计算质量；手动修改球后的旧指标会标记为失效。

`fit` 生成的 `spheres.json` 是脚本消费结果的主要接口。人类阅读用表格可以随易用性调整而变化，自动化应读取 JSON 文件或使用上述 `--json` 命令。

正常完成返回状态码 0；无效参数、输入不存在、依赖不可用或拟合/导出错误返回 2。帮助和版本查询返回 0。错误写到标准错误，预期的用户输入错误不输出 Python traceback。

原有 `fit -o`、`view`、`report --json` 和 `export -o` 用法全部保留；`fit` 新增可省略输出目录的便利行为，`export` 仍要求显式目标目录。
