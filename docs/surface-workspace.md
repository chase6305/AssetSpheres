# 表面体素化临时数组优化

默认 `subdivide` 后端按批次把三角形拆为四个子三角形。旧实现先分配四份子三角形数组，再用
`np.stack` 分配组合数组，组合时两套数组同时存活。现在直接分配最终缓冲区并填入相同顶点，
入队前释放选中三角形和中点临时数组。中点运算、子三角形顺序、批次大小、舍入规则、
深度限制与网格预算保持原样；适用于使用该后端的 `voxel`、`refined` 和 `refined-cover`。

## 阶段测量

2026-11-09，相对 main `9dab365`，Python 3.10.0 / NumPy 1.26.4 / Trimesh 4.12.2。
下表是单独表面体素化阶段的 9 次交替配对测量中位数。网格创建、导入及预热在计时外，
源网格的三角形缓存已预热。分配峰值在计时之外用 tracemalloc 单独测量，不是 RSS、
原生库总内存或完整拟合峰值。主机未独占，时间只作参考。

| 输入 / 最长轴分辨率 | 时间，之前 → 之后 (ms) | 分配峰值，之前 → 之后 (MB，十进制) |
| --- | --- | --- |
| box / 48 | 24.69 → 23.07 | 10.63 → 8.86 |
| box / 96 | 95.42 → 90.79 | 13.01 → 11.24 |
| cylinder / 48 | 163.58 → 159.82 | 14.18 → 12.41 |
| sphere / 48 | 12.58 → 11.60 | 9.86 → 8.09 |

这些输入的分配峰值下降约 12%–18%，耗时中位数下降约 2%–8%。不据此推断完整拟合、
真实机器人或其他后端的提速；这里只减少构造子三角形时的临时空间，没有改变渐近复杂度。
原始逐对数据见 [surface-workspace.json](surface-workspace.json)。

```bash
git show 9dab365:src/assetspheres/surface_grid.py > /tmp/surface-before.py
PYTHONPATH=src python benchmarks/benchmark_surface_workspace.py \
  --baseline /tmp/surface-before.py --output /tmp/surface-workspace.json
```

## 正确性验证

- 原有六类几何与大坐标平移测试比较完整表面格点、填充格点及变换。
- 新增不同大小的旋转组件，覆盖同批次内已完成与待细分三角形混合的路径；批次大小 1、7、8192
  均与独立的整网格细分参考实现逐元素一致。
- box / cylinder / sphere，分别用 `voxel` / `refined` / `refined-cover`、8 球、分辨率 24、
  256 样本完成 9 组新旧完整拟合对照，球参数及报告完全相等。
- 主环境全量测试：478 passed / 13 skipped。最低依赖环境的 surface-grid、voxelization、
  fitting 测试：45 passed / 23 skipped（可选后端不可用）。Ruff 检查和格式检查通过。
