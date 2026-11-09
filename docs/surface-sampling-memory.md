# 大样本表面采样内存优化

`sample_surface` 用面积加权选择三角形，再生成均匀重心坐标。之前会一次性展开所有选中
三角形、随机偏移和插值临时数组。现在保留全部面索引与最终输出，超过 32,768 个样本时
分批生成偏移并插值。面索引仍先全部抽取，再依次消耗同一均匀随机数流，保留样本顺序、
浮点运算顺序及固定种子下的结果。小样本继续单批计算。

这影响 surface 拟合、各方法的质量报告、refined 训练及独立质量评价中的表面采样。
不改变面积加权分布、配置、输出格式或全局 NumPy 随机状态。最终输出与面索引仍需随样本数
线性增长；输入网格的 Trimesh 几何缓存也仍存在，此优化只约束采样计算的临时数组。

## 阶段测量

2026-11-09，相对 main `db6efc1`，Python 3.10.0 / NumPy 1.26.4 / Trimesh 4.12.2。
输入为半径 0.1 的 subdivisions=3 icosphere，种子 831。9 次交替配对测量，网格构造、
导入和缓存预热在计时外。分配峰值由 tracemalloc 在计时外另测，不是 RSS 或完整拟合峰值。
主机未独占；时间仅代表该采样阶段与该输入。

| 样本数 | 分配峰值，之前 → 之后 (MB，十进制) | 耗时中位数，之前 → 之后 (ms) |
| --- | --- | --- |
| 10,000 | 1.76 → 1.76 | 1.11 → 1.11 |
| 100,000 | 16.97 → 8.54 | 11.89 → 10.75 |
| 1,000,000 | 169.07 → 37.34 | 247.06 → 97.49 |

大样本案例的分配峰值下降约 50% / 78%。默认 6,000 样本在单批路径内，不据此宣称
默认拟合或真实机器人的总体提速。逐对原始数据见 [surface-sampling-memory.json](surface-sampling-memory.json)。

```bash
git show db6efc1:src/assetspheres/geometry.py > /tmp/sampling-before.py
PYTHONPATH=src python benchmarks/benchmark_surface_sampling.py \
  --baseline /tmp/sampling-before.py --output /tmp/sampling.json
```

## 验证

- 样本数 0、1、32,767、32,768、32,769、70,001，在原点与 1e12 平移坐标下，逐元素等于
  原来的整批参考实现；输入顶点、面和全局随机状态保持一致。
- box / sphere × surface、cover、voxel、refined、refined-cover、slab，33,001 验证样本，
  refined 训练样本也设为 33,001，共 12 组完整拟合。新旧球参数和报告完全相等。
- 全量测试：491 passed / 13 skipped。
- 最低依赖及 NumPy 2 环境的采样与拟合测试均为 36 passed；Ruff、格式检查及 sdist / wheel 构建通过。
