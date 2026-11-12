# 拟合计算的临时工作区

本轮针对点球距离、体素距离重建、体素坐标转换和三角表面修补的临时数组，复用或分块分配工作区。
拟合配置、球参数、报告格式、采样种子和覆盖含义沿用既有约定，没有新增依赖。

- **点球距离。** 保留每块最多 2,048 点、128 球，首块由 SciPy 决定浮点类型，随后复用其连续缓冲。
  点和球两个维度的短尾块都取缓冲前缀再重塑，避免不连续 `out`。
  每块仍减去半径、取最近球，并写回 float64 输出；保留 longdouble 输入的扩展精度距离和原舍入顺序。
- **体素距离。** SciPy 的最近背景格索引保持原样；平方距离按原轴序累加。
  offset 缓冲仅容纳最多 32,768 个格，或一个首轴切片（取较大者）；每轴还需长度随该轴增长的坐标向量。
  输出和最近格索引仍随整个网格增长；单切片已占满首轴的输入没有这项内存收益。
- **体素坐标。** 仅在最后一次索引用途之后，复用本次调用独占的、可写的原生 64 位 intp 列优先大缓冲。
  逐块完成整数坐标读取后再写入 float64 点位；保持原来的分块大小、列优先布局及 Trimesh 计算顺序。
  小输入、只读或其他 dtype/布局继续沿用原转换；原转换 helper 仍不修改调用者的输入。
- **表面修补。** 顶点超过 32,768 时，用坐标界求相同的 longdouble 平移尺度，
  坐标界采用逐列归约，再复用一个顶点块做扩展精度减法、除法，写入完整 float64 归一化顶点数组。
  小输入保留原路径；修补遍历、预算、球心、半径增长和浮点余量保持原样。

## 实测口径

基线 `15ecb68`，实现 `5c8d35f`。以下数据为 11 次交替配对的 warm call，
导入和输入创建不计时；计时与内存测量分开。
所有计时在同一台共享主机上完成，数值库单线程。分配峰值用 tracemalloc，MB 为十进制，
不代表 RSS、第三方原生内存、显存或整个程序的上限。不能把阶段收益外推到所有完整拟合。
[原始记录](fitting-workspaces.json)保留版本、源码哈希、逐次耗时、输出指纹与全部案例。

| 隔离对照范围 / 输入 | CPU 时长：之前 → 之后 | 分配峰值：之前 → 之后 |
| --- | ---: | ---: |
| 点球距离，6,000 点 × 128 球 | 1.73 → 1.71 ms | 4.25 → 2.23 MB |
| 仅替换距离函数的完整 cover，128 球、6,000 样本 | 8.88 → 8.55 ms | 4.61 → 2.60 MB |
| 距离重建，114³ 格 | 97.05 → 98.40 ms | 41.62 → 29.97 MB |
| 仅替换体素准备的完整 voxel，cube112、32 球 | 711.91 → 702.97 ms | 83.71 → 74.25 MB |
| 仅替换体素准备的完整 refined，cube112、32 球 | 858.73 → 864.24 ms | 83.71 → 74.25 MB |
| 完整表面修补，327,680 面、1 个已包络球 | 54.48 → 51.74 ms | 19.66 → 7.93 MB |

cube112 有 1,442,897 个占据点，完整调用峰值实际减少约 9.46 MB；
复用点缓冲的收益仍受其他阶段限制。表面修补一行只衡量修补阶段，初始化不在测量内。
完整 refined-cover 的组合效果需要使用独立旧包对照，不能把表中的阶段差值相加。

小输入及其他瓶颈案例也保留在原始记录中：128 点 × 16 球的距离峰值约 0.037 MB，
单切片 1 × 256 × 256 距离重建约 1.97 MB，cube24 完整拟合约 8.88 MB，均基本不变。
100,000 样本的完整 surface / cover 峰值仍约 8.76 MB，采样阶段会主导峰值。
耗时总体接近，包含略慢的案例；隔离完整拟合的旧路径还包括 patch 上下文开销，
微小时间或几 KB 的差别不作为算法收益。

## 验证

冻结源码在主环境为 **652 passed**，最低依赖为 **579 passed / 42 skipped**，
Python 3.12 / NumPy 2 / Trimesh 5 为 **580 passed / 41 skipped**。
最低依赖保留两条原有 SLSQP 路径的试探步裁剪警告。
新增检查涵盖分块边界、短尾、只读与非连续数组、longdouble、带符号零、次正规数、
极端坐标，以及完整球参数、报告和源 mesh 不变的对照。

## 复现

准备独立基线后，在资源允许时串行运行。与其他任务共享主机时，先检查当前内存、负载和进程，
按本地资源约定使用共享排他锁。

```bash
git worktree add --detach /tmp/assetspheres-workspace-before 15ecb68
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
baseline=/tmp/assetspheres-workspace-before/src/assetspheres
python benchmarks/benchmark_gap_workspace.py --reference "$baseline/geometry.py" \
  --repeats 11 --full-fit --output outputs/gap-workspace.json
python benchmarks/benchmark_grid_workspace.py --baseline "$baseline/grid_distance.py" \
  --repeats 11 --full-fit --output outputs/grid-workspace.json
python benchmarks/benchmark_voxel_memory.py --reference "$baseline/voxel_fit.py" \
  --repeats 11 --full-fit --output outputs/voxel-workspace.json
python benchmarks/benchmark_repair_workspace.py --baseline "$baseline/surface_cover.py" \
  --repeats 11 --output outputs/repair-workspace.json
```

每个基准只替换对应的基线模块或函数，以隔离该项改动；其他步骤使用当前实现。
完整组合对照另行记录，不能将孤立基准的峰值差直接相加。
