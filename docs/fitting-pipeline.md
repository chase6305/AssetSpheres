# 拟合准备与进程传输

这轮优化复用拟合副本的几何缓存，减少闭合性报告的临时数组，并缩小进程任务。
拟合方法、球预算、采样种子和结果格式沿用原有约定；没有增加核心依赖或配置项。
[完整配对记录](fitting-pipeline.json)保留每次耗时、输入状态、库版本、源码哈希和完整结果指纹。

## 实现

- **面积只在副本上计算。** 顶点有限性仍先检查；面积验证与退化面清理共用独立副本的三角形缓存。
  没有预计算面积的源网格不再额外保留整份三角形、叉积和面积缓存。
- **闭合性报告。** 对普通 Trimesh，把无向边的两个索引编码为 64 位整数，排序后检查每条边恰好出现两次。
  重复四次也判为不闭合，不能仅检查计数为偶数。空面、负索引、超出编码范围和自定义子类沿用原判断。
  这用于 surface/cover 系列的报告；体积方法和 slab 继续复用原有拓扑与朝向检查。
- **进程任务。** 普通 Trimesh 只传顶点、面索引和 metadata，排除视觉、自定义属性和运行时缓存。
  父进程持有数组引用，序列化前不额外复制几何；worker 保持索引顺序，再按原流程清理独立副本。
  自定义网格子类沿用原对象路径，保留 `copy()` 等行为。metadata 仍随任务传递。

线程、单 worker、link 顺序和 CUDA 单所有者约束保持一致；进程池仍使用 spawn，每次调用独立管理生命周期。
闭合性与 Trimesh 的边计数判据一致，不增加朝向、自交或实体内部覆盖保证。

## 配对实测

基线 `83c4df6`，实现 `b0f3f98`；AMD Ryzen 9 9950X、Python 3.10.0、
NumPy 1.26.4、SciPy 1.15.3、Trimesh 4.12.2。固定逻辑核，OpenBLAS/OMP 各 1 线程，主机未独占。
采用 11 次交替配对中位数，输出必须完全一致；加载、输入恢复和导出不计入拟合时间。

![实测的完整拟合分配峰值、surface 拟合耗时与完整进程生命周期](fitting-pipeline.png)

下表为约 33 万面的合成球网格、24 球预算、6,000 个评价样本的**完整 fit_mesh**。
“无面积缓存”指 Trimesh 的输入状态，不是冷启动解释器或清空 CPU/文件缓存。
MB 使用十进制，峰值由 **tracemalloc** 跟踪，不是 RSS、Pillow 原生像素内存或显存。

| 输入状态 / 方法 | 耗时：之前 → 之后 | 分配峰值：之前 → 之后 |
| --- | ---: | ---: |
| 无面积缓存 / cover | 0.673 → 0.617 s | 159.0 → 98.5 MB |
| 已有面积缓存 / cover | 0.646 → 0.599 s | 124.9 → 98.5 MB |
| 无面积缓存 / surface | 0.129 → 0.060 s | 158.9 → 80.0 MB |
| 已有面积缓存 / surface | 0.118 → 0.063 s | 124.8 → 80.0 MB |

同一网格的单次准备峰值为 **114.0 → 80.0 MB**；已有面积缓存时，准备阶段峰值基本不变。
小网格的完整拟合耗时也基本持平，不能把大输入的收益推广到所有配置。

进程案例使用 4 个网格任务、每任务 24 球、6,000 个评价样本、2 个 worker。
每次在新父进程中测量，包含派发、spawn、worker 导入、拟合、报告和关闭池；父进程导入与输入准备在计时外。

| 输入 | 完整进程调用：之前 → 之后 | 提交任务的序列化估计：之前 → 之后 |
| --- | ---: | ---: |
| 无纹理 | 0.260 → 0.259 s | 0.052 → 0.049 MB |
| 已预计算几何缓存 | 0.282 → 0.277 s | 6.562 → 0.741 MB |
| 自定义顶点属性 | 0.295 → 0.292 s | 5.990 → 0.741 MB |
| 512 × 512 RGBA / PBR | 0.268 → 0.265 s | 4.257 → 0.049 MB |
| 4096 × 4096 RGBA / PBR | 0.482 → 0.256 s | 268.498 → 0.049 MB |

序列化估计来自实际提交的任务对象，不包含函数、消息封装或传输层开销，也不是进程内存峰值。
大纹理案例的完整调用耗时减少约 **47%**；小任务仍主要受进程启动成本影响。

真实机械臂同样采用 24 球/link、6,000 个评价样本、2 个进程，几何加载在计时外：

| 输入 | 完整 fit_asset：之前 → 之后 |
| --- | ---: |
| UR5e collision | 0.427 → 0.430 s |
| UR5e visual | 0.427 → 0.420 s |
| Panda 手臂 collision | 0.365 → 0.361 s |
| Panda 手臂 visual | 0.461 → 0.446 s |

机械臂耗时大致持平，没有稳定的大幅提速；球参数、报告和 link 顺序逐项相同。
输入来源和许可见[第三方资产说明](third-party-assets.md)，仓库不附带这些 URDF 或 mesh。

## 验证与复现

主环境 **460 passed**；最低依赖 **387 passed / 42 skipped**；
Python 3.12、NumPy 2、Trimesh 5 环境 **402 passed / 36 skipped**。
新增测试覆盖忽略不可序列化的视觉/属性、子类复制行为、metadata 错误、非法几何及边计数反例。
120 组准备对照含 40 组一致的拒绝结果，115 次完整拟合对照相同。
UR5e/Panda 的 CLI、五种导出、线程/进程结果、GLB 变换及查看器 HTTP 通过复核，已有图示参数保持一致。
仅安装核心依赖的独立 wheel 也通过三种方法的 spawn 拟合、导出和报告回读。

在仓库根目录运行：

```bash
git worktree add --detach /tmp/assetspheres-pipeline-reference 83c4df6
ASSETSPHERES_REFERENCE=/tmp/assetspheres-pipeline-reference/src/assetspheres
python -m pip install -e '.[plots]'

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/benchmark_preparation.py --reference "$ASSETSPHERES_REFERENCE/fit.py" \
  --full-fit --method cover --repeats 11 --output outputs/preparation-cover.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/benchmark_preparation.py --reference "$ASSETSPHERES_REFERENCE/fit.py" \
  --full-fit --method surface --repeats 11 --output outputs/preparation-surface.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/benchmark_process_transport.py --reference "$ASSETSPHERES_REFERENCE" \
  --repeats 11 --output outputs/process-transport.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/benchmark_process_transport.py --reference "$ASSETSPHERES_REFERENCE" \
  --case robot --input /path/to/ur5e/robot.urdf --geometry collision \
  --repeats 11 --output outputs/ur5e-process.json

python examples/render_pipeline_performance.py
```

图表读取发布的 JSON，不重新测量。本机 cover/surface 分别固定到逻辑核 4/5，进程案例限制到核 8–11。
其他机器应选择可用核并以实际数据为准；此前的[拟合内存](fitting-performance.md)和[启动/预览](startup-and-preview.md)记录单独保留。
