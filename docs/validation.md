# 初始版本验证记录

日期：2026-11-03。验证针对本地实现，不代表 Isaac Sim 原生后端或下游规划器已完成集成测试。

以下首先保留 v0.1 初始化阶段的记录；该阶段的 socket 权限限制属于历史环境。v0.2 的 CPU / CUDA 与新算法验证补充在文末。

## 环境

Python 3.10.0，NumPy 1.26.4，SciPy 1.15.3，trimesh 4.12.2，PyYAML 6.0.3，defusedxml 0.7.1，Viser 1.1.0，pytest 9.1.1。使用仓库内独立虚拟环境，复用宿主已安装依赖。宿主 ROS 的全局 pytest 插件存在版本冲突，因此本地测试通过 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 关闭第三方插件自动加载。

## 自动验证

- 31 项测试通过，覆盖三种 CPU 算法、球数预算、确定性、全局随机状态隔离、开口 mesh 显式回退和体素网格预算预检。
- 在开口的细长三角形上，`cover` 对独立生成的 20,000 个表面点均无正间隙。
- 验证 URDF 非均匀 scale → geometry origin → joint origin 的变换顺序、多个 mesh 场景实例、link 选择以及 visual 自动回退。
- 验证 JSON、cuRobo/Lula 球片段、URDF、GLB 的导出与重载；保留 visual/joint；输出 mesh 路径按新目录重写；源文件移动改名后重新导出可再次读取。
- 验证 Viser 场景的 link 位姿、球编辑、旧覆盖率失效标记、表面包络证书移除和 JSON 保存。使用离线 GUI/scene 替身，并绑定已安装 Viser API 的实际方法签名。
- 验证 Lula 适配器的公开调用契约、顶点和索引类型。使用接口替身，未运行实际 Lula。
- Ruff lint 与格式检查通过；wheel 和源码分发包构建通过。

## 资产验证

| 输入 | 参数 | 结果 |
| --- | --- | --- |
| 自带双连杆 URDF | `voxel`，每 link 24 球，padding 3 mm | 2 个 link 完成，5 类输出可生成；base 表面采样覆盖约 39.4%，arm 约 87.3%，报告保留实际间隙 |
| 自带双连杆 URDF | `cover`，每 link 12 球 | 2 个 link 的三角表面均被包络，5 类输出可生成 |
| 自生成带环形把手 STL | `cover`，24 球 | 独立 mesh 拟合、URDF 导出与 GLB 预览完成 |
| 工作区已有 UR10 资产 | `cover`，每 link 16 球 | 7 个 link、112 个球，全部表面样本被覆盖，导出完成 |
| 工作区已有 Robotiq 2F85 资产 | `voxel`，显式 `--non-watertight cover`，每 link 20 球 | 9 个 link、180 个球，开放 mesh 均显式回退为 cover，导出完成 |

第三方机器人资产及其生成结果没有加入版本控制。运行结果保存在本地 `outputs/` 下，并由 `.gitignore` 排除。

## 尚未验证

- 当前执行环境禁止创建监听 socket，因此未启动 Viser HTTP/WebSocket 服务，也未进行真实浏览器交互验证。离线 GUI 测试不能替代此项检查。
- 当前解释器没有 Isaac Sim 的原生 `lula`，因此未进行原生球化运行测试。
- 两份规划器 YAML 是结构验证过的球片段；未在 cuRobo / RMPflow 中加载并执行规划。
- 知乎参考链接无法读取，未据此实现或验证其方法。

## v0.2：CPU / CUDA 与新生成算法

- 112 项本地测试通过，包含真实 CUDA float64 距离/贪心、内存工作预算、完整 voxel 的 CPU/CUDA 同球结果、NumPy 解析梯度与 Torch 更新一致性、真实 spawn 多进程和线程并行。
- `cover-tight` 的支持集优化始终回到完整点集检查，优化失败时保留已有包围球；共线、共面、微小尺度、大坐标和重复点均有覆盖。
- `slab` 在圆柱、box、torus 的独立高密度表面样本、box 内部 20,000 点、跨多个 slab 的单三角形、旋转/平移/缩放等输入上验证；包络构造仍有文档列明的实体解释和浮点假设。
- 手动编辑 `slab` 球后，条件性实体包络标记与三角表面包络标记都被移除，旧质量数据标为失效。
- `compare_sphere_models` 在原 mesh 和全部球组成的共同评价域上统计 TP/FN/FP；多闭合壳默认不猜测实体语义，显式 union-components 与 parity 有反例测试。
- 原始 mesh 监督的 refined 路线使用独立分类样本，不默认把体素标签视为精确 SDF。它只支持单闭合表面体，仍假设没有自交。
- UR10 用 `slab`、每 link 24 球、4 线程完成 7 个 link、168 球和全部 5 类文件导出；全部独立表面样本被包含。其部分输入 mesh 开口，不把这些 link 宣称为已验证的闭合实体。
- 真实命令行分别跑通 spawn 进程 `slab` 和 CUDA `refined`。原生 Lula、真实浏览器交互和下游规划器集成仍未验证。

机器、计时范围、完整数据、质量取舍和复现命令见 [performance.md](performance.md) 与[实验快照](benchmarks-2026-11-03.json)。初始化阶段的 31 项测试已经扩展为上述 112 项；历史数字不代表当前覆盖范围。
