# 初始版本验证记录

日期：2026-11-03。验证针对本地实现，不代表 Isaac Sim 原生后端或下游规划器已完成集成测试。

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
