# 图示中的第三方资产来源

## Universal Robots UR5e

- 上游模型：[UniversalRobots/Universal_Robots_ROS2_Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description)。
- 固定版本：[3c173361e155231e5376d2682ddf920d3d79c112](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/tree/3c173361e155231e5376d2682ddf920d3d79c112)。
- 图示输入：工作区 HumanoidAssets/UniversalRobotsUR5e/robot.urdf，由上游 Xacro 展开并改为可移植资源路径；
  collision STL 保持上游几何，拟合时采用 7 个 collision link。
- 展示姿态：各关节均为 0；颜色区分 link，不代表机械臂的原始外观材质。
- 本仓库仅提交实际拟合图、查看器截图及生成参数和球模型记录，不附带第三方 URDF 或 mesh 文件。
  这些图示展示碰撞几何近似，不表示 Universal Robots 对本项目的认可。

对应图片：ur5e-comparison.png、ur5e-viewer.png、ur5e-cover-opt.png；
数据：ur5e-comparison.json、ur5e-cover-opt.json、ur5e-cover-opt-benchmark.json。
输入 URDF 的 SHA-256 和加载后的各 link 网格哈希保存在数据文件中。

### 上游许可文本

以下按资产包所附 LICENSE 原样保留：

```text
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

   * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from
     this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

```
