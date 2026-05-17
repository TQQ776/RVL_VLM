# RVL_VLM-VR / FR3 Omni 视觉语言抓取项目

## 项目演示 Demo

下面三个视频是当前项目的核心演示，建议先看 Demo，再阅读环境搭建和启动说明：

| 演示内容 | 视频 |
| --- | --- |
| 基础移动和视觉识别 | [播放视频](<videos/基础移动和视觉识别.mp4>) |
| 指定物体放置位置 | [播放视频](<videos/指定物体放置位置.mp4>) |
| 指定类别放进篮子里 | [播放视频](<videos/指定类别放进篮子里.mp4>) |

这些 Demo 覆盖了当前系统的主要能力：FR3 基础运动控制、RealSense 视觉识别、
Qwen-Omni 对话理解、API 视觉框选目标、EconomicGrasp 抓取位姿生成，以及抓取后
放置到用户指定位置或容器中。

## 项目简介

这个仓库是一个 ROS 2 Humble 工作空间，用于让 Franka FR3 机械臂通过
Qwen-Omni 对话框/语音理解用户指令，再结合 RealSense 手部相机、API 视觉框、
EconomicGrasp 生成抓取位姿，最后由 MoveIt 和 Franka 控制器执行抓取。

新人可以按这份 README 从环境搭建一直走到启动 `omini` 项目。

> 说明：项目里历史上常写成 `omini`，代码和配置里主要叫 `omni` / `mcp_omni_client`。
> 本文里的 `omini 项目` 指的就是这个 Qwen-Omni + MCP + EconomicGrasp 抓取栈。

当前默认一键启动命令：

```bash
source ~/TQQ_ws/setup_franka.sh
ros2 launch mcp llm_api_grasp.launch.py
```

启动后会打开：

```text
fr3对话框          # 文本/语音对话和快捷按钮
千问视觉检测框      # 手部 D435i 视觉检测画面
全局相机视野        # 辅助 D435 全局观察画面
RViz / MoveIt      # 机械臂模型、规划场景和轨迹显示
```

## 1. 整体结构

仓库建议放在：

```bash
~/TQQ_ws
```

目录作用：

```text
~/TQQ_ws
├── franka/                       # Franka 官方 ROS2/libfranka 相关源码和编译空间
├── tqq/                          # 本项目二次开发 ROS2 包
│   └── src/
│       ├── mcp/                  # Omni 客户端、MCP 工具服务端、一键启动 launch
│       ├── economic_grasp_roi/   # API 框 -> ROI 点云 -> EconomicGrasp -> 机械臂抓取
│       ├── control/              # 旧的目标点控制、回 home 等工具
│       ├── franka_camera/        # RealSense / ArUco / 手眼标定辅助
│       ├── franka_vr_teleop/     # Quest3 原始 VR 遥操，非 Servo 版本
│       ├── easy_handeye2/        # 手眼标定依赖
│       └── audio_dialog/         # 对话框/音频录制基础包
├── third_party/
│   └── EconomicGrasp/            # EconomicGrasp 第三方抓取网络
├── setup_franka.sh               # 每次开终端后 source 的环境脚本
└── readme.md                     # 本文档
```

一键启动链路：

```text
RealSense D435i
  -> Qwen API 视觉检测目标框
  -> 根据目标框切 ROI 点云
  -> EconomicGrasp 生成抓取姿态
  -> MoveIt 规划到预抓取点
  -> Cartesian path 垂直下降
  -> Franka gripper 抓取
```

对话控制链路：

```text
文本/录音输入
  -> Qwen-Omni 理解语义
  -> LLM 选择 MCP 工具
  -> mcp_server 调用机械臂/视觉/抓取服务
  -> 对话框显示和语音播报结果
```

## 2. 硬件和软件前提

硬件：

```text
Franka FR3 机械臂
Franka Hand 夹爪
Intel RealSense D435i 手部相机
Ubuntu 主机，建议 NVIDIA GPU
稳定的局域网，主机能访问 FR3 控制柜
```

当前项目默认参数：

```text
robot_ip: 192.168.22.212
hand camera serial: 327122079035
ROS 2: Humble
Ubuntu: 22.04
Python: 系统 /usr/bin/python3，也就是 Python 3.10
```

非常重要：

```text
不要在 conda base 环境里直接运行 ROS 节点。
本项目默认依赖装在系统 Python /usr/bin/python3 的用户目录里。
每次运行前都 source ~/TQQ_ws/setup_franka.sh，它会主动清理 conda 环境变量。
```

## 3. 克隆仓库和拉取 LFS 权重

如果是新机器，从 GitHub 克隆：

```bash
cd ~
sudo apt update
sudo apt install -y git git-lfs
git lfs install

git clone git@github.com:TQQ776/RVL_VLM-VR.git TQQ_ws
cd ~/TQQ_ws
git lfs pull
```

检查 EconomicGrasp 权重是否真的拉下来了：

```bash
ls -lh third_party/EconomicGrasp/checkpoints/economicgrasp_realsense.tar
```

正常应该是大约 189MB。如果只有几百字节，说明 Git LFS 没拉成功，需要重新执行：

```bash
git lfs install
git lfs pull
```

## 4. 安装 ROS 2 Humble 和基础工具

如果系统还没装 ROS 2 Humble，先按 ROS 2 官方方式安装 Humble Desktop。
安装好后，确认：

```bash
source /opt/ros/humble/setup.bash
ros2 --version
```

安装本项目常用系统工具：

```bash
sudo apt update
sudo apt install -y \
  python3-pip \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool \
  python3-opencv \
  python3-numpy \
  python3-pil \
  python3-pyaudio \
  portaudio19-dev \
  librealsense2-utils \
  ros-humble-moveit \
  ros-humble-realsense2-camera \
  ros-humble-aruco-ros \
  ros-humble-cv-bridge \
  ros-humble-tf-transformations
```

初始化 `rosdep`：

```bash
sudo rosdep init 2>/dev/null || true
rosdep update
```

## 5. 编译 Franka 官方依赖

本仓库已经包含 `franka/src` 下的 Franka 官方 ROS2/libfranka 相关源码。

先装依赖：

```bash
cd ~/TQQ_ws/franka
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
```

编译：

```bash
cd ~/TQQ_ws/franka
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

编译完成后检查：

```bash
source ~/TQQ_ws/franka/install/setup.bash
ros2 pkg prefix franka_fr3_moveit_config
ros2 pkg prefix franka_bringup
ros2 pkg prefix franka_gripper
```

都能输出路径才说明 Franka underlay 编译成功。

## 6. 编译本项目 TQQ overlay

安装本项目 ROS 依赖：

```bash
cd ~/TQQ_ws/tqq
source /opt/ros/humble/setup.bash
source ~/TQQ_ws/franka/install/setup.bash
rosdep install --from-paths src --ignore-src -r -y
```

编译：

```bash
cd ~/TQQ_ws/tqq
source /opt/ros/humble/setup.bash
source ~/TQQ_ws/franka/install/setup.bash
colcon build --symlink-install
```

检查关键包：

```bash
source ~/TQQ_ws/tqq/install/setup.bash
ros2 pkg prefix mcp
ros2 pkg prefix economic_grasp_roi
ros2 pkg prefix franka_camera
ros2 pkg prefix easy_handeye2
```

## 7. 每次开新终端都要 source 环境

以后每开一个新终端，都执行：

```bash
source ~/TQQ_ws/setup_franka.sh
```

这个脚本做了三件事：

```bash
export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda3 | grep -v conda | tr '\n' ':' | sed 's/:$//')
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE
source /opt/ros/humble/setup.bash
source ~/TQQ_ws/franka/install/setup.bash
source ~/TQQ_ws/tqq/install/setup.bash
```

这样可以避免 conda 的 Python 抢走 ROS 2 Humble 的系统 Python。

## 8. 安装 Qwen-Omni / 对话框 Python 依赖

Omni 客户端需要 DashScope SDK、websocket 和音频录制依赖：

```bash
source ~/TQQ_ws/setup_franka.sh
/usr/bin/python3 -m pip install --user -U "dashscope>=1.23.9" websocket-client pyaudio
```

如果 `pyaudio` 报错，先装系统依赖后重试：

```bash
sudo apt install -y portaudio19-dev python3-pyaudio
/usr/bin/python3 -m pip install --user -U pyaudio
```

检查：

```bash
/usr/bin/python3 - <<'PY'
import dashscope
import websocket
import pyaudio
print('Omni Python deps OK')
PY
```

## 9. 安装 EconomicGrasp / CUDA 抓取网络依赖

EconomicGrasp 不是普通 ROS 包，它还需要 PyTorch、MinkowskiEngine、
pointnet2、knn 等 CUDA/Python 依赖。

关键原则：

```text
这些依赖必须装进 /usr/bin/python3 能看到的环境。
不要装进 conda base，否则 ROS 节点 import 不到。
```

先确认系统 Python：

```bash
source ~/TQQ_ws/setup_franka.sh
/usr/bin/python3 - <<'PY'
import sys
print(sys.executable)
print(sys.version)
PY
```

应该看到 `/usr/bin/python3` 和 Python 3.10。

### 9.1 安装 PyTorch

根据你的 CUDA 版本，从 PyTorch 官网选择 Linux + Pip + Python + CUDA 对应命令。
当前工作站使用的是 CUDA 12.x，对应的 PyTorch 需要能在 `/usr/bin/python3`
里 `import torch`，并且 `torch.cuda.is_available()` 为 `True`。

安装后检查：

```bash
/usr/bin/python3 - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda version:', torch.version.cuda)
PY
```

如果 `cuda available` 是 `False`，EconomicGrasp 会非常慢或无法使用 GPU。

### 9.2 编译 EconomicGrasp 扩展

本仓库提供了安装脚本：

```bash
source ~/TQQ_ws/setup_franka.sh
cd ~/TQQ_ws
/home/tqq/TQQ_ws/tqq/src/economic_grasp_roi/scripts/install_economic_grasp_deps.sh
```

如果你的显卡不是当前机器的架构，可以临时改 CUDA 架构，例如：

```bash
TORCH_CUDA_ARCH_LIST=8.6 \
/home/tqq/TQQ_ws/tqq/src/economic_grasp_roi/scripts/install_economic_grasp_deps.sh
```

安装完成后检查：

```bash
/usr/bin/python3 - <<'PY'
import torch
import MinkowskiEngine
import open3d
import pointnet2
import knn_pytorch
print('EconomicGrasp deps OK')
PY
```

## 10. 设置 DashScope API Key

Omni 客户端和服务端都通过环境变量读取 API Key：

```bash
export DASHSCOPE_API_KEY=你的_DashScope_API_Key
```

建议写进 `~/.bashrc`：

```bash
echo 'export DASHSCOPE_API_KEY=你的_DashScope_API_Key' >> ~/.bashrc
```

不要把真实 API Key 写进仓库里的 YAML 或 README。

当前模型配置在：

```text
~/TQQ_ws/tqq/src/mcp/config/mcp_omni_client.yaml
~/TQQ_ws/tqq/src/mcp/config/mcp_server.yaml
```

默认模型：

```yaml
omni_model: qwen3.5-omni-plus-realtime
omni_text_model: qwen3.5-omni-plus
```

## 11. 确认 RealSense 手部相机

如果你有多个 RealSense，相机必须指定 D435i 手部相机序列号。

查看相机：

```bash
rs-enumerate-devices | grep -E "Name|Serial Number" -A 1
```

当前项目默认手部 D435i：

```text
327122079035
```

一键启动里对应参数：

```bash
camera_serial_no:=327122079035
```

注意：如果你把序列号写进 YAML，必须写成字符串：

```yaml
serial_no: "327122079035"
```

不要写成：

```yaml
serial_no: 327122079035
```

否则 RealSense 会报：

```text
parameter 'serial_no' has invalid type
```

## 12. 第一次使用必须做手眼标定

Omni 抓取需要知道相机坐标系和机械臂基座坐标系之间的关系。
一键启动会自动发布名为 `fr3_d435i_handeye` 的标定结果，但前提是你已经标定过。

如果还没标定，按下面做。

### 12.1 启动相机

终端 1：

```bash
source ~/TQQ_ws/setup_franka.sh
ros2 launch franka_camera realsense.launch.py \
  camera_namespace:=camera \
  camera_name:=camera \
  serial_no:=327122079035 \
  params_file:=/home/tqq/TQQ_ws/tqq/src/control/config/realsense_aligned_depth.yaml
```

检查图像：

```bash
ros2 topic list | grep camera
```

至少应该看到：

```text
/camera/camera/color/image_raw
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/color/camera_info
```

### 12.2 启动 ArUco 检测

终端 2：

```bash
source ~/TQQ_ws/setup_franka.sh
ros2 run aruco_ros single \
  --ros-args \
  -p marker_id:=582 \
  -p marker_size:=0.15 \
  -p reference_frame:=camera_color_optical_frame \
  -p camera_frame:=camera_color_optical_frame \
  -p marker_frame:=aruco_marker_frame \
  -p image_is_rectified:=true \
  -r /image:=/camera/camera/color/image_raw \
  -r /camera_info:=/camera/camera/color/camera_info
```

`marker_id` 和 `marker_size` 要和你实际打印的 ArUco 标定板一致。

### 12.3 启动机械臂 MoveIt

终端 3：

```bash
source ~/TQQ_ws/setup_franka.sh
ros2 launch franka_fr3_moveit_config moveit.launch.py \
  robot_ip:=192.168.22.212 \
  use_fake_hardware:=false
```

### 12.4 打开 easy_handeye2 标定界面

终端 4：

```bash
source ~/TQQ_ws/setup_franka.sh
ros2 launch easy_handeye2 calibrate.launch.py \
  calibration_type:=eye_in_hand \
  name:=fr3_d435i_handeye \
  robot_base_frame:=fr3_link0 \
  robot_effector_frame:=fr3_hand_tcp \
  tracking_base_frame:=camera_color_optical_frame \
  tracking_marker_frame:=aruco_marker_frame
```

在界面中按流程采样、计算并保存标定结果。

### 12.5 发布标定结果

标定完成后，以后可以用：

```bash
source ~/TQQ_ws/setup_franka.sh
ros2 launch easy_handeye2 publish.launch.py name:=fr3_d435i_handeye
```

一键启动文件会自动执行这一步。

## 13. 启动真实 FR3 前检查

在 Franka Desk 里确认：

```text
FCI 已启用
机器人处于可控制状态
急停已释放
夹爪连接正常
主机能 ping 通 robot_ip
```

检查网络：

```bash
ping 192.168.22.212
```

如果启动时报：

```text
libfranka: Connection to FCI refused
```

说明不是代码问题，而是 Franka Desk 没启用 FCI，或者机器人当前不允许外部控制。

## 14. 一键启动 Omni 抓取项目

这是最终启动命令。它会按顺序启动：

```text
RealSense
MoveIt / FR3
easy_handeye2 TF 发布
economic_grasp_roi 抓取控制
mcp_server 工具服务端
mcp_omni_client 对话框客户端
```

终端执行：

```bash
source ~/TQQ_ws/setup_franka.sh
export DASHSCOPE_API_KEY=你的_DashScope_API_Key

ros2 launch mcp llm_yolo_grasp.launch.py \
  robot_ip:=192.168.22.212 \
  camera_serial_no:=327122079035 \
  use_fake_hardware:=false
```

启动成功后，会自动弹出文本对话框，不需要按 `t`。

对话框用法：

```text
Enter：发送文本
Shift + Enter：换行
开始录音：开始录音
停止录音并发送：把语音发给 Qwen-Omni 理解
回到初始位置：调用 home 工具
打开夹爪：调用 open gripper 工具
关闭夹爪：调用 close gripper 工具
```

可以输入或说：

```text
你能看到什么
现在能抓什么
抓橘子
抓苹果
把橘子放到苹果右边 5 厘米
先抓橘子，再回到初始位置，然后打开夹爪
向上移动 3 厘米
向左移动 2 厘米
打开夹爪
关闭夹爪
回到初始位置
```

方向约定：

```text
右 = FR3 基座 x 正方向
左 = FR3 基座 x 负方向
前 = FR3 基座 y 正方向
后 = FR3 基座 y 负方向
上 = FR3 基座 z 正方向
下 = FR3 基座 z 负方向
```

## 15. 当前抓取策略说明

当前 `economic_grasp_roi` 的抓取不是直接让机械臂斜着撞向物体。

现在流程是：

```text
1. Qwen 视觉 API 返回目标框
2. 根据目标框切出物体 ROI 点云
3. EconomicGrasp 在 ROI 点云上生成抓取姿态
4. 抓取姿态只保留绕 z 轴的 yaw，保持夹爪垂直于地面
5. 优先选择靠近物体中心的抓取点，避免夹在边缘
6. 先生成高于目标 8cm 的预抓取位姿
7. MoveIt 规划到预抓取位姿
8. Cartesian path 沿 z 轴平滑下降 8cm
9. 关闭夹爪
```

抓取-放置时，`mcp_server` 会在拍照检测时建立一份场景记忆：把视觉框、深度图和手眼 TF 合成参考物体的 `base_xyz`。
这样抓起一个物体、相机视角变化以后，仍然可以把它放到之前看到的苹果/橘子左边、右边、前后或上下指定距离。
如果目标是开口箱子，服务端会按 30cm x 22cm x 14cm、碰撞壁厚 1.5cm 建立空心箱子模型，只加入四壁和底板，顶部和内部保持可进入。
拍照检测完成后，服务端会先把任务目标和画面中其他清晰可见的独立障碍物转换成 MoveIt planning scene 里的碰撞模型并发布，所以 RViz 会先看到模型，再看到机械臂动作。
水果在世界里先显示为橙色保守球/盒模型，箱子显示为绿色模型，其他可见障碍物显示为蓝色通用盒模型；抓取目标水果时会先从 world collision 中移除以免挡住抓取规划，抓取成功后再作为 attached object 挂到 `fr3_hand_tcp` 上，让后续放置规划把手里夹着的水果也算进避障。
打开夹爪释放后，attached fruit 会自动 detach/remove。

放置流程是：

```text
1. 观察场景并记住参考物体位置
2. 先把水果/箱子碰撞模型发布到 MoveIt，并在 RViz 中显示
3. 抓取目标物体，抓住后把水果模型 attached 到夹爪
4. 保持当前夹爪姿态，先沿 base z 轴抬高
5. MoveIt 规划到参考物体目标位置正上方，并考虑箱子/手持水果避障
6. Cartesian path 沿 base z 轴下降
7. 打开夹爪并 detach 手持水果模型
8. 再沿 base z 轴抬起离开
```

“把苹果放到箱子里”的流程会略有不同：抓起苹果后，会先沿 base z 轴小幅慢速抬升，避开刚抓取位置附近的物体，再把箱子碰撞模型加入 MoveIt planning scene，低速规划到箱口上方，最后垂直下降进箱子。

相关配置：

```text
~/TQQ_ws/tqq/src/economic_grasp_roi/config/roi_economic_grasp.yaml
~/TQQ_ws/tqq/src/mcp/config/mcp_server.yaml
```

关键参数：

```yaml
staged_grasp_enabled: true
pre_grasp_lift_m: 0.08
cartesian_descend_m: 0.08
cartesian_descend_duration_sec: 6.0
economic_grasp_orientation_mode: yaw_only
target_offset_xyz_base: [0.0, 0.0, -0.04]
grasp_center_filter_enabled: true
grasp_center_priority_enabled: true
```

点云预览窗口仍会显示抓取位姿，但默认不需要按 Enter 确认：

```yaml
popup_preview_before_execute: true
popup_preview_require_confirmation: false
```

## 16. 单轴移动和回 home

单轴移动由 `mcp_server` 调用 MoveIt Cartesian path 实现，不是简单给一个点到点随机规划。
这样可以让“向上/向下/向左/向右/向前/向后”尽量只沿指定轴移动。

配置：

```text
~/TQQ_ws/tqq/src/mcp/config/mcp_server.yaml
```

关键参数：

```yaml
axis_move_execution_mode: cartesian
axis_cartesian_max_step_m: 0.002
axis_cartesian_min_fraction: 0.99
axis_cartesian_duration_per_cm_sec: 0.6
max_single_axis_move_cm: 10.0
```

回 home 使用关节空间插值生成中间点，并带时间戳执行，不再是最后爆停：

```yaml
home_move_duration_sec: 4.0
home_trajectory_dt_sec: 0.05
```

手动测试：

```bash
source ~/TQQ_ws/setup_franka.sh
ros2 service call /mcp_server/go_home std_srvs/srv/Trigger {}
ros2 service call /mcp_server/open_gripper std_srvs/srv/Trigger {}
ros2 service call /mcp_server/close_gripper std_srvs/srv/Trigger {}
ros2 service call /mcp_server/call_tool mcp/srv/CallTool "{name: move_z_cm, arguments_json: '{\"centimeters\": 2.0}'}"
```

## 17. 如果不想一键启动，可以分开启动

### 17.1 启动相机

```bash
source ~/TQQ_ws/setup_franka.sh
ros2 launch franka_camera realsense.launch.py \
  camera_namespace:=camera \
  camera_name:=camera \
  serial_no:=327122079035 \
  params_file:=/home/tqq/TQQ_ws/tqq/src/control/config/realsense_aligned_depth.yaml
```

### 17.2 启动 MoveIt / FR3

```bash
source ~/TQQ_ws/setup_franka.sh
ros2 launch franka_fr3_moveit_config moveit.launch.py \
  robot_ip:=192.168.22.212 \
  use_fake_hardware:=false
```

### 17.3 发布手眼标定

```bash
source ~/TQQ_ws/setup_franka.sh
ros2 launch easy_handeye2 publish.launch.py name:=fr3_d435i_handeye
```

### 17.4 启动 EconomicGrasp 控制节点

```bash
source ~/TQQ_ws/setup_franka.sh
ros2 launch economic_grasp_roi roi_economic_grasp.launch.py \
  params_file:=/home/tqq/TQQ_ws/tqq/src/economic_grasp_roi/config/roi_economic_grasp.yaml
```

### 17.5 启动 MCP 服务端

```bash
source ~/TQQ_ws/setup_franka.sh
ros2 launch mcp mcp_server.launch.py \
  params_file:=/home/tqq/TQQ_ws/tqq/src/mcp/config/mcp_server.yaml
```

### 17.6 启动 Omni 客户端

```bash
source ~/TQQ_ws/setup_franka.sh
export DASHSCOPE_API_KEY=你的_DashScope_API_Key

ros2 launch mcp mcp_omni_client.launch.py \
  params_file:=/home/tqq/TQQ_ws/tqq/src/mcp/config/mcp_omni_client.yaml
```

## 18. 常见问题

### 18.1 `Package 'mcp' not found`

说明没有 source overlay：

```bash
source ~/TQQ_ws/setup_franka.sh
```

如果还不行，重新编译：

```bash
cd ~/TQQ_ws/tqq
source /opt/ros/humble/setup.bash
source ~/TQQ_ws/franka/install/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### 18.2 `Connection to FCI refused`

说明 Franka Desk 没启用 FCI，或者机器人没进入可外部控制状态。
先在 Desk 里启用 FCI，再重新启动 launch。

### 18.3 RealSense 用错相机

有两个 RealSense 时，必须指定手部 D435i：

```bash
ros2 launch mcp llm_yolo_grasp.launch.py camera_serial_no:=327122079035
```

如果改 YAML，序列号要加引号：

```yaml
serial_no: "327122079035"
```

### 18.4 EconomicGrasp import 失败

通常是因为你在 conda 环境里装了依赖，但 ROS 节点用的是 `/usr/bin/python3`。

先退出 conda 并重新 source：

```bash
conda deactivate 2>/dev/null || true
source ~/TQQ_ws/setup_franka.sh
```

检查：

```bash
/usr/bin/python3 - <<'PY'
import torch
import MinkowskiEngine
import pointnet2
import knn_pytorch
print('OK')
PY
```

### 18.5 缺少 `economicgrasp_realsense.tar`

重新拉 Git LFS：

```bash
cd ~/TQQ_ws
git lfs install
git lfs pull
```

### 18.6 没有弹出对话框

检查配置：

```yaml
text_popup_enabled: true
text_popup_auto_open: true
```

文件：

```text
~/TQQ_ws/tqq/src/mcp/config/mcp_omni_client.yaml
```

### 18.7 LLM 不调用工具

当前架构是：

```text
mcp_server 提供工具列表和工具说明
mcp_omni_client 只负责把工具说明交给 LLM，并执行 LLM 选择的工具
```

检查服务是否存在：

```bash
ros2 service list | grep mcp_server
```

应该能看到：

```text
/mcp_server/list_tools
/mcp_server/call_tool
```

### 18.8 机械臂抓取前撞到物体

确认分阶段抓取是开启的：

```yaml
staged_grasp_enabled: true
pre_grasp_lift_m: 0.08
cartesian_descend_m: 0.08
```

如果物体更高或场景更危险，可以先增大 `pre_grasp_lift_m`，再降低速度。

## 19. VR 遥操说明

Quest3 原始 VR 遥操仍保留在：

```text
~/TQQ_ws/tqq/src/franka_vr_teleop
```

详细安装和 Unity APK 设置见：

```text
~/TQQ_ws/README_VR_TELEOP_ORIGINAL.md
```

启动：

```bash
source ~/TQQ_ws/setup_franka.sh
ros2 launch franka_vr_teleop quest3_franka_teleop.launch.py \
  robot_ip:=192.168.22.212 \
  use_fake_hardware:=false
```

注意：MoveIt Servo 版 VR 遥操代码已经删除，当前保留的是原始 Cartesian velocity 直控版本。

## 20. 最小启动清单

如果你已经完成所有安装、编译、标定，日常只需要：

```bash
cd ~/TQQ_ws
source setup_franka.sh
export DASHSCOPE_API_KEY=你的_DashScope_API_Key

ros2 launch mcp llm_yolo_grasp.launch.py \
  robot_ip:=192.168.22.212 \
  camera_serial_no:=327122079035 \
  use_fake_hardware:=false
```

看到对话框后，输入：

```text
现在能抓什么
```

如果能弹出视觉框图、点云抓取预览，并且机械臂能移动到预抓取点再垂直下降，
说明整个 Omni 视觉语言抓取链路已经跑通。
