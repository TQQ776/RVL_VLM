# Quest3 + Franka FR3 原始 VR 遥操完整说明

这份文档只讲“原始 VR 遥操版本”，也就是不使用 MoveIt Servo 的版本。

对应启动文件是：

```bash
ros2 launch franka_vr_teleop quest3_franka_teleop.launch.py
```

它和 Servo 版本的区别是：原始版本直接使用 Franka 的 `cartesian_velocity` command interface，让机械臂末端按照速度命令运动；它不走 MoveIt 逆解，也不走 `/fr3_arm_controller/joint_trajectory`。


## 1. 这套系统在做什么

目标：用 Quest3 右手柄控制 Franka FR3 末端移动和旋转。

你在 VR 里移动右手柄，电脑收到手柄的位姿。ROS2 节点把手柄位姿变化转换成机械臂末端速度，再把速度写入 Franka 官方的笛卡尔速度接口。

整体链路是：

```text
Quest3 右手柄
  -> Unity App 读取右手柄 position / rotation / button
  -> UDP JSON 发送到电脑 5055 端口
  -> quest3_udp_bridge.py
  -> /quest3/right_controller/pose
  -> quest3_pose_to_twist.py
  -> /vr_cartesian_velocity_controller/twist_cmd
  -> VRCartesianVelocityController
  -> Franka cartesian_velocity command interface
  -> FR3 末端速度运动
```

夹爪链路是：

```text
Quest3 trigger
  -> /quest3/right_controller/trigger_pressed
  -> quest3_gripper_bridge.py
  -> /franka_gripper/move
  -> Franka gripper
```

启用遥操的按钮是右手柄 grip。只有按住 grip 时，机械臂主体才会动；松开 grip 会发送零速度。


## 2. 核心概念

### 2.1 位姿不是直接给机械臂的目标点

这套原始遥操不是这样：

```text
手柄位置 -> 机械臂目标位姿 -> IK -> 轨迹规划
```

它实际是这样：

```text
手柄两帧之间的位置变化 / 时间
  -> 手柄速度
  -> 坐标轴映射
  -> 末端速度 vx vy vz wx wy wz
  -> Franka 笛卡尔速度控制
```

所以它是“速度控制”，不是“点到点规划”。

### 2.2 为什么机械臂不会一直追着某个绝对位置

当前配置里：

```yaml
motion_mode: hand_velocity
```

含义是：手柄正在动，机械臂才动；手柄停住，机械臂也停住。

以前也有一种模式：

```yaml
motion_mode: anchor_offset
```

含义是：按下 grip 时记录一个手柄锚点，只要手柄偏离这个锚点，机械臂就持续朝偏移方向运动。这个模式容易出现“我只把手柄向左移动一小段，机械臂却一直向左走”的现象。所以现在推荐使用 `hand_velocity`。

### 2.3 TwistStamped 是什么

ROS 里 `geometry_msgs/msg/TwistStamped` 表示带时间戳的速度命令：

```text
linear.x   x 方向线速度，单位 m/s
linear.y   y 方向线速度，单位 m/s
linear.z   z 方向线速度，单位 m/s
angular.x  绕 x 轴角速度，单位 rad/s
angular.y  绕 y 轴角速度，单位 rad/s
angular.z  绕 z 轴角速度，单位 rad/s
```

在本工程中，它发布到：

```bash
/vr_cartesian_velocity_controller/twist_cmd
```


## 3. 代码和配置文件分别负责什么

包位置：

```bash
~/TQQ_ws/tqq/src/franka_vr_teleop
```

主要文件：

```text
launch/quest3_franka_teleop.launch.py
  原始 VR 遥操启动文件。

config/quest3_teleop.yaml
  UDP、话题名、输入增益、速度上限、死区、平滑参数、夹爪参数。

config/quest3_axis_mapping.yaml
  Quest3 手柄坐标轴到 Franka 基座坐标轴的映射。

config/vr_teleop_controllers.yaml
  ros2_control 控制器配置，里面注册 vr_cartesian_velocity_controller。

scripts/quest3_udp_bridge.py
  UDP 接收器，把 Unity 发来的 JSON 转成 ROS topic。

scripts/quest3_pose_to_twist.py
  把手柄位姿变化转换成 TwistStamped。

scripts/quest3_gripper_bridge.py
  把 trigger 按钮转换成夹爪动作。

src/vr_cartesian_velocity_controller.cpp
  C++ ros2_control 控制器，把 TwistStamped 写入 Franka 笛卡尔速度命令接口。

include/franka_vr_teleop/vr_cartesian_velocity_controller.hpp
  控制器头文件。

franka_vr_teleop.xml
  pluginlib 插件描述文件，让 controller_manager 能找到这个控制器。
```


## 4. 安装和环境准备

### 4.1 基础前提

这套工程默认你已经有：

```text
Ubuntu 22.04
ROS2 Humble
Franka ROS2 / libfranka 环境
~/TQQ_ws/setup_franka.sh
~/TQQ_ws/tqq 工作空间
Quest3 Unity App
```

每次开新终端，都先加载环境：

```bash
cd ~/TQQ_ws
source setup_franka.sh

cd ~/TQQ_ws/tqq
source install/setup.bash
```

如果你忘记 `source install/setup.bash`，常见现象是：

```text
Package 'franka_vr_teleop' not found
```

### 4.2 编译 franka_vr_teleop 包

```bash
cd ~/TQQ_ws
source setup_franka.sh

cd ~/TQQ_ws/tqq
colcon build --packages-select franka_vr_teleop --symlink-install
source install/setup.bash
```

检查包是否能找到：

```bash
ros2 pkg prefix franka_vr_teleop
```

检查可执行脚本是否安装：

```bash
ros2 pkg executables franka_vr_teleop
```

应该能看到类似：

```text
franka_vr_teleop quest3_udp_bridge.py
franka_vr_teleop quest3_pose_to_twist.py
franka_vr_teleop quest3_gripper_bridge.py
```


## 5. Unity 和 Quest3 App 从零安装配置

这一章讲 VR 端怎么从零搭起来。新人最容易卡在这里：Unity 没装 Android 模块、Quest 没开开发者模式、Build Settings 没切 Android、OpenXR 没启用、App 装上后不知道在哪里打开。

你现在的 Unity 工程在：

```bash
/home/tqq/My project
```

主要脚本在：

```bash
/home/tqq/My project/Assets/Quest3UdpSender.cs
```

当前项目配置：

```text
Unity Editor: 2022.3.62f3
Product Name: tqq
Package Name: com.tqq.teleop
Scene: Assets/Scenes/SampleScene.unity
XR: OpenXR
Android architecture: ARM64
```

### 5.1 Unity 在这套系统里负责什么

Unity App 很简单，它不是机器人控制器，也不跑 ROS。

它只做四件事：

```text
1. 读取 Quest3 右手柄位置 position。
2. 读取 Quest3 右手柄旋转 rotation。
3. 读取右手柄按钮 grip / trigger / A / B。
4. 把这些数据用 UDP JSON 发到电脑。
```

Unity App 发出的 JSON 长这样：

```json
{
  "pose": {
    "position": [0.12, -0.03, 1.42],
    "orientation": [0.0, 0.0, 0.0, 1.0]
  },
  "grip_pressed": true,
  "trigger_pressed": false,
  "calibrate_pressed": false,
  "reset_calibration_pressed": false
}
```

电脑上的 ROS2 节点收到这个 JSON 后，才会把它变成 ROS 话题。

### 5.2 安装 Unity Hub

如果电脑还没有 Unity Hub，先安装 Unity Hub。

可以到 Unity 官网下载安装：

```text
https://unity.com/download
```

Ubuntu 上安装完成后，打开 Unity Hub：

```bash
unityhub
```

如果命令找不到，也可以从应用菜单里打开 Unity Hub。

### 5.3 安装 Unity Editor

当前项目使用：

```text
Unity 2022.3.62f3
```

在 Unity Hub 里：

```text
Installs
-> Install Editor
-> Archive 或 Official releases
-> 选择 Unity 2022.3.x LTS
```

最好和项目版本一致，也就是 `2022.3.62f3`。如果找不到完全一样的版本，尽量选择 Unity 2022.3 LTS 的接近版本。

安装 Editor 时必须勾选 Android 模块：

```text
Android Build Support
  -> Android SDK & NDK Tools
  -> OpenJDK
```

这三个都要有。少一个都可能 Build 失败。

如果你已经装了 Unity Editor，但忘了 Android 模块：

```text
Unity Hub
-> Installs
-> 找到 2022.3.x
-> 齿轮图标
-> Add modules
-> 勾选 Android Build Support、Android SDK & NDK Tools、OpenJDK
```

### 5.4 打开 Unity 项目

在 Unity Hub 里：

```text
Projects
-> Add
-> Add project from disk
-> 选择 /home/tqq/My project
```

第一次打开会比较慢，Unity 会导入 package。

打开后确认场景：

```text
Assets
-> Scenes
-> SampleScene
```

双击打开 `SampleScene`。

这个场景里应该至少有：

```text
Main Camera
Directional Light
Quset3UdpSender
```

注意：场景里对象名现在拼成了 `Quset3UdpSender`，这只是名字拼写问题，不影响功能。

### 5.5 Unity 脚本代码在哪里

Unity 端最关键的脚本是：

```bash
/home/tqq/My project/Assets/Quest3UdpSender.cs
```

我也在 ROS 包里放了一份标准副本，方便以后新建 Unity 项目时复制：

```bash
~/TQQ_ws/tqq/src/franka_vr_teleop/unity/Quest3UdpSender.cs
```

这个脚本负责：

```text
1. 读取 Quest3 右手柄位置和旋转。
2. 读取 grip、trigger、A、B 等按钮。
3. 在 VR 里显示电脑 IP 输入面板。
4. 把手柄数据通过 UDP 发给电脑 5055 端口。
```

如果 Unity 项目里没有这个脚本，可以这样复制：

```bash
cp ~/TQQ_ws/tqq/src/franka_vr_teleop/unity/Quest3UdpSender.cs \
  "/home/tqq/My project/Assets/Quest3UdpSender.cs"
```

复制后回到 Unity，等 Unity 自动编译。编译成功后，`Quest3UdpSender` 才能在 `Add Component` 里搜到。

### 5.6 确认脚本挂载

在 Hierarchy 里点：

```text
Quset3UdpSender
```

Inspector 里应该能看到：

```text
Quest3UdpSender (Script)
```

脚本参数大概是：

```text
Pc Ip
Pc Port: 5055
Send Hz: 60
Show Ip Panel On Start: true
```

如果没有脚本：

```text
Add Component
-> 搜索 Quest3UdpSender
-> 添加
```

如果搜不到 `Quest3UdpSender`，说明脚本还没有放进 `Assets`，或者 Unity 编译报错。先确认这个文件存在：

```bash
ls "/home/tqq/My project/Assets/Quest3UdpSender.cs"
```

然后看 Unity Console 有没有红色报错。只要有 C# 编译错误，Unity 就不会让你挂载脚本。

挂载脚本的意思是：把 `Quest3UdpSender.cs` 这个 C# 组件加到场景里的某个 GameObject 上。这样 App 运行时，Unity 才会调用脚本里的 `Start()` 和 `Update()`，手柄数据才会被读取并通过 UDP 发出去。

如果场景里没有 `Quset3UdpSender` 这个对象，可以手动创建：

```text
Hierarchy
-> 右键
-> Create Empty
-> 命名 Quest3UdpSender
-> Inspector
-> Add Component
-> 搜索 Quest3UdpSender
-> 添加
```

### 5.7 切换到 Android 平台

Unity 菜单：

```text
File
-> Build Settings
```

左侧选择：

```text
Android
```

然后点击：

```text
Switch Platform
```

如果 Switch Platform 按钮不能点，通常是 Unity 没安装 Android Build Support。

### 5.8 设置 Android Player Settings

在 Build Settings 里点击：

```text
Player Settings
```

检查这些配置。

#### Company Name / Product Name

```text
Company Name: tqq
Product Name: tqq
```

Product Name 会影响 App 在 Quest 里显示的名字。

#### Package Name

路径：

```text
Player
-> Android
-> Other Settings
-> Identification
-> Package Name
```

当前应该是：

```text
com.tqq.teleop
```

如果你以前装过旧的 `com.DefaultCompany.Myproject`，那是旧 App。现在这个新包名会安装成新的 App。

如果要卸载旧 App：

```bash
adb uninstall com.DefaultCompany.Myproject
```

如果要卸载现在这个 App：

```bash
adb uninstall com.tqq.teleop
```

#### Minimum API Level

```text
Minimum API Level: Android 7.0 API Level 24 或更高
```

当前项目是 API 24。

#### Target Architectures

路径：

```text
Player
-> Android
-> Other Settings
-> Target Architectures
```

Quest3 必须使用：

```text
ARM64
```

如果 ARM64 不能勾选，常见原因：

```text
1. 当前平台还没有切到 Android。
2. Android Build Support 没装完整。
3. Scripting Backend 不是 IL2CPP。
```

处理方式：

```text
Player Settings
-> Android
-> Other Settings
-> Configuration
-> Scripting Backend: IL2CPP
-> Target Architectures: ARM64
```

#### Internet 权限

这个 App 要发 UDP，所以需要网络权限。Unity Android 一般会自动处理网络相关权限，但如果你遇到装上后完全发不出包，可以检查 Android manifest 是否包含：

```xml
<uses-permission android:name="android.permission.INTERNET" />
```

当前脚本使用 `UdpClient` 发 UDP，不需要 ROS、不需要特殊 native 插件。

### 5.9 设置 XR / OpenXR

Unity 菜单：

```text
Edit
-> Project Settings
-> XR Plug-in Management
```

选择 Android 标签页，确认：

```text
OpenXR 已勾选
```

再进入：

```text
XR Plug-in Management
-> OpenXR
```

Interaction Profiles 建议至少有：

```text
Meta Quest Touch Plus Controller Profile
Oculus Touch Controller Profile
```

OpenXR Features 里和 Meta/Quest 相关的选项应该启用。当前项目已经有 OpenXR 配置文件：

```text
Assets/XR/Loaders/OpenXRLoader.asset
Assets/XR/Settings/OpenXR Package Settings.asset
```

如果 Project Settings 里看不到 XR Plug-in Management：

```text
Window
-> Package Manager
-> Unity Registry
-> 安装 XR Plug-in Management
-> 安装 OpenXR Plugin
```

当前项目的 `Packages/manifest.json` 已包含：

```json
"com.unity.xr.management": "4.6.0",
"com.unity.xr.openxr": "1.14.3"
```

### 5.10 Quest3 开发者模式

要让 Unity 直接 Build And Run 到 Quest3，头显必须开启开发者模式。

通常步骤：

```text
1. 注册 Meta 开发者账号。
2. 创建一个 Organization。
3. 手机 Meta Horizon App 连接 Quest3。
4. 在设备设置里打开 Developer Mode。
5. 重启 Quest3。
```

如果没有开发者模式，Unity/adb 可能看不到设备。

### 5.11 连接 Quest3 到电脑

用 USB-C 数据线连接 Quest3 和电脑。

Quest3 头显里会弹出 USB debugging 授权：

```text
Allow USB debugging?
```

选择允许。如果有 Always allow from this computer，可以勾选。

电脑上检查：

```bash
adb devices
```

正常应该看到：

```text
List of devices attached
XXXXXXXX	device
```

如果显示：

```text
unauthorized
```

说明头显里还没点允许。

如果没有 adb：

```bash
sudo apt update
sudo apt install android-tools-adb
```

### 5.12 Build Settings 里选择设备

Unity：

```text
File
-> Build Settings
-> Android
```

确认：

```text
Run Device: 你的 Quest3 设备
```

如果 Run Device 没有 Quest3：

```text
1. adb devices 看设备是否存在。
2. Quest3 是否授权 USB debugging。
3. USB 线是否支持数据传输。
4. Unity 是否安装 Android SDK/NDK/OpenJDK。
```

### 5.13 Build And Run

在 Build Settings 点击：

```text
Build And Run
```

第一次构建会比较慢。

如果 Unity 要你选择 APK 保存位置，可以选项目下的：

```text
/home/tqq/My project/Builds/tqq.apk
```

构建成功后，Unity 会自动安装到 Quest3 并启动 App。

### 5.14 在 Quest3 里找到 App

如果 Build And Run 成功但没有自动打开，可以在 Quest3 里找：

```text
Apps
-> 右上角筛选
-> Unknown Sources
-> tqq
```

如果你不知道包是否安装成功，在电脑运行：

```bash
adb shell pm list packages -3
```

应该能看到：

```text
package:com.tqq.teleop
```

也可以直接启动：

```bash
adb shell monkey -p com.tqq.teleop 1
```

### 5.15 App 内输入电脑 IP

打开 App 后，眼前会出现一个 IP 面板。

面板显示：

```text
PC IP
xxx.xxx.xxx.xxx:5055
```

操作方式：

```text
移动右手柄射线指向数字键
Trigger / Grip / A / 摇杆按下：确认当前选中的键
B：关闭面板
摇杆：也可以移动选择键
OK：保存 IP
DEL：删除一位
CLR：清空
CLOSE：关闭不保存
```

电脑 IP 用下面命令查：

```bash
hostname -I
```

例如电脑 IP 是：

```text
192.168.1.23
```

就在 App 里输入：

```text
192.168.1.23
```

然后点 `OK`。

保存后 IP 会写入 Quest3 本地 `PlayerPrefs`，下次打开 App 会自动记住。

如果以后电脑 IP 变了，不需要重新 Build App，只要在 App 里重新输入 IP。

打开 IP 面板的方法：

```text
启动时默认显示；
或者按下右手柄摇杆；
或者长按 B 超过约 1.2 秒。
```

### 5.16 App 运行时按钮含义

遥操时：

```text
右手柄 grip
  按住才启用机械臂主体移动。

右手柄 trigger
  控制夹爪开合。

右手柄 A
  标定按钮。当前配置 use_heading_calibration: false，所以一般不会影响移动。

右手柄 B
  重置标定或关闭 IP 面板。
```

### 5.17 验证 App 是否真的发 UDP

电脑上先启动 tcpdump：

```bash
sudo tcpdump -i any udp port 5055
```

然后打开 Quest3 App。

如果有 UDP 包，会看到持续刷新的 packet。

只看到：

```text
listening on any
```

说明还没收到包。优先检查：

```text
1. App 里的电脑 IP 是否正确。
2. Quest3 和电脑是否在同一个 Wi-Fi。
3. 电脑防火墙是否放行 UDP 5055。
4. Quest3 是否连上网络。
```

### 5.18 Unity 常见问题

#### ARM64 不能勾选

检查：

```text
Build Settings 是否切到 Android
Scripting Backend 是否是 IL2CPP
Android Build Support 是否安装完整
```

#### Build 失败，提示 SDK/NDK/JDK

回到 Unity Hub：

```text
Installs
-> Unity 2022.3.x 齿轮
-> Add modules
-> Android SDK & NDK Tools
-> OpenJDK
```

#### adb devices 看不到 Quest3

检查：

```text
Quest3 是否开启 Developer Mode
头显里是否允许 USB debugging
USB 线是否是数据线
电脑是否安装 android-tools-adb
```

#### App 只有文字或粉色框

这个 App 本身就是一个很轻量的 UDP sender，不是完整 3D 游戏。它的主要界面就是 IP 面板。只要 ROS 端能收到 `/quest3/right_controller/pose`，就是正常的。

如果出现大面积粉色材质，一般是 Unity shader/渲染管线问题，但当前 App 主要用 `TextMesh` 显示文字，不影响 UDP 发送。

#### App 打开后机械臂没反应

先别看机械臂，先看电脑 ROS 话题：

```bash
ros2 topic echo /quest3/right_controller/pose
```

如果没有 pose，说明 Unity/Quest/UDP/IP 这段还没通。

### 5.19 如果没有现成 Unity 项目，怎么从空项目创建

如果你是在一台新电脑上从零做，不是打开已有的 `/home/tqq/My project`，可以这样创建。

1. Unity Hub 新建项目：

```text
Projects
-> New project
-> 选择 3D Core
-> Editor 选择 Unity 2022.3.x LTS
-> Project name: My project
-> Location: /home/tqq
-> Create project
```

2. 安装 XR 包：

```text
Window
-> Package Manager
-> Unity Registry
-> 安装 XR Plug-in Management
-> 安装 OpenXR Plugin
```

3. 启用 OpenXR：

```text
Edit
-> Project Settings
-> XR Plug-in Management
-> Android
-> 勾选 OpenXR
```

4. 设置 Android：

```text
File
-> Build Settings
-> Android
-> Switch Platform
```

5. 设置包名和架构：

```text
Edit
-> Project Settings
-> Player
-> Android
-> Product Name: tqq
-> Package Name: com.tqq.teleop
-> Scripting Backend: IL2CPP
-> Target Architectures: ARM64
```

6. 创建脚本：

```text
Assets
-> Create
-> C# Script
-> 命名 Quest3UdpSender
```

把本工程里的标准脚本复制到 Unity 项目的 `Assets` 目录：

```bash
cp ~/TQQ_ws/tqq/src/franka_vr_teleop/unity/Quest3UdpSender.cs \
  "/home/tqq/My project/Assets/Quest3UdpSender.cs"
```

如果你是用 Unity 菜单先创建了一个空的 `Quest3UdpSender.cs`，就用上面的复制命令覆盖它。

7. 创建空物体并挂脚本：

```text
Hierarchy
-> Create Empty
-> 命名 Quest3UdpSender
-> Inspector
-> Add Component
-> Quest3UdpSender
```

8. 保存场景：

```text
File
-> Save
```

9. 把场景加入 Build：

```text
File
-> Build Settings
-> Add Open Scenes
```

10. 连接 Quest3，执行：

```text
Build And Run
```

新手建议优先使用已经存在的 `/home/tqq/My project`，因为 OpenXR、场景、包名、脚本都已经设置好。从空项目创建只是备用路线。


## 6. Quest3 App 端数据格式

Unity 脚本位置：

```bash
/home/tqq/My project/Assets/Quest3UdpSender.cs
```

这个脚本做了几件事：

1. 读取 Quest3 右手柄位置：

```csharp
CommonUsages.devicePosition
```

2. 读取 Quest3 右手柄旋转：

```csharp
CommonUsages.deviceRotation
```

3. 读取按钮：

```text
gripButton      启用遥操
triggerButton   控制夹爪
primaryButton   A 键，标定相关
secondaryButton B 键，重置相关
```

4. 按 60Hz 发送 UDP JSON 到电脑：

```json
{
  "pose": {
    "position": [0.12, -0.03, 1.42],
    "orientation": [0.0, 0.0, 0.0, 1.0]
  },
  "grip_pressed": true,
  "trigger_pressed": false,
  "calibrate_pressed": false,
  "reset_calibration_pressed": false
}
```

### 6.1 查电脑 IP

在电脑上运行：

```bash
hostname -I
```

或者：

```bash
ip -4 addr show
```

找和 Quest3 在同一个 Wi-Fi/局域网下的 IP，例如：

```text
192.168.1.23
```

Quest3 App 里要填的就是这个电脑 IP。

### 6.2 UDP 端口

当前 ROS 端监听端口是：

```yaml
listen_port: 5055
```

配置文件：

```bash
~/TQQ_ws/tqq/src/franka_vr_teleop/config/quest3_teleop.yaml
```

Unity 端也要发到同一个端口：

```csharp
public int pcPort = 5055;
```

### 6.3 防火墙

如果电脑开了防火墙，可以放行 UDP 5055：

```bash
sudo ufw allow 5055/udp
```

### 6.4 如何确认 Quest3 真的发包了

电脑终端运行：

```bash
sudo tcpdump -i any udp port 5055
```

如果只看到：

```text
listening on any, link-type LINUX_SLL2
```

这只是表示 tcpdump 正在监听，还不代表有包进来。

当 Quest3 App 真的发包时，会出现一行一行的 UDP packet 日志。


## 7. 假硬件测试

第一次不要直接上真机，先用假硬件确认 ROS 链路。

终端 1：

```bash
cd ~/TQQ_ws
source setup_franka.sh

cd ~/TQQ_ws/tqq
source install/setup.bash

ros2 launch franka_vr_teleop quest3_franka_teleop.launch.py \
  use_fake_hardware:=true
```

正常日志应该包含：

```text
Loaded vr_cartesian_velocity_controller
Configured and activated vr_cartesian_velocity_controller
Quest3 UDP bridge listening on 0.0.0.0:5055
Quest3 pose bridge ready
Quest3 gripper bridge ready
```

终端 2 检查控制器：

```bash
source ~/TQQ_ws/setup_franka.sh
source ~/TQQ_ws/tqq/install/setup.bash

ros2 control list_controllers
```

应该看到：

```text
joint_state_broadcaster              active
vr_cartesian_velocity_controller     active
```

终端 3 检查 Quest3 pose：

```bash
source ~/TQQ_ws/setup_franka.sh
source ~/TQQ_ws/tqq/install/setup.bash

ros2 topic echo /quest3/right_controller/pose
```

如果 Quest3 App 没打开或 IP 不对，可能出现：

```text
WARNING: topic [/quest3/right_controller/pose] does not appear to be published yet
Could not determine the type for the passed topic
```

这是正常提示，意思是 ROS 还没收到这个话题。

当 Quest3 App 发包成功时，会持续输出：

```yaml
pose:
  position:
    x: ...
    y: ...
    z: ...
  orientation:
    x: ...
    y: ...
    z: ...
    w: ...
```

终端 4 检查手柄是否转换成速度：

```bash
ros2 topic echo /vr_cartesian_velocity_controller/twist_cmd
```

按住右手柄 grip 并移动手柄时，应该看到 `linear` 或 `angular` 里面出现非零值。

松开 grip 后，应该回到接近 0。


## 8. 真机启动

真机前先确认：

```text
1. 机械臂周围没有人和障碍物。
2. Franka Desk 状态正常，FCI 可用。
3. 急停按钮在手边。
4. 先小幅移动手柄，不要猛甩。
5. 不要同时启动 MoveIt/Servo 控制同一台机械臂。
```

启动命令：

```bash
cd ~/TQQ_ws
source setup_franka.sh

cd ~/TQQ_ws/tqq
source install/setup.bash

ros2 launch franka_vr_teleop quest3_franka_teleop.launch.py \
  robot_ip:=192.168.22.212 \
  use_fake_hardware:=false
```

启动后：

1. 打开 Quest3 App。
2. 确认 App 里电脑 IP 是当前电脑 IP。
3. 等终端显示 UDP bridge ready。
4. 按住右手柄 grip。
5. 缓慢移动右手柄。
6. 松开 grip，机械臂应停止。
7. 按 trigger，夹爪开合。


## 9. 启动文件做了什么

文件：

```bash
~/TQQ_ws/tqq/src/franka_vr_teleop/launch/quest3_franka_teleop.launch.py
```

它启动了这些东西：

```text
franka_bringup/launch/franka.launch.py
  启动 Franka ROS2 driver、robot_state_publisher、controller_manager、gripper 等。

spawner vr_cartesian_velocity_controller
  加载并激活自定义笛卡尔速度控制器。

quest3_udp_bridge.py
  监听 UDP 5055，把 Quest3 JSON 转为 ROS 话题。

quest3_pose_to_twist.py
  把手柄位姿变化转为机械臂末端速度。

quest3_gripper_bridge.py
  把 trigger 按钮转成夹爪 action。

rviz2
  显示机器人模型。
```

常用 launch 参数：

```bash
robot_ip:=192.168.22.212
use_fake_hardware:=false
launch_udp_bridge:=true
launch_gripper_bridge:=true
launch_rviz:=true
```

例如不启动 RViz：

```bash
ros2 launch franka_vr_teleop quest3_franka_teleop.launch.py \
  robot_ip:=192.168.22.212 \
  use_fake_hardware:=false \
  launch_rviz:=false
```


## 10. 坐标轴映射

Quest3 手柄坐标和 Franka 基座坐标不一定一致，所以需要映射。

配置文件：

```bash
~/TQQ_ws/tqq/src/franka_vr_teleop/config/quest3_axis_mapping.yaml
```

当前配置：

```yaml
linear_axis_map: ["x", "z", "y"]
angular_axis_map: ["-x", "-z", "-y"]
```

含义：

```text
base_x <- controller_x
base_y <- controller_z
base_z <- controller_y
```

`-x` 表示方向取反。

如果你发现：

```text
手柄往左，机械臂往前
手柄往上，机械臂往侧面
旋转方向反了
```

优先改这个文件，而不是改代码。

改完后重启 launch 生效，不需要重新编译。


## 11. 速度、死区和平滑参数

配置文件：

```bash
~/TQQ_ws/tqq/src/franka_vr_teleop/config/quest3_teleop.yaml
```

关键参数：

```yaml
linear_gain: 2.4
angular_gain: 0.5
max_linear_velocity_mps: 0.12
max_angular_velocity_radps: 0.25
deadband_linear_velocity_mps: 0.01
deadband_angular_velocity_radps: 0.03
input_smoothing_alpha: 0.15
idle_decay_alpha: 0.08
```

解释：

```text
linear_gain
  手柄线速度放大倍数。越大越跟手，也越容易抖。

angular_gain
  手柄旋转速度放大倍数。

max_linear_velocity_mps
  机械臂末端线速度上限，单位 m/s。

max_angular_velocity_radps
  机械臂末端角速度上限，单位 rad/s。

deadband_linear_velocity_mps
  小于这个值的手柄线速度当作 0，用于抑制手柄噪声。

deadband_angular_velocity_radps
  小于这个值的手柄角速度当作 0。

input_smoothing_alpha
  输入低通滤波。越小越平滑，但延迟越大。

idle_decay_alpha
  停止输入时回到零速度的快慢。越大归零越快。
```

如果机械臂太慢：

```yaml
linear_gain: 3.0
max_linear_velocity_mps: 0.15
```

如果机械臂太抖：

```yaml
deadband_linear_velocity_mps: 0.02
input_smoothing_alpha: 0.08
```

真机上不要一次把速度调太大。Franka 对速度和加速度不连续很敏感，过激输入可能触发 reflex。


## 12. 自定义控制器参数

配置文件：

```bash
~/TQQ_ws/tqq/src/franka_vr_teleop/config/vr_teleop_controllers.yaml
```

当前控制器参数：

```yaml
vr_cartesian_velocity_controller:
  ros__parameters:
    twist_topic: ~/twist_cmd
    command_timeout_sec: 0.20
    max_linear_velocity_mps: 0.12
    max_angular_velocity_radps: 0.25
    max_linear_acceleration_mps2: 0.12
    max_angular_acceleration_radps2: 0.30
    lowpass_alpha: 0.03
```

注意：这里的 `~/twist_cmd` 会解析成：

```bash
/vr_cartesian_velocity_controller/twist_cmd
```

这正好和 `quest3_pose_to_twist.py` 发布的 `twist_topic` 对上。

控制器内部做了三层保护：

```text
1. 速度限幅
   linear 不超过 max_linear_velocity_mps
   angular 不超过 max_angular_velocity_radps

2. 加速度限幅
   每个控制周期内速度变化不能太突兀。

3. 超时归零
   如果 command_timeout_sec 内没有新命令，就逐渐回到 0。
```

这样做是为了避免 Quest3 断流、Wi-Fi 丢包、手柄突然跳变时，机械臂继续执行旧命令。


## 13. C++ 控制器实现原理

核心文件：

```bash
~/TQQ_ws/tqq/src/franka_vr_teleop/src/vr_cartesian_velocity_controller.cpp
```

它是一个 `ros2_control` controller plugin。

### 13.1 它向 controller_manager 申请什么接口

```cpp
franka_cartesian_velocity_->get_command_interface_names()
```

这会申请 Franka 的笛卡尔速度命令接口：

```text
vx vy vz wx wy wz
```

也就是 3 个线速度和 3 个角速度。

### 13.2 它订阅什么话题

```cpp
geometry_msgs::msg::TwistStamped
```

话题名来自参数：

```yaml
twist_topic: ~/twist_cmd
```

### 13.3 每个控制周期做什么

伪代码：

```text
if 收到的新命令没有超时:
    command = latest_command
else:
    command = 0

for 每个速度分量:
    先做低通滤波
    再做加速度限制

linear = [vx, vy, vz]
angular = [wx, wy, wz]
franka_cartesian_velocity_->setCommand(linear, angular)
```

这个控制器跟 MoveIt 没关系，它直接把末端速度写给 Franka 硬件接口。


## 14. UDP 桥接实现原理

文件：

```bash
~/TQQ_ws/tqq/src/franka_vr_teleop/scripts/quest3_udp_bridge.py
```

它做的事情：

```text
1. 创建 UDP socket。
2. bind 到 0.0.0.0:5055。
3. 等待 Quest3 发 JSON。
4. 解析 position、orientation、buttons。
5. 发布 ROS topic。
```

发布的话题：

```bash
/quest3/right_controller/pose
/quest3/right_controller/grip_pressed
/quest3/right_controller/trigger_pressed
/quest3/right_controller/calibrate_pressed
/quest3/right_controller/reset_calibration_pressed
/quest3/head_forward
```

这里的 `/quest3/head_forward` 主要给头部朝向标定用。当前你的配置里：

```yaml
use_heading_calibration: false
```

所以原始坐标映射主要由 `quest3_axis_mapping.yaml` 控制。


## 15. Pose 转 Twist 实现原理

文件：

```bash
~/TQQ_ws/tqq/src/franka_vr_teleop/scripts/quest3_pose_to_twist.py
```

当前模式：

```yaml
motion_mode: hand_velocity
```

线速度计算：

```text
dx = current_position.x - previous_position.x
vx_controller = dx / dt
```

角速度计算：

```text
q_delta = q_current * inverse(q_previous)
rotvec = quaternion_to_rotation_vector(q_delta)
angular_velocity = rotvec / dt
```

然后：

```text
1. 根据 quest3_axis_mapping.yaml 做坐标轴映射。
2. 根据 deadband 去掉小噪声。
3. 乘以 linear_gain / angular_gain。
4. 根据 max_linear_velocity_mps / max_angular_velocity_radps 限幅。
5. 根据 input_smoothing_alpha 做平滑。
6. 发布 TwistStamped 到 /vr_cartesian_velocity_controller/twist_cmd。
```


## 16. 常用调试命令

### 16.1 看所有 Quest3 话题

```bash
ros2 topic list | grep quest3
```

### 16.2 看 UDP bridge 是否启动

```bash
ros2 node list | grep quest3_udp_bridge
```

### 16.3 看 pose 是否进来

```bash
ros2 topic echo /quest3/right_controller/pose
```

### 16.4 看 grip 是否进来

```bash
ros2 topic echo /quest3/right_controller/grip_pressed
```

### 16.5 看 trigger 是否进来

```bash
ros2 topic echo /quest3/right_controller/trigger_pressed
```

### 16.6 看速度命令

```bash
ros2 topic echo /vr_cartesian_velocity_controller/twist_cmd
```

### 16.7 看控制器状态

```bash
ros2 control list_controllers
```

### 16.8 看当前参数

```bash
ros2 param list /quest3_pose_to_twist
ros2 param get /quest3_pose_to_twist linear_gain
ros2 param get /quest3_pose_to_twist max_linear_velocity_mps
```

### 16.9 手动发一个小速度测试

只建议在假硬件或非常安全的真机环境下做。

```bash
ros2 topic pub --once /vr_cartesian_velocity_controller/twist_cmd \
  geometry_msgs/msg/TwistStamped \
  "{header: {frame_id: fr3_link0}, twist: {linear: {x: 0.01, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}}"
```


## 17. 常见问题

### 17.1 第二个终端 echo 一直卡住

例如：

```bash
ros2 topic echo /quest3/right_controller/pose
```

如果没有输出，通常是因为还没有消息进来。检查：

```text
1. Quest3 App 是否打开。
2. App 里的电脑 IP 是否正确。
3. Quest3 和电脑是否在同一个局域网。
4. UDP 5055 是否被防火墙挡住。
5. 第一个终端的 launch 是否启动了 quest3_udp_bridge.py。
```

### 17.2 tcpdump 只有 listening

```bash
sudo tcpdump -i any udp port 5055
```

只显示 listening 不代表收到包。收到包时会继续刷 UDP 日志。

### 17.3 夹爪能动，但机械臂主体不动

按顺序检查：

```bash
ros2 topic echo /quest3/right_controller/grip_pressed
```

确认按住 grip 时是：

```yaml
data: true
```

再看：

```bash
ros2 topic echo /vr_cartesian_velocity_controller/twist_cmd
```

按住 grip 并移动手柄，`linear` 是否有非零值。

再看：

```bash
ros2 control list_controllers
```

确认：

```text
vr_cartesian_velocity_controller active
```

### 17.4 移动方向不对

改：

```bash
~/TQQ_ws/tqq/src/franka_vr_teleop/config/quest3_axis_mapping.yaml
```

例如方向反了，就加负号：

```yaml
linear_axis_map: ["-x", "z", "y"]
```

改完重启 launch。

### 17.5 一动就触发 Franka reflex

典型报错：

```text
cartesian_motion_generator_joint_acceleration_discontinuity
```

意思是速度或加速度变化太突兀。

优先降低：

```yaml
linear_gain
max_linear_velocity_mps
max_linear_acceleration_mps2
```

对应文件：

```bash
config/quest3_teleop.yaml
config/vr_teleop_controllers.yaml
```

也要注意实际操作：不要按住 grip 后猛甩手柄。

### 17.6 RViz 里看不到明显运动

假硬件模式主要用于验证节点、话题、控制器是否正常，不一定能完整模拟 Franka 真实笛卡尔速度运动效果。

确认链路时优先看：

```bash
ros2 topic echo /quest3/right_controller/pose
ros2 topic echo /vr_cartesian_velocity_controller/twist_cmd
ros2 control list_controllers
```


## 18. 新人应该怎么学这套系统

建议按这个顺序理解：

```text
1. 先理解 ROS topic：
   Quest3 pose 和 button 是普通 ROS topic。

2. 再理解 Twist：
   机械臂不是收到目标点，而是收到末端速度。

3. 再理解坐标系：
   手柄坐标系和机器人基座坐标系不一样，所以需要 axis mapping。

4. 再理解 ros2_control：
   vr_cartesian_velocity_controller 是一个 controller plugin。

5. 最后理解 Franka 接口：
   这个控制器最终写入 Franka 的 cartesian_velocity command interface。
```

一句话总结：

```text
Quest3 负责提供人的手柄运动；
ROS2 负责把运动转成机器人能理解的速度；
Franka controller 负责把速度安全地交给机械臂底层。
```


## 19. 最小启动清单

每次真机使用，最少做这些：

```bash
cd ~/TQQ_ws
source setup_franka.sh

cd ~/TQQ_ws/tqq
source install/setup.bash

ros2 launch franka_vr_teleop quest3_franka_teleop.launch.py \
  robot_ip:=192.168.22.212 \
  use_fake_hardware:=false
```

然后：

```text
1. Quest3 打开 tqq App。
2. App 里确认电脑 IP。
3. 右手柄按住 grip。
4. 小幅移动手柄。
5. 松开 grip 停止。
6. trigger 控制夹爪。
```

如果出问题，按这个顺序查：

```text
UDP 是否进来
-> /quest3/right_controller/pose 是否有数据
-> /quest3/right_controller/grip_pressed 是否为 true
-> /vr_cartesian_velocity_controller/twist_cmd 是否非零
-> vr_cartesian_velocity_controller 是否 active
-> Franka 真机状态是否正常
```
