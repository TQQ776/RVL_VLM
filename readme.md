# 标定
1.启动相机 ros2 launch franka_camera realsense.launch.py camera_namespace:=/
export DASHSCOPE_API_KEY=你的 DashScope API Key
2.检测掩码 ros2 run aruco_ros single   --ros-args   -p marker_id:=582   -p marker_size:=0.15   -p reference_frame:=camera_color_optical_frame   -p camera_frame:=camera_color_optical_frame   -p marker_frame:=aruco_marker_frame   -p image_is_rectified:=true   -r /image:=/camera/color/image_raw   -r /camera_info:=/camera/color/camera_info

3.启动驱动 ros2 launch franka_bringup franka.launch.py     robot_type:=fr3     robot_ip:=192.168.22.212     load_gripper:=true


4. 标定软件 ros2 launch easy_handeye2 calibrate.launch.py   calibration_type:=eye_in_hand   name:=fr3_d435i_handeye   robot_base_frame:=fr3_link0   robot_effector_frame:=fr3_hand_tcp   tracking_base_frame:=camera_color_optical_frame   tracking_marker_frame:=aruco_marker_frame

5. 发布标定文件 ros2 launch easy_handeye2 publish.launch.py name:=fr3_d435i_handeye


# Quest3 VR 遥操

先编译并加载工作空间：

```bash
cd ~/TQQ_ws
source setup_franka.sh

cd ~/TQQ_ws/tqq
colcon build --packages-select franka_vr_teleop --symlink-install
source install/setup.bash
```

Franka Cartesian velocity 直控模式：

```bash
ros2 launch franka_vr_teleop quest3_franka_teleop.launch.py \
  robot_ip:=192.168.22.212 \
  use_fake_hardware:=false
```

Quest app 仍然发右手柄位姿到电脑 UDP 5055。按住 grip 才启用遥操，trigger 控制夹爪。

常用配置文件：

```text
~/TQQ_ws/tqq/src/franka_vr_teleop/config/quest3_axis_mapping.yaml
  手柄坐标轴到 FR3 基座坐标轴的映射
```



# moveit+yolo识别，分开启动

## 终端1：启动 RealSense 对齐深度

```bash
ros2 launch franka_camera realsense.launch.py \
  camera_namespace:=camera \
  camera_name:=camera \
  params_file:=/home/tqq/TQQ_ws/tqq/src/control/config/realsense_aligned_depth.yaml
```

检查深度话题：

```bash
ros2 topic list | grep depth
```

应该能看到：

```bash
/camera/camera/aligned_depth_to_color/image_raw
```

## 终端2：启动 YOLO

```bash
source ~/TQQ_ws/setup_franka.sh
source ~/TQQ_ws/tqq/install/setup.bash

ros2 launch yolo yolo_realsense.launch.py \
  image_topic:=/camera/camera/color/image_raw
```

## 终端3：启动 MoveIt/机械臂

```bash

ros2 launch franka_fr3_moveit_config moveit.launch.py \
  robot_ip:=192.168.22.212 \
  use_fake_hardware:=false
```

## 终端4：只启动 control 节点

速度从 object_target_control.yaml 读取：

```bash

ros2 launch control object_target_control.launch.py \
  params_file:=/home/tqq/TQQ_ws/tqq/src/control/config/object_target_control.yaml \
  launch_camera:=false \
  launch_yolo:=false \
  launch_moveit:=false
```

如果想临时覆盖速度，不改 yaml：

```bash
ros2 launch control object_target_control.launch.py \
  params_file:=/home/tqq/TQQ_ws/tqq/src/control/config/object_target_control.yaml \
  launch_camera:=false \
  launch_yolo:=false \
  launch_moveit:=false \
  max_velocity_scaling:=0.01 \
  max_acceleration_scaling:=0.01
```

检查当前生效速度：

```bash
ros2 param get /object_target_controller max_velocity_scaling
ros2 param get /object_target_controller max_acceleration_scaling
```

## 终端5：选择要抓取的物体

```bash
ros2 run control object_target_cli
```

## 回到 home 位置

```bash
ros2 launch control move_home.launch.py \
  params_file:=/home/tqq/TQQ_ws/tqq/src/control/config/move_home.yaml
```


# 语音 MCP 控制机械臂

## 一键启动 LLM + API 视觉抓取栈

这个 launch 会一起启动 RealSense、MoveIt/FR3、手眼 TF、抓取控制节点、
MCP 服务端和 Qwen-Omni 客户端：

```bash
source ~/TQQ_ws/setup_franka.sh
source ~/TQQ_ws/tqq/install/setup.bash
export DASHSCOPE_API_KEY=你的 DashScope API Key

ros2 launch mcp llm_yolo_grasp.launch.py
```

当前这条一键启动链路默认不启动 YOLO。你说“现在能抓什么”或“抓橘子”时，
Omni 会把当前相机图发给视觉 API，让 API 返回目标框，再由本地深度、TF 和
MoveIt 执行抓取。

当前 Qwen-Omni 默认模型配置在：

```yaml
/home/tqq/TQQ_ws/tqq/src/mcp/config/mcp_omni_client.yaml
omni_model: qwen3.5-omni-plus-realtime
omni_text_model: qwen3.5-omni-plus
```

先启动 MoveIt/机械臂，也就是上面的终端3：

```bash
source ~/TQQ_ws/setup_franka.sh
source ~/TQQ_ws/tqq/install/setup.bash

ros2 launch franka_fr3_moveit_config moveit.launch.py \
  robot_ip:=192.168.22.212 \
  use_fake_hardware:=false
```

## 终端4：启动 MCP 服务端

```bash
source ~/TQQ_ws/setup_franka.sh
source ~/TQQ_ws/tqq/install/setup.bash

ros2 launch mcp mcp_server.launch.py \
  params_file:=/home/tqq/TQQ_ws/tqq/src/mcp/config/mcp_server.yaml
```

它提供这些服务：

```bash
/mcp_server/go_home
/mcp_server/move_x_cm
/mcp_server/move_y_cm
/mcp_server/move_z_cm
/mcp_server/grab_object
/mcp_server/open_gripper
/mcp_server/close_gripper
```

每次单步移动默认最大 10cm，超过会直接拒绝。限制参数在：

```yaml
/home/tqq/TQQ_ws/tqq/src/mcp/config/mcp_server.yaml
max_single_axis_move_cm: 10.0
```

## 终端5：启动 Qwen-Omni 全模态 MCP 客户端

这个节点不使用 Whisper/FunASR，不做“录音转文字”这一步。它会把录到的音频
直接发给阿里 Qwen-Omni，让模型直接理解语音，然后本地执行机械臂服务。
文本弹窗也会走 Qwen-Omni。

当前默认使用：

```yaml
omni_model: qwen3.5-omni-plus-realtime
omni_text_model: qwen3.5-omni-plus
omni_native_audio_output_enabled: true
omni_native_audio_fallback_to_local_tts: false
omni_realtime_voice: Ethan
omni_speech_rate: normal
omni_speech_emotion: natural
```

语音输入和语音输出都走 Qwen3.5-Omni Realtime。机械臂控制使用阿里云官方
Function Calling / tools，不再让模型输出自定义 JSON 控制计划。工具执行后，
最终结果再用 Omni 原生语音朗读，不再用 edge-tts。

第一次使用 Realtime 前需要安装阿里云 DashScope Realtime SDK 和 pyaudio：

```bash
/usr/bin/python3 -m pip install --user -U "dashscope>=1.23.9" websocket-client pyaudio
```

如果 `pyaudio` 安装失败，先装系统依赖：

```bash
sudo apt install portaudio19-dev python3-pyaudio
```
export DASHSCOPE_API_KEY=你的 DashScope API Key
调整音色、语速、情绪在这里改：

```yaml
/home/tqq/TQQ_ws/tqq/src/mcp/config/mcp_omni_client.yaml
omni_realtime_voice: Ethan
omni_speech_rate: normal
omni_speech_volume: normal
omni_speech_emotion: natural
omni_speech_style: 清晰、自然、友好，适合机器人语音助手
```

也可以启动时临时覆盖，例如：

```bash
ros2 launch mcp mcp_omni_client.launch.py \
  params_file:=/home/tqq/TQQ_ws/tqq/src/mcp/config/mcp_omni_client.yaml \
  omni_realtime_voice:=Ethan \
  omni_speech_rate:=fast \
  omni_speech_emotion:=happy
```

```bash
source ~/TQQ_ws/setup_franka.sh
source ~/TQQ_ws/tqq/install/setup.bash
export DASHSCOPE_API_KEY=你的 DashScope API Key

ros2 launch mcp mcp_omni_client.launch.py \
  params_file:=/home/tqq/TQQ_ws/tqq/src/mcp/config/mcp_omni_client.yaml
```

启动后会自动弹出对话窗口，不需要再按终端按键：

```text
输入文本后按 Enter 发送，Shift+Enter 换行。
点击“开始录音”开始说话，再次点击“停止录音并发送”执行语音命令。
如果回答还在播放，下一次发送文本或开始录音会自动打断当前语音。
```

可以说：

```text
回到 home 点
末端沿基座 x 轴移动 3 厘米
末端沿基座 y 轴移动负 2 厘米
末端沿基座 z 轴上升 5 厘米
向右移动 3 厘米
向左移动 3 厘米
向前移动 3 厘米
向后移动 3 厘米
向上移动 3 厘米
向下移动 3 厘米
你能看到什么
你现在能抓什么
现在有哪些可以抓的东西
打开夹爪
关闭夹爪
抓橘子
拿苹果
```

方向约定：

```text
右 = 基座 x 正方向，左 = 基座 x 负方向
前 = 基座 y 正方向，后 = 基座 y 负方向
上 = 基座 z 正方向，下 = 基座 z 负方向
```

## MCP 服务手动测试

```bash
ros2 service call /mcp_server/go_home std_srvs/srv/Trigger {}
ros2 service call /mcp_server/move_x_cm mcp/srv/MoveAxis "{centimeters: 2.0}"
ros2 service call /mcp_server/move_y_cm mcp/srv/MoveAxis "{centimeters: -2.0}"
ros2 service call /mcp_server/move_z_cm mcp/srv/MoveAxis "{centimeters: 1.0}"
ros2 service call /mcp_server/grab_object mcp/srv/ObjectName "{name: orange}"
ros2 service call /mcp_server/open_gripper std_srvs/srv/Trigger {}
ros2 service call /mcp_server/close_gripper std_srvs/srv/Trigger {}
```
