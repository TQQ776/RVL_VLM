# mcp

Voice-driven robot tools for FR3.

## Services

`mcp_server` provides:

```text
/mcp_server/go_home
/mcp_server/move_x_cm
/mcp_server/move_y_cm
/mcp_server/move_z_cm
/mcp_server/open_gripper
/mcp_server/close_gripper
/mcp_server/place_relative_to_object
/mcp_server/place_into_container
/mcp_server/list_tools
/mcp_server/call_tool
```

`go_home` uses the saved home joints from the `control` package.

The axis services move the end effector in the base frame:

```text
base frame: fr3_link0
end effector: fr3_hand_tcp
unit: centimeters
single-step limit: 10 cm by default
```

The axis services reject any single request whose absolute distance is greater than
`max_single_axis_move_cm`.

If a robot tool is rejected or fails, the MCP client reports the failure directly
and does not split the request into smaller motions automatically.

For API-vision grasping, `mcp_server` sends the latest camera image to the
vision model, publishes detection JSON on `/mcp_omni_client/api_detections_json`,
and sends a one-shot target command to the EconomicGrasp ROI controller. The ROI
controller uses the API box, RGB-D points, TF to `fr3_link0`, MoveIt planning,
and the Franka gripper to execute the grasp.

## Build

```bash
cd ~/TQQ_ws/tqq
source ~/TQQ_ws/setup_franka.sh
colcon build --packages-select audio_dialog mcp --symlink-install
source install/setup.bash
```

## Run

### One-Command API-Vision Grasp Stack

This starts RealSense, FR3 MoveIt, hand-eye TF, EconomicGrasp ROI,
`mcp_server`, and `mcp_omni_client` together. Object recognition is done
through the vision API:

```bash
source ~/TQQ_ws/setup_franka.sh
export DASHSCOPE_API_KEY=your_dashscope_key

ros2 launch mcp llm_api_grasp.launch.py
```

Useful overrides:

```bash
ros2 launch mcp llm_api_grasp.launch.py \
  robot_ip:=192.168.22.212 \
  handeye_name:=fr3_d435i_handeye \
  max_velocity_scaling:=0.01 \
  max_acceleration_scaling:=0.01
```

If one part is already running, disable it:

```bash
ros2 launch mcp llm_api_grasp.launch.py \
  launch_camera:=false \
  launch_moveit:=false
```

Start MoveIt first:

```bash
ros2 launch franka_fr3_moveit_config moveit.launch.py \
  robot_ip:=192.168.22.212 \
  use_fake_hardware:=false
```

Start the MCP service node:

```bash
ros2 launch mcp mcp_server.launch.py \
  params_file:=/home/tqq/TQQ_ws/tqq/src/mcp/config/mcp_server.yaml
```

Start the Qwen-Omni multimodal client. The recorded audio is sent directly to
Qwen-Omni, and the node executes robot tools through Aliyun's official Function
Calling / tools.

The default config uses Qwen3.5 Omni Realtime:

```yaml
omni_model: qwen3.5-omni-plus-realtime
omni_text_model: qwen3.5-omni-plus
omni_native_audio_output_enabled: true
omni_native_audio_fallback_to_local_tts: false
omni_realtime_voice: Ethan
```

Voice input and voice output both use Qwen3.5-Omni Realtime. Robot control uses
Aliyun's official Function Calling / tools instead of a custom JSON action plan.
After local tool execution, the final reply is spoken with Omni native audio
instead of edge-tts.

Install the DashScope Realtime SDK and pyaudio before using it:

```bash
/usr/bin/python3 -m pip install --user -U "dashscope>=1.23.9" websocket-client pyaudio
```

Tune native voice output in `config/mcp_omni_client.yaml`:

```yaml
omni_realtime_voice: Ethan
omni_speech_rate: normal
omni_speech_volume: normal
omni_speech_emotion: natural
omni_speech_style: 清晰、自然、友好，适合机器人语音助手
```

Temporary launch overrides also work:

```bash
ros2 launch mcp mcp_omni_client.launch.py \
  params_file:=/home/tqq/TQQ_ws/tqq/src/mcp/config/mcp_omni_client.yaml \
  omni_realtime_voice:=Ethan \
  omni_speech_rate:=fast \
  omni_speech_emotion:=happy
```

```bash
export DASHSCOPE_API_KEY=your_dashscope_key
ros2 launch mcp mcp_omni_client.launch.py \
  params_file:=/home/tqq/TQQ_ws/tqq/src/mcp/config/mcp_omni_client.yaml
```

The text/voice dialog opens automatically after startup:

```text
Press Enter to send text, Shift+Enter to insert a newline.
Click the record button once to start recording, then click it again to stop and execute.
Press Esc to close the dialog.
If the answer is still being spoken, sending the next message or starting a recording interrupts playback.
```

Example commands:

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
你现在能看到什么
你现在能抓什么
现在有哪些可以抓的东西
打开夹爪
关闭夹爪
抓橘子
拿苹果
```

Direction mapping:

```text
右 = 基座 x 正方向，左 = 基座 x 负方向
前 = 基座 y 正方向，后 = 基座 y 负方向
上 = 基座 z 正方向，下 = 基座 z 负方向
```

## Manual Service Tests

```bash
ros2 service call /mcp_server/go_home std_srvs/srv/Trigger {}
ros2 service call /mcp_server/move_x_cm mcp/srv/MoveAxis "{centimeters: 2.0}"
ros2 service call /mcp_server/move_y_cm mcp/srv/MoveAxis "{centimeters: -2.0}"
ros2 service call /mcp_server/move_z_cm mcp/srv/MoveAxis "{centimeters: 1.0}"
ros2 service call /mcp_server/open_gripper std_srvs/srv/Trigger {}
ros2 service call /mcp_server/close_gripper std_srvs/srv/Trigger {}
ros2 service call /mcp_server/list_tools mcp/srv/ListTools {}
ros2 service call /mcp_server/call_tool mcp/srv/CallTool "{name: grab_api_object, arguments_json: '{\"object_name\":\"orange\",\"motion_speed\":0.05}'}"
```

Change the single-step limit in `config/mcp_server.yaml`:

```yaml
max_single_axis_move_cm: 10.0
```
