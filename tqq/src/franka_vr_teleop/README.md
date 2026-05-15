# franka_vr_teleop

Independent Quest 3 teleoperation package for Franka FR3. It does not use MCP.

The package contains:

- `quest3_udp_bridge.py`: receives Quest 3 JSON packets over UDP and republishes ROS topics.
- `VRCartesianVelocityController`: a `ros2_control` controller plugin that writes directly to Franka's official `cartesian_velocity` command interface.
- `quest3_pose_to_twist.py`: maps Quest 3 controller motion into the direct Cartesian velocity controller.
- `quest3_gripper_bridge.py`: maps a Quest 3 button to the Franka gripper action.

Quest 3 does not need to run ROS. Send UDP packets to the PC, and the bridge will republish them as ROS topics.

Expected UDP payload:

```json
{
  "pose": {
    "position": [0.12, -0.03, 1.42],
    "orientation": [0.0, 0.0, 0.0, 1.0]
  },
  "grip_pressed": true,
  "trigger_pressed": false
}
```

The bridge republishes:

```bash
/quest3/right_controller/pose            geometry_msgs/msg/PoseStamped
/quest3/right_controller/grip_pressed    std_msgs/msg/Bool
/quest3/right_controller/trigger_pressed  std_msgs/msg/Bool
```

Adjust host, port, and topic names in `config/quest3_teleop.yaml` if needed.

Build:

```bash
cd ~/TQQ_ws/tqq
colcon build --packages-select franka_vr_teleop --symlink-install
source install/setup.bash
```

Test with fake hardware first:

```bash
ros2 launch franka_vr_teleop quest3_franka_teleop.launch.py use_fake_hardware:=true
```

Run on the real FR3:

```bash
ros2 launch franka_vr_teleop quest3_franka_teleop.launch.py robot_ip:=192.168.22.212
```

Useful tuning files:

```text
config/quest3_axis_mapping.yaml      controller axes -> FR3 base axes
config/quest3_teleop.yaml            default direct Cartesian velocity gains
```

Hold the configured grip button to enable teleop. Trigger is mapped to the gripper action.

Runtime heading calibration:

1. Stand where you want to operate the robot.
2. Face the direction that should correspond to Franka base forward.
3. Press the right controller `A` button once.
4. Hold grip and move the controller. The hand motion direction is rotated by the calibrated headset heading.

Press the right controller `B` button to reset the heading calibration.

Minimal Unity sender sketch:

```csharp
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class Quest3UdpSender : MonoBehaviour
{
    public string targetIp = "192.168.22.10";
    public int targetPort = 5055;
    private UdpClient client;

    void Start() { client = new UdpClient(); }
    void Update()
    {
        var p = transform.position;
        var q = transform.rotation;
        string json = $"{{\"pose\":{{\"position\":[{p.x:F6},{p.y:F6},{p.z:F6}],\"orientation\":[{q.x:F6},{q.y:F6},{q.z:F6},{q.w:F6}]}},\"grip_pressed\":true,\"trigger_pressed\":false}}";
        byte[] data = Encoding.UTF8.GetBytes(json);
        client.Send(data, data.Length, targetIp, targetPort);
    }
}
```
