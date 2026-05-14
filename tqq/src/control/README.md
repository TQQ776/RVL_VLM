# control

ROS 2 package that converts YOLO bounding boxes and RealSense aligned depth into a Franka FR3 target pose.

The node:

- subscribes to `/yolo/detections_json`;
- reads the selected bounding-box center pixel;
- samples `/camera/camera/aligned_depth_to_color/image_raw`;
- back-projects the pixel with `/camera/camera/color/camera_info`;
- transforms the 3D point into `fr3_link0` using TF;
- calls MoveIt `/compute_ik` for `fr3_hand_tcp`;
- executes through the MoveGroup action by default.

The node does not close the gripper. It only moves the `fr3_hand_tcp` frame to the detected object point, optionally with a configurable offset.

## Build

```bash
cd ~/TQQ_ws/tqq
colcon build --packages-select control --symlink-install
source install/setup.bash
```

## Run

Start the robot/MoveIt stack, RealSense aligned depth, and YOLO first. The helper launch can start camera and YOLO:

```bash
ros2 launch control object_target_control.launch.py launch_camera:=true launch_yolo:=true
```

If MoveIt is not already running, add:

```bash
ros2 launch control object_target_control.launch.py launch_camera:=true launch_yolo:=true launch_moveit:=true robot_ip:=192.168.22.212
```

The default RealSense config in this package enables depth alignment. It publishes:

```text
/camera/camera/color/image_raw
/camera/camera/color/camera_info
/camera/camera/aligned_depth_to_color/image_raw
```

## Move To The Current Target

The node keeps updating the latest target pose but does not move automatically by default. Trigger motion with:

```bash
ros2 service call /object_target_controller/move_to_target std_srvs/srv/Trigger {}
```

By default the node asks you to choose a class before moving. Start the terminal chooser:

```bash
ros2 run control object_target_cli
```

It prints the detected object names, for example:

```text
I can see: apple x2, orange x1
grab>
```

Type only the object name:

```text
grab> apple
```

The node waits for the next matching detection and starts one move automatically. Type `clear` to clear the selected target, or `q` to quit the terminal chooser.

For unattended testing only:

```bash
ros2 launch control object_target_control.launch.py auto_execute:=true
```

## Move Home

Return the arm to the saved home posture:

```bash
ros2 launch control move_home.launch.py
```

Home joint angles are stored in `config/move_home.yaml`. The default pose is:

- fr3_joint1: 74 deg
- fr3_joint2: -3 deg
- fr3_joint3: -7 deg
- fr3_joint4: -115 deg
- fr3_joint5: -1 deg
- fr3_joint6: 110 deg
- fr3_joint7: 22 deg

## Useful Parameters

```bash
ros2 launch control object_target_control.launch.py target_class_name:=bottle
ros2 launch control object_target_control.launch.py depth_topic:=/camera/aligned_depth_to_color/image_raw camera_info_topic:=/camera/color/camera_info
```

Important parameters in `config/object_target_control.yaml`:

- `target_offset_xyz_base`: offset added to the detected object point in `fr3_link0`.
- `use_latest_tf`: true uses the latest TF transform; false uses the depth image timestamp.
- `orientation_mode`: `current` keeps the current gripper orientation; `fixed` uses `fixed_orientation_xyzw`.
- `execution_mode`: `move_group`, `trajectory`, or `ik_only`.
- `plan_only`: when true, MoveIt plans but does not execute.
- `ask_for_target`: when true, wait for a target name on `/object_target_controller/target_class_name`.
- `execute_on_target_command`: when true, move once after the requested class is detected.
- `auto_execute`: when true, motion starts from detections without a service call.

Use `ik_only` or `plan_only` first when tuning TF, depth alignment, and offsets around real hardware.
