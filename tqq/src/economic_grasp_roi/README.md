# economic_grasp_roi

Independent ROS 2 ROI grasping package for EconomicGrasp. It is not connected
to the LLM/MCP stack.

Pipeline:

```text
RealSense RGB-D -> ROI point cloud -> EconomicGrasp -> TF to fr3_link0 -> MoveIt IK -> gripper close
```

The package uses the full EconomicGrasp 6D grasp output. The only extra pose
parameters are the robot-specific transform from the network gripper frame to
`fr3_hand_tcp`.

## Third-party Code

EconomicGrasp has been cloned here:

```bash
/home/tqq/TQQ_ws/third_party/EconomicGrasp
```

The RealSense checkpoint has been downloaded here:

```bash
/home/tqq/TQQ_ws/third_party/EconomicGrasp/checkpoints/economicgrasp_realsense.tar
```

EconomicGrasp is MIT licensed. The local clone is used as an external
third-party dependency; this ROS package keeps robot control in ROS + MoveIt.

## Dependencies

EconomicGrasp requires a heavier CUDA/PyTorch stack than the ROS package itself:

```text
Python 3.10
CUDA 12.x
PyTorch in the same Python that runs ROS 2
MinkowskiEngine
pointnet2
knn_pytorch
open3d/scipy/Pillow/tqdm/pyyaml
```

On this workstation the package is configured for ROS Humble's system Python:

```bash
source /home/tqq/miniconda3/etc/profile.d/conda.sh
conda deactivate

/home/tqq/TQQ_ws/tqq/src/economic_grasp_roi/scripts/install_economic_grasp_deps.sh
```

The current base conda Python is 3.13, so do not run this node from base conda.
The dependency script defaults to `/usr/bin/python3`, installs Python packages
with `--user`, and compiles CUDA extensions with `TORCH_CUDA_ARCH_LIST=8.9`.

Current verified local dependency state:

```text
torch 2.10.0+cu128, CUDA available
MinkowskiEngine 0.5.4
EconomicGrasp RealSense checkpoint epoch 10
```

## Build

```bash
source /opt/ros/humble/setup.bash
cd /home/tqq/TQQ_ws/tqq
colcon build --packages-select economic_grasp_roi --symlink-install
```

## Run

Start RealSense, FR3 MoveIt, and hand-eye TF first. Then:

```bash
source /home/tqq/miniconda3/etc/profile.d/conda.sh
conda deactivate

source /opt/ros/humble/setup.bash
source /home/tqq/TQQ_ws/franka/install/setup.bash
source /home/tqq/TQQ_ws/tqq/install/setup.bash

ros2 launch economic_grasp_roi roi_economic_grasp.launch.py \
  params_file:=/home/tqq/TQQ_ws/tqq/src/economic_grasp_roi/config/roi_economic_grasp.yaml
```

In the camera window, drag an ROI and press Enter or `g`. In the Open3D preview
window, press Enter to execute or `c`/`q`/Esc to cancel.

The node executable is installed with a `/usr/bin/python3` shebang. That matters
because ROS Humble, rclpy, PyTorch, MinkowskiEngine, pointnet2, and knn_pytorch
must all be visible in the same Python environment.

## Important Parameters

Config file:

```bash
/home/tqq/TQQ_ws/tqq/src/economic_grasp_roi/config/roi_economic_grasp.yaml
```

Useful parameters:

```yaml
economic_grasp_repo_dir: /home/tqq/TQQ_ws/third_party/EconomicGrasp
economic_grasp_checkpoint_path: /home/tqq/TQQ_ws/third_party/EconomicGrasp/checkpoints/economicgrasp_realsense.tar
economic_grasp_orientation_mode: economic_grasp
economic_grasp_tcp_rotation_rpy_grasp: [0.0, 1.57079632679, 0.0]
economic_grasp_tcp_offset_xyz_grasp: [0.0, 0.0, 0.0]
max_velocity_scaling: 0.01
max_acceleration_scaling: 0.01
```

If IK fails because the full 6D pose is unreachable, keep the network output
unchanged and tune only `economic_grasp_tcp_rotation_rpy_grasp` /
`economic_grasp_tcp_offset_xyz_grasp` for the physical TCP-frame mapping.
