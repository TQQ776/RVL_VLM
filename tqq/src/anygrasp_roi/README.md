# anygrasp_roi

Independent ROS 2 ROI grasping package for AnyGrasp. It is not connected to the
LLM/MCP stack.

Pipeline:

```text
RealSense RGB-D -> ROI point cloud -> AnyGrasp -> TF to fr3_link0 -> MoveIt IK -> gripper close
```

## AnyGrasp SDK setup

The package expects the SDK at:

```bash
/home/tqq/TQQ_ws/third_party/anygrasp_sdk
```

The local SDK binary aliases and OpenSSL 1.1 compatibility path have already
been prepared on this machine:

```bash
/home/tqq/TQQ_ws/third_party/anygrasp_sdk/grasp_detection/gsnet.so
/home/tqq/TQQ_ws/third_party/anygrasp_sdk/grasp_detection/lib_cxx.so
/home/tqq/TQQ_ws/third_party/anygrasp_sdk/lib/libcrypto.so.1.1
/home/tqq/TQQ_ws/third_party/anygrasp_sdk/lib/libssl.so.1.1
```

AnyGrasp is licensed by the upstream project. Get the machine feature id with:

```bash
PATH=/home/tqq/TQQ_ws/third_party/anygrasp_sdk/bin:$PATH \
LD_LIBRARY_PATH=/home/tqq/TQQ_ws/third_party/anygrasp_sdk/lib:$LD_LIBRARY_PATH \
/home/tqq/TQQ_ws/third_party/anygrasp_sdk/license_registration/license_checker -f
```

Current machine feature id:

```text
8956876413985820177
```

Submit that feature id to the official AnyGrasp license form:

```text
https://docs.google.com/forms/d/e/1FAIpQLSf_l-onxoDoBV63PMNkr_Q4pp8LJ_muPiIJgKCcRN9Qv-xgeg/viewform?usp=send_form
```

The form requires your name, affiliation, advisor/mentor, non-commercial and
non-distribution agreement choices, the machine feature id, and your personal
signature. Google may require login or captcha, so this step must be completed
manually by the account owner.

After receiving the license zip and `checkpoint_detection.tar`, install them:

```bash
/home/tqq/TQQ_ws/tqq/src/anygrasp_roi/scripts/install_anygrasp_assets.sh \
  --license /path/to/anygrasp_license.zip \
  --checkpoint /path/to/checkpoint_detection.tar
```

The script installs the files to:

```bash
/home/tqq/TQQ_ws/third_party/anygrasp_sdk/grasp_detection/license
/home/tqq/TQQ_ws/third_party/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar
```

The checkpoint path can be changed in:

```bash
/home/tqq/TQQ_ws/tqq/src/anygrasp_roi/config/roi_anygrasp.yaml
```

Check the SDK import after installing the license and checkpoint:

```bash
LD_LIBRARY_PATH=/home/tqq/TQQ_ws/third_party/anygrasp_sdk/lib:$LD_LIBRARY_PATH \
/usr/bin/python3 - <<'PY'
import sys
root = '/home/tqq/TQQ_ws/third_party/anygrasp_sdk'
det = root + '/grasp_detection'
for p in [root, det, det + '/gsnet_versions', root + '/pointnet2']:
    sys.path.insert(0, p)
from gsnet import AnyGrasp
print('AnyGrasp import ok')
PY
```

## Build

```bash
source /opt/ros/humble/setup.bash
cd /home/tqq/TQQ_ws/tqq
colcon build --packages-select anygrasp_roi --symlink-install
```

## Run

Start RealSense, FR3 MoveIt, and hand-eye TF first. Then:

```bash
source /opt/ros/humble/setup.bash
source /home/tqq/TQQ_ws/franka/install/setup.bash
source /home/tqq/TQQ_ws/tqq/install/setup.bash

ros2 launch anygrasp_roi roi_anygrasp.launch.py \
  params_file:=/home/tqq/TQQ_ws/tqq/src/anygrasp_roi/config/roi_anygrasp.yaml
```

In the camera window, drag an ROI and press Enter or `g`. In the Open3D preview
window, press Enter to execute or `c`/`q`/Esc to cancel.
