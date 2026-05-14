export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda3 | grep -v conda | tr '\n' ':' | sed 's/:$//')
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE
source /opt/ros/humble/setup.bash
source ~/TQQ_ws/franka/install/setup.bash    # underlay：franka 官方包
source ~/TQQ_ws/tqq/install/setup.bash       # overlay：你二开的包  ← 新增
