from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tqq',
    maintainer_email='tqq@todo.todo',
    description='Move a Franka FR3 end effector to image detections using RealSense aligned depth and MoveIt IK.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_target_controller = control.object_target_controller:main',
            'graspnet_target_controller = control.graspnet_target_controller:main',
            'object_target_cli = control.object_target_cli:main',
            'move_home = control.move_home:main',
        ],
    },
)
