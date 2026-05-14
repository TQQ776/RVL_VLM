from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'anygrasp_roi'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'scripts'), glob(os.path.join('scripts', '*.sh'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tqq',
    maintainer_email='tqq@todo.todo',
    description='Interactive ROI based AnyGrasp grasping with RealSense RGBD and MoveIt.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'roi_anygrasp_controller = anygrasp_roi.roi_anygrasp_controller:main',
        ],
    },
)
