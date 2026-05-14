from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'economic_grasp_roi'

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
    description='Interactive ROI based EconomicGrasp 6D grasping with RealSense RGB-D and MoveIt.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'roi_economic_grasp_controller = economic_grasp_roi.roi_economic_grasp_controller:main',
        ],
    },
)
