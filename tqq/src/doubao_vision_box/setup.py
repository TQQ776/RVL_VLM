from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'doubao_vision_box'

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
    description='Independent Doubao vision box annotation node for camera images.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'doubao_vision_box = doubao_vision_box.doubao_vision_box_node:main',
        ],
    },
)
