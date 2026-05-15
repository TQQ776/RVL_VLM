from setuptools import find_packages, setup

package_name = 'audio_dialog'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tqq',
    maintainer_email='tqq@todo.todo',
    description='Reusable audio recording, text popup, and TTS helpers for Qwen-Omni clients.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)
