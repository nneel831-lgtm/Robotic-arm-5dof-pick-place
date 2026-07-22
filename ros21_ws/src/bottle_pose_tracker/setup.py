from setuptools import setup
import os
from glob import glob

package_name = 'bottle_pose_tracker'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robotics-engineer',
    maintainer_email='engineer@example.com',
    description='Real-time YOLOv8 + ByteTrack single-bottle detection with '
                 'robust 3D pose estimation in camera frame using RealSense D435i.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bottle_pose_node = bottle_pose_tracker.bottle_pose_node:main',
        ],
    },
)
