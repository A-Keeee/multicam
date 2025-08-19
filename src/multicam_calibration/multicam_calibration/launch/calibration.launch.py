# -----------------------------------------------------------------------------
# Copyright 2020 Bernd Pfrommer <bernd.pfrommer@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# launch file for calibration
#

from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration as LaunchConfig
from launch.actions import DeclareLaunchArgument as LaunchArg
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Launch calibration."""
    pkg_dir = get_package_share_directory('multicam_calibration')
    calib_dir = pkg_dir + '/calib/example_camera'
    node_name_arg = LaunchArg('node_name', default_value='multicam_calibration',
                              description='name of calibration node')
    # new launch args
    approx_sync_slop_arg = LaunchArg('approx_sync_slop', default_value='0.5', description='ApproximateTime 同步允许的最大秒差')
    force_publish_debug_arg = LaunchArg('force_publish_debug', default_value='true', description='即使无订阅也发布debug图像')
    node = Node(package='multicam_calibration',
                executable='calibration_node',
                output='screen',
                name=LaunchConfig('node_name'),
#                prefix=['xterm -e gdb -ex run --args'],
                parameters=[
                    {'calib_dir': calib_dir,
                     'target_type': 'aprilgrid',
                     'tag_rows': 8,
                     'tag_cols': 11,
                     'tag_size': 0.04,
                     'tag_spacing': 0.25,
                     'black_border': 2,
                     'tag_family': '36h11',
                     'corners_in_file': '',
                     'corners_out_file': 'corners.csv',
                     'use_approximate_sync': True,
                     'device_name': 'example_camera',
                     # pass through new params
                     'approx_sync_slop': LaunchConfig('approx_sync_slop'),
                     'force_publish_debug': LaunchConfig('force_publish_debug')
                     }],
                remappings=[('~/left_image', '/color/image'),
                            ('~/right_image', '/odin1/image'),
                            ('~/left_camera_info', '/color/camera_info'),
                            ('~/right_camera_info', '/odin1/rgb_camera_info')
                ])
    return LaunchDescription([node_name_arg, approx_sync_slop_arg, force_publish_debug_arg, node])
