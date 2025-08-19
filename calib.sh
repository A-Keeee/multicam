cd ~/odin_ros_driver

gnome-terminal -- bash -i -c "source install/setup.bash && ros2 launch odin_ros_driver cam_odin.launch.py; exec bash"

sleep 3

cd ~/track_ros2

gnome-terminal -- bash -i -c "source install/setup.bash && ros2 launch oak_d_camera camera.launch.py; exec bash"

sleep 5


# cd ~/multicam

# gnome-terminal -- bash -i -c "source install/setup.bash && ros2 launch multicam_calibration calibration.launch.py; exec bash"

