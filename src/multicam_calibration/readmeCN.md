# multicam_calibration - 相机的外参和内参标定

本仓库包含一个用于多相机系统的内参与外参标定的 ROS2 包，使用的是 AprilGrid（不支持棋盘格）。

支持鱼眼（`equidistant`）和标准的 `radtan` 畸变模型。

在 Ubuntu 20.04 + ROS2 Foxy 下已测试。

## 安装
你需要安装 libceres：
```
	sudo apt install libceres-dev
```

下载本包：
``` 
mkdir -p my_ros_ws/src
cd my_ros_ws/src
git clone --branch ros2	https://github.com/KumarRobotics/multicam_calibration.git
```

你还需要 ROS2 版本的 apriltag ROS2 包（apriltag wrapper）：https://github.com/versatran01/apriltag/tree/ros2 。按该链接的说明进行安装，通常像这样：
```
git clone --branch ros2 https://github.com/berndpfrommer/apriltag.git
cd ..
wstool init src src/apriltag/apriltag_umich/apriltag_umich.rosinstall 
```

在源码构建前请先 source 你的 ROS2 环境：
```
cd my_rows_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

构建后覆盖环境：
```
source install/setup.bash
```

## 使用方法

首先尽量给出一个较好的初始估计并编辑它。示例文件位于 `calib/example/example_camera-initial.yaml`：
```
	cam0:
	  camera_model: pinhole
	  intrinsics: [605.054, 604.66, 641.791, 508.728]
	  distortion_model: equidistant
	  distortion_coeffs: [-0.0146915, 0.000370396, -0.00425216, 0.0015107]
	  resolution: [1280, 1024]
	  rostopic: /rig/left/image_mono
	cam1:
	  T_cn_cnm1:
	  - [ 0.99999965648, -0.00013331925,  0.00081808159, -0.19946344647]
	  - [ 0.00013388601,  0.99999975107, -0.00069277676, -0.00005674605]
	  - [-0.00081798903,  0.00069288605,  0.99999942540,  0.00010022941]
	  - [ 0.00000000000,  0.00000000000,  0.00000000000,  1.00000000000]
	  camera_model: pinhole
	  intrinsics: [605.097, 604.321, 698.772, 573.558]
	  distortion_model: equidistant
	  distortion_coeffs: [-0.0146155, -0.00291494, -0.000681981, 0.000221855]
	  resolution: [1280, 1024]
	  rostopic: /rig/right/image_mono
```

调整 `rostopic` 以匹配你的相机话题，保证分辨率、畸变模型等正确。此文件格式与 [Kalibr 的文件格式](https://github.com/ethz-asl/kalibr) 相同。将编辑后的文件放入一个目录（例如 `stereo_cam`），并编辑 `launch` 目录下的 `calibration.launch.py`：
```python
parameters=[{'calib_dir': calib_dir,
             'target_type': 'aprilgrid',
             'tag_rows': 5,
             'tag_cols': 7,
             'tag_size': 0.017,
             'tag_spacing': 0.3,
             'black_border': 2,
             'tag_family': '36h11',
             'corners_in_file': '',
             'use_approximate_sync': False,
             'device_name': device_name}],
```

将 `device_name` 调整为你初始标定文件的名称（例如 `stereo_cam`）。设置 AprilGrid 的行、列和 tag 大小。注意黑边（`black_border`）的大小；如果你的相机图像不是同步的（即各相机消息时间戳不相同），请将 `use_approximate_sync` 设为 true。

完成编辑后，重新运行 colcon 构建（如果需要），然后启动：
```bash
ros2 launch multicam_calibration calibration.launch.py
```

你会看到类似的输出：
```
[calibration_node-1] parsing initial camera calibration file: /mypath/calib/stereo_cam/stereo_cam-initial.yaml
[calibration_node-1] [INFO] [1613243221.909816377] [multicam_calibration]: Found 2 cameras!
[calibration_node-1] [INFO] [1613243221.909937970] [multicam_calibration]: not using approximate sync
[calibration_node-1] [INFO] [1613243221.910176548] [multicam_calibration]: writing extracted corners to file /mypath/calib/stereo_cam/corners.csv
[calibration_node-1] [INFO] [1613243221.924387353] [multicam_calibration]: calibration_node started up!
[calibration_node-1] [INFO] [1613243225.043543976] [multicam_calibration]: frames:   10, total # tags found:  115 67
[calibration_node-1] [INFO] [1613243228.906891399] [multicam_calibration]: frames:   20, total # tags found:  233 131
[calibration_node-1] [INFO] [1613243257.529047985] [multicam_calibration]: frames:   30, total # tags found:  350 198
[calibration_node-1] [INFO] [1613243260.514001182] [multicam_calibration]: frames:   40, total # tags found:  467 266
```

可视化相机调试图像：
```bash
ros2 run rqt_gui rqt_gui
```

查找 `debug_image` 话题，你会看到类似 `images/example_gui.jpg` 中的界面。

然后播放你的标定 bag（或进行实时标定）：

	rosbag play falcam_rig_2018-01-09-14-28-56.bag

你应该能看到 tag 被检测。收集足够的帧（通常约 5000 个 tag 足够），然后开始标定：

	ros2 service call /calibration std_srvs/srv/Trigger

你将得到类似的输出（这是 Ceres 求解的过程）：
```
	Num params: 2476
	Num residuals: 201928
	iter      cost      cost_change  |gradient|   |step|    tr_ratio  tr_radius  ls_iter  iter_time  total_time
	0  4.478809e+03    0.00e+00    5.32e+06   0.00e+00   0.00e+00  1.00e+04        0    2.45e-01    3.10e-01
	1  1.291247e+03    3.19e+03    2.03e+05   1.46e+00   1.55e+00  3.00e+04        1    5.11e-01    8.21e-01
	2  1.288842e+03    2.40e+00    6.22e+03   2.38e-01   1.04e+00  9.00e+04        1    4.56e-01    1.28e+00
	3  1.288794e+03    4.79e-02    3.19e+02   3.57e-02   1.02e+00  2.70e+05        1    4.37e-01    1.71e+00
	4  1.288792e+03    2.27e-03    3.73e+01   7.64e-03   1.01e+00  8.10e+05        1    4.38e-01    2.15e+00
	5  1.288792e+03    2.61e-05    5.09e+00   7.20e-04   1.01e+00  2.43e+06        1    4.38e-01    2.59e+00
	6  1.288792e+03    6.92e-08    5.35e-01   3.46e-05   1.03e+00  7.29e+06        1    4.37e-01    3.03e+00

	...（略）

	[ INFO] [1515674589.074056064]: writing calibration to /home/pfrommer/Documents/foo/src/multicam_calibration/calib/example/example_camera-2018-01-11-07-43-09.yaml
	cam0:
	camera_model: pinhole
	intrinsics: [604.355, 604.153, 642.488, 508.135]
	distortion_model: equidistant
	distortion_coeffs: [-0.014811, -0.00110814, -0.00137418, 0.000474477]
	resolution: [1280, 1024]
	rostopic: /rig/left/image_mono
	cam1:
	T_cn_cnm1:
	- [ 0.99999720028,  0.00030730438,  0.00234627487, -0.19936845450]
	- [-0.00030303357,  0.99999829718, -0.00182038902,  0.00004464487]
	- [-0.00234683029,  0.00181967292,  0.99999559058,  0.00029671670]
	- [ 0.00000000000,  0.00000000000,  0.00000000000,  1.00000000000]
	camera_model: pinhole
	intrinsics: [604.364, 603.62, 698.645, 573.02]
	distortion_model: equidistant
	distortion_coeffs: [-0.0125438, -0.00503567, 0.00031359, 0.000546495]
	resolution: [1280, 1024]
	rostopic: /rig/right/image_mono
	[ INFO] [1515674589.251025662]: ----------------- reprojection errors: ---------------
	[ INFO] [1515674589.251045482]: total error:     0.283519 px
	[ INFO] [1515674589.251053450]: avg error cam 0: 0.28266 px
	[ INFO] [1515674589.251059520]: avg error cam 1: 0.284286 px
	[ INFO] [1515674589.251070091]: max error: 8.84058 px at frame: 110 for cam: 1
	[ INFO] [1515674589.251410620]: -------------- simple homography test ---------
	[ INFO] [1515674589.331235450]: camera: 0 points: 47700 reproj err: 0.440283
	[ INFO] [1515674589.331257726]: camera: 1 points: 53252 reproj err: 0.761365
```

在标定目录下你可以找到标定输出文件：

	ls -1
	stereo_camera-2018-01-11-08-24-22.yaml
	stereo_camera-initial.yaml
	stereo_camera-latest.yaml

## 管理式（分步）标定

有时标定需要按步骤进行，例如先标定每个传感器的内参，然后再标定传感器之间的外参。这在相机间图像不同步时尤其有用。

为此，你可以编写一个小的 Python 程序来按顺序运行多个标定并将上一次的输出作为下一次的初始值。实际上只需修改 `src/example_calib_manager.py` 中的一个段落。示例（按需调整）：

        # first do intrinsics of cam0
        set_p(FIX_INTRINSICS, "cam0", False)
        set_p(FIX_EXTRINSICS, "cam0", True)
        set_p(SET_ACTIVE,     "cam0", True)
        set_p(FIX_INTRINSICS, "cam1", True)
        set_p(FIX_EXTRINSICS, "cam1", True)
        set_p(SET_ACTIVE,     "cam1", False)
        run_cal()
        # then do intrinsics of cam1
        set_p(FIX_INTRINSICS, "cam0", True)
        set_p(FIX_EXTRINSICS, "cam0", True)
        set_p(SET_ACTIVE,     "cam0", False)
        set_p(FIX_INTRINSICS, "cam1", False)
        set_p(FIX_EXTRINSICS, "cam1", True)
        set_p(SET_ACTIVE,     "cam1", True)
        run_cal()
        # now extrinsics between the two
        set_p(FIX_INTRINSICS, "cam0", True)
        set_p(FIX_EXTRINSICS, "cam0", True)
        set_p(SET_ACTIVE,     "cam0", True)
        set_p(FIX_INTRINSICS, "cam1", True)
        set_p(FIX_EXTRINSICS, "cam1", False)
        set_p(SET_ACTIVE,     "cam1", True)
        run_cal()

运行标定管理器：
```bash
ros2 run multicam_calibration example_calib_manager.py -n multicam_calibration
```

触发其执行：
```bash
ros2 service call /run_calibration_manager std_srvs/srv/Trigger
```

## 去畸变（Undistortion）

为方便起见，本仓库包含一个用于对鱼眼（equidistant）相机图像去畸变的节点。运行方法（在调整 launch 文件参数后）：
```
ros2 launch multicam_calibration undistort.launch.py
```

## 单元测试

关于标定代码的单元测试，请参阅 `multicam_calibratin/test/README.md` 页面。注意：这些测试已经被移植，但从未在 ROS2 下运行过，可能存在问题。

