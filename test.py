#!/usr/bin/env python3

# Copyright 2025 Google LLC.
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

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
import yaml
import os
import time

# GPU加速相关
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        CUDA_AVAILABLE = True
        print(f"CUDA设备数量: {cv2.cuda.getCudaEnabledDeviceCount()}")
    else:
        CUDA_AVAILABLE = False
        print("警告: 没有检测到CUDA设备，将使用CPU处理")
except:
    CUDA_AVAILABLE = False
    print("警告: OpenCV未编译CUDA支持，将使用CPU处理")

class ImageStitcherNode(Node):
    def __init__(self):
        super().__init__('image_stitcher_node')
        self.get_logger().info('图像拼接节点已启动...')
        
        # 调试标志
        self.debug_save_images = False
        self.frame_count = 0
        
        # 图像压缩设置
        self.enable_compression = True
        self.compression_scale = 0.25
        self.output_use_compressed = True # 核心开关

        # GPU加速设置
        self.use_gpu = CUDA_AVAILABLE and True
        if self.use_gpu:
            self.get_logger().info('启用GPU加速处理')
            self.cuda_stream = cv2.cuda_Stream()
            self._warmup_gpu()
        else:
            self.get_logger().info('使用CPU处理')
            self.cuda_stream = None

        # --- 1. 加载和解析相机参数 ---
        self.config_file = '/home/ake/multicam/cam_calib.yaml'
        self.cam0_params, self.cam1_params = self._load_camera_params()
        self.bridge = CvBridge()
        
        self.K0 = self.cam0_params['intrinsics']
        self.D0 = self.cam0_params['distortion_coeffs']
        self.w0, self.h0 = self.cam0_params['resolution']
        
        self.K1 = self.cam1_params['intrinsics']
        self.D1 = self.cam1_params['distortion_coeffs']
        self.w1, self.h1 = self.cam1_params['resolution']

        # --- 2. 计算变换关系 ---
        T_01 = self.cam1_params['T_cn_cnm1']
        T_10 = np.linalg.inv(T_01)
        R_10 = T_10[0:3, 0:3]
        
        self.H = self.K0 @ R_10 @ np.linalg.inv(self.K1)
        self.get_logger().info('使用纯旋转单应性（忽略平移）')

        # --- 3. 预计算全分辨率和压缩分辨率下的几何参数 ---
        self.get_logger().info("正在预计算全分辨率几何参数...")
        self.output_size_full, self.warp_matrix_0_full, self.warp_matrix_1_full = self._calculate_output_geometry(self.H, self.w0, self.h0, self.w1, self.h1)
        self._prepare_full_res_maps()
        self.get_logger().info(f"全分辨率拼接尺寸: {self.output_size_full}")
        
        if self.output_use_compressed and self.enable_compression and self.compression_scale < 1.0:
            self.get_logger().info("正在预计算压缩分辨率几何参数...")
            self._prepare_compressed_geometry_and_maps()
            self.get_logger().info(f"压缩分辨率拼接尺寸: {self.output_size_compressed}, 缩放比例: {self.compression_scale}")

        # --- 4. 初始化ROS订阅器和发布器 ---
        self.stitched_image_publisher = self.create_publisher(Image, '/image_stitched', 10)
        sub_cam0 = message_filters.Subscriber(self, Image, self.cam0_params['rostopic'])
        sub_cam1 = message_filters.Subscriber(self, Image, self.cam1_params['rostopic'])
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer([sub_cam0, sub_cam1], queue_size=10, slop=0.1)
        self.time_synchronizer.registerCallback(self.image_callback)
        self.get_logger().info('订阅器已设置，等待图像消息...')

    def _load_camera_params(self):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"配置文件未找到: {self.config_file}")
        with open(self.config_file, 'r') as file:
            config = yaml.safe_load(file)
        cam0_config = config['cam0']
        cam0_params = {
            'intrinsics': np.array([[cam0_config['intrinsics'][0], 0, cam0_config['intrinsics'][2]], [0, cam0_config['intrinsics'][1], cam0_config['intrinsics'][3]], [0, 0, 1]], dtype=np.float32),
            'distortion_coeffs': np.array(cam0_config['distortion_coeffs'], dtype=np.float32),
            'resolution': cam0_config['resolution'],
            'rostopic': cam0_config['rostopic'],
        }
        cam1_config = config['cam1']
        cam1_params = {
            'T_cn_cnm1': np.array(cam1_config['T_cn_cnm1'], dtype=np.float32),
            'intrinsics': np.array([[cam1_config['intrinsics'][0], 0, cam1_config['intrinsics'][2]], [0, cam1_config['intrinsics'][1], cam1_config['intrinsics'][3]], [0, 0, 1]], dtype=np.float32),
            'distortion_coeffs': np.array(cam1_config['distortion_coeffs'], dtype=np.float32),
            'resolution': cam1_config['resolution'],
            'rostopic': cam1_config['rostopic'],
        }
        self.get_logger().info(f'成功从 {self.config_file} 加载相机参数')
        return cam0_params, cam1_params
    
    def _calculate_output_geometry(self, H, w0, h0, w1, h1):
        corners0 = np.float32([[0, 0], [0, h0], [w0, h0], [w0, 0]]).reshape(-1, 1, 2)
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners1_transformed = cv2.perspectiveTransform(corners1, H)
        all_corners = np.concatenate((corners0, corners1_transformed), axis=0)
        xmin, ymin = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        xmax, ymax = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        translation_matrix = np.array([[1.0, 0.0, -xmin], [0.0, 1.0, -ymin], [0.0, 0.0, 1.0]], dtype=np.float32)
        warp_matrix_0 = translation_matrix.copy()
        warp_matrix_1 = translation_matrix @ H
        output_size = (xmax - xmin, ymax - ymin)
        return output_size, warp_matrix_0, warp_matrix_1

    def _prepare_full_res_maps(self):
        self.map1_0_full, self.map2_0_full = cv2.initUndistortRectifyMap(self.K0, self.D0, None, self.K0, (self.w0, self.h0), cv2.CV_32FC1)
        self.map1_1_full, self.map2_1_full = cv2.fisheye.initUndistortRectifyMap(self.K1, self.D1, np.eye(3), self.K1, (self.w1, self.h1), cv2.CV_32FC1)
        if self.use_gpu:
            self.gpu_map1_0_full, self.gpu_map2_0_full = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
            self.gpu_map1_1_full, self.gpu_map2_1_full = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
            self.gpu_map1_0_full.upload(self.map1_0_full)
            self.gpu_map2_0_full.upload(self.map2_0_full)
            self.gpu_map1_1_full.upload(self.map1_1_full)
            self.gpu_map2_1_full.upload(self.map2_1_full)
        self.get_logger().info('全分辨率去畸变映射表已生成')

    def _prepare_compressed_geometry_and_maps(self):
        s = self.compression_scale
        self.w0_c, self.h0_c = int(self.w0 * s), int(self.h0 * s)
        self.w1_c, self.h1_c = int(self.w1 * s), int(self.h1 * s)
        
        K0_scaled = self.K0.copy()
        K0_scaled[0:2, :] *= s
        K1_scaled = self.K1.copy()
        K1_scaled[0:2, :] *= s
        
        S = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float32)
        S_inv = np.linalg.inv(S)
        H_compressed = S @ self.H @ S_inv
        
        self.output_size_compressed, self.warp_matrix_0_compressed, self.warp_matrix_1_compressed = \
            self._calculate_output_geometry(H_compressed, self.w0_c, self.h0_c, self.w1_c, self.h1_c)

        self.map1_0_compressed, self.map2_0_compressed = cv2.initUndistortRectifyMap(self.K0, self.D0, None, K0_scaled, (self.w0_c, self.h0_c), cv2.CV_32FC1)
        self.map1_1_compressed, self.map2_1_compressed = cv2.fisheye.initUndistortRectifyMap(self.K1, self.D1, np.eye(3), K1_scaled, (self.w1_c, self.h1_c), cv2.CV_32FC1)
        
        if self.use_gpu:
            self.gpu_map1_0_compressed, self.gpu_map2_0_compressed = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
            self.gpu_map1_1_compressed, self.gpu_map2_1_compressed = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
            self.gpu_map1_0_compressed.upload(self.map1_0_compressed)
            self.gpu_map2_0_compressed.upload(self.map2_0_compressed)
            self.gpu_map1_1_compressed.upload(self.map1_1_compressed)
            self.gpu_map2_1_compressed.upload(self.map2_1_compressed)
        self.get_logger().info('压缩分辨率去畸变映射表已生成')
    
    def _blend_with_masks(self, warped_img0, warped_img1, mask0, mask1):
        overlap_mask = cv2.bitwise_and(mask0, mask1)
        stitched_image = np.zeros_like(warped_img1)
        stitched_image[mask1 > 0] = warped_img1[mask1 > 0]
        cam0_only_mask = cv2.bitwise_and(mask0, cv2.bitwise_not(mask1))
        stitched_image[cam0_only_mask > 0] = warped_img0[cam0_only_mask > 0]
        if np.any(overlap_mask > 0):
            alpha = 0.5
            overlap_region = (warped_img0[overlap_mask > 0] * alpha + warped_img1[overlap_mask > 0] * (1 - alpha)).astype(np.uint8)
            stitched_image[overlap_mask > 0] = overlap_region
        return stitched_image

    def _blend_images_cpu(self, warped_img0, warped_img1):
        gray0 = cv2.cvtColor(warped_img0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY)
        _, mask0 = cv2.threshold(gray0, 1, 255, cv2.THRESH_BINARY)
        _, mask1 = cv2.threshold(gray1, 1, 255, cv2.THRESH_BINARY)
        return self._blend_with_masks(warped_img0, warped_img1, mask0, mask1)

    def _warmup_gpu(self):
        try:
            test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            gpu_test = cv2.cuda_GpuMat()
            gpu_test.upload(test_img)
            cv2.cuda.cvtColor(gpu_test, cv2.COLOR_BGR2GRAY)
            self.cuda_stream.waitForCompletion()
            self.get_logger().info('GPU预热完成')
        except Exception as e:
            self.get_logger().warn(f'GPU预热失败: {e}')
            
    def image_callback(self, ros_img0_msg, ros_img1_msg):
        start_time = time.time()
        try:
            cv_img0 = self.bridge.imgmsg_to_cv2(ros_img0_msg, 'bgr8')
            cv_img1 = self.bridge.imgmsg_to_cv2(ros_img1_msg, 'bgr8')

            # --- 方案A: 快速低分辨率处理路径 ---
            if self.output_use_compressed and self.compression_scale < 1.0:
                if self.use_gpu:
                    gpu_img0, gpu_img1 = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
                    gpu_img0.upload(cv_img0)
                    gpu_img1.upload(cv_img1)

                    # 1.1 GPU去畸变+缩放 (一步完成)
                    # 将全分辨率图像(gpu_img0)和压缩尺寸映射表(gpu_map1_0_compressed)传入
                    # 输出即为小尺寸、已去畸变的图像
                    gpu_undistorted0 = cv2.cuda.remap(gpu_img0, self.gpu_map1_0_compressed, self.gpu_map2_0_compressed, cv2.INTER_LINEAR, stream=self.cuda_stream)
                    gpu_undistorted1 = cv2.cuda.remap(gpu_img1, self.gpu_map1_1_compressed, self.gpu_map2_1_compressed, cv2.INTER_LINEAR, stream=self.cuda_stream)

                    # 1.2 GPU透视变换
                    gpu_warped0 = cv2.cuda.warpPerspective(gpu_undistorted0, self.warp_matrix_0_compressed, self.output_size_compressed, stream=self.cuda_stream)
                    gpu_warped1 = cv2.cuda.warpPerspective(gpu_undistorted1, self.warp_matrix_1_compressed, self.output_size_compressed, stream=self.cuda_stream)
                    self.cuda_stream.waitForCompletion()
                    
                    warped_img0 = gpu_warped0.download()
                    warped_img1 = gpu_warped1.download()
                else: # CPU Path
                    # 1.1 CPU去畸变+缩放 (一步完成)
                    undistorted_img0 = cv2.remap(cv_img0, self.map1_0_compressed, self.map2_0_compressed, cv2.INTER_LINEAR)
                    undistorted_img1 = cv2.remap(cv_img1, self.map1_1_compressed, self.map2_1_compressed, cv2.INTER_LINEAR)
                    
                    # 1.2 CPU透视变换
                    warped_img0 = cv2.warpPerspective(undistorted_img0, self.warp_matrix_0_compressed, self.output_size_compressed)
                    warped_img1 = cv2.warpPerspective(undistorted_img1, self.warp_matrix_1_compressed, self.output_size_compressed)

            # --- 方案B: 慢速高质量处理路径 ---
            else:
                if self.use_gpu:
                    gpu_img0, gpu_img1 = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
                    gpu_img0.upload(cv_img0)
                    gpu_img1.upload(cv_img1)
                    
                    # 2.1 GPU去畸变
                    gpu_undistorted0 = cv2.cuda.remap(gpu_img0, self.gpu_map1_0_full, self.gpu_map2_0_full, cv2.INTER_LINEAR, stream=self.cuda_stream)
                    gpu_undistorted1 = cv2.cuda.remap(gpu_img1, self.gpu_map1_1_full, self.gpu_map2_1_full, cv2.INTER_LINEAR, stream=self.cuda_stream)

                    # 2.2 GPU透视变换
                    gpu_warped0 = cv2.cuda.warpPerspective(gpu_undistorted0, self.warp_matrix_0_full, self.output_size_full, stream=self.cuda_stream)
                    gpu_warped1 = cv2.cuda.warpPerspective(gpu_undistorted1, self.warp_matrix_1_full, self.output_size_full, stream=self.cuda_stream)
                    self.cuda_stream.waitForCompletion()
                    
                    warped_img0 = gpu_warped0.download()
                    warped_img1 = gpu_warped1.download()
                else: # CPU Path
                    # 2.1 CPU去畸变
                    undistorted_img0 = cv2.remap(cv_img0, self.map1_0_full, self.map2_0_full, cv2.INTER_LINEAR)
                    undistorted_img1 = cv2.remap(cv_img1, self.map1_1_full, self.map2_1_full, cv2.INTER_LINEAR)
                    
                    # 2.2 CPU透视变换
                    warped_img0 = cv2.warpPerspective(undistorted_img0, self.warp_matrix_0_full, self.output_size_full)
                    warped_img1 = cv2.warpPerspective(undistorted_img1, self.warp_matrix_1_full, self.output_size_full)
            
            # --- 步骤3: 图像融合 ---
            stitched_image = self._blend_images_cpu(warped_img0, warped_img1)

            # --- 步骤4: 性能统计和发布 ---
            self.frame_count += 1
            total_time = time.time() - start_time
            if self.frame_count % 10 == 0:
                processing_mode = "GPU" if self.use_gpu else "CPU"
                pipeline_mode = "Compressed" if (self.output_use_compressed and self.compression_scale < 1.0) else "Full-Res"
                self.get_logger().info(f'性能统计 - {processing_mode}/{pipeline_mode}, '
                                     f'时间: {total_time:.4f}s, FPS: {1.0/total_time:.1f}, '
                                     f'输出: {stitched_image.shape[1]}x{stitched_image.shape[0]}')

            stitched_msg = self.bridge.cv2_to_imgmsg(stitched_image, 'bgr8')
            stitched_msg.header.stamp = ros_img0_msg.header.stamp
            stitched_msg.header.frame_id = 'stitched_camera_frame'
            self.stitched_image_publisher.publish(stitched_msg)

        except Exception as e:
            import traceback
            self.get_logger().error(f'处理图像时发生错误: {e}\n{traceback.format_exc()}')


def main(args=None):
    rclpy.init(args=args)
    image_stitcher_node = ImageStitcherNode()
    try:
        rclpy.spin(image_stitcher_node)
    except KeyboardInterrupt:
        pass
    finally:
        image_stitcher_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()