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
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
import yaml
import os
import time
import traceback

class Stitcher:
    """
    一个封装了图像拼接核心逻辑的类。
    负责处理所有与图像计算相关的任务，与ROS解耦。
    """
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.prev_affine = None
        self.prev_homo = None
        self.prev_dyn_w0 = None
        self.prev_dyn_w1 = None
        self.crop_bounds = None
        
        # --- 1. 加载和解析相机参数 ---
        self.cam0_params, self.cam1_params = self._load_camera_params(config['config_file_path'])
        
        self.K0, self.D0, (self.w0, self.h0) = self.cam0_params['intrinsics'], self.cam0_params['distortion_coeffs'], self.cam0_params['resolution']
        self.K1, self.D1, (self.w1, self.h1) = self.cam1_params['intrinsics'], self.cam1_params['distortion_coeffs'], self.cam1_params['resolution']

        # --- 2. 进行预计算 ---
        self._prepare_for_reprojection()

        # --- 3. 初始化特征检测器和多频段融合器 ---
        self.orb = cv2.ORB_create(nfeatures=self.config['refine_orb_nfeatures'], fastThreshold=self.config['refine_orb_fast_threshold'])
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 仅在需要时创建融合器
        if self.config['use_multiband_blender']:
            # 修正：移除了 'try_use_gpu' 参数以兼容更多OpenCV版本
            self.blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_MULTI_BAND)
            self.logger.info("多频段融合已启用 (CPU密集型)。")
        else:
            self.blender = None
            self.logger.info("快速线性融合已启用。")

    def _load_camera_params(self, config_file):
        """从YAML文件中加载并解析两个相机的参数。"""
        if not os.path.exists(config_file):
            self.logger.error(f"配置文件未找到: {config_file}")
            raise FileNotFoundError(f"配置文件未找到: {config_file}")
        with open(config_file, 'r') as file:
            config_data = yaml.safe_load(file)
        
        def parse_cam_config(cam_config):
            intrinsics_flat = cam_config['intrinsics']
            distortion_model = cam_config.get('distortion_model', 'plumb_bob')
            self.logger.info(f"相机 {cam_config['rostopic']} 使用 '{distortion_model}' 畸变模型。")
            return {
                'intrinsics': np.array([[intrinsics_flat[0], 0, intrinsics_flat[2]], [0, intrinsics_flat[1], intrinsics_flat[3]], [0, 0, 1]], dtype=np.float32),
                'distortion_coeffs': np.array(cam_config['distortion_coeffs'], dtype=np.float32),
                'distortion_model': distortion_model,
                'resolution': cam_config['resolution'],
                'rostopic': cam_config['rostopic'],
                'T_cn_cnm1': np.array(cam_config.get('T_cn_cnm1', np.eye(4)), dtype=np.float32)
            }

        cam0_params = parse_cam_config(config_data['cam0'])
        cam1_params = parse_cam_config(config_data['cam1'])
        
        self.logger.info(f'成功从 {config_file} 加载相机参数')
        return cam0_params, cam1_params

    def _prepare_for_reprojection(self):
        """为三维重投影模式准备所有计算，主要是生成映射表(LUT)。"""
        self.logger.info("正在为 'reprojection' 模式进行矢量化预计算...")
        start_time = time.time()

        self.newK0, self.ud_map1_0, self.ud_map2_0 = self._make_undistort_maps(self.K0, self.D0, (self.w0, self.h0), self.cam0_params['distortion_model'])
        self.newK1, self.ud_map1_1, self.ud_map2_1 = self._make_undistort_maps(self.K1, self.D1, (self.w1, self.h1), self.cam1_params['distortion_model'])
        
        scale = self.config['compression_scale']
        self.pano_h = int(self.h0 * scale)
        self.pano_w = int(self.w0 * scale * self.config['pano_width_multiplier'])
        
        K0_s = self.newK0.copy(); K0_s[:2, :] *= scale
        K1_s = self.newK1.copy(); K1_s[:2, :] *= scale
        
        T_01 = self.cam1_params['T_cn_cnm1']
        R_01 = T_01[0:3, 0:3].copy()
        t_01 = T_01[0:3, 3] if not self.config['ignore_translation'] else np.zeros(3, dtype=np.float32)
        
        u_pano, v_pano = np.meshgrid(np.arange(self.pano_w), np.arange(self.pano_h))
        f = K0_s[0, 0]
        cx_pano = self.pano_w / 2.0
        cy_pano = self.pano_h / 2.0
        theta = (u_pano - cx_pano) / f
        h_coord = (v_pano - cy_pano)
        p_cam0 = np.stack([np.sin(theta), h_coord / f, np.cos(theta)], axis=-1)
        norm = np.linalg.norm(p_cam0, axis=-1, keepdims=True)
        p_cam0 /= np.clip(norm, 1e-8, None)
        
        p_cam1 = (p_cam0 @ R_01.T) + t_01
        
        def project_unit_dirs(p_dirs, K):
            px, py, pz = p_dirs[..., 0], p_dirs[..., 1], p_dirs[..., 2]
            valid = pz > 1e-6
            u = np.full_like(px, -1, dtype=np.float32)
            v = np.full_like(py, -1, dtype=np.float32)
            u[valid] = K[0, 0] * (px[valid] / pz[valid]) + K[0, 2]
            v[valid] = K[1, 1] * (py[valid] / pz[valid]) + K[1, 2]
            return u, v
        
        self.map_x0, self.map_y0 = project_unit_dirs(p_cam0, K0_s)
        self.map_x1, self.map_y1 = project_unit_dirs(p_cam1, K1_s)

        if self.config['auto_crop']:
            self._compute_crop_bounds()

        self.logger.info(f"重投影预计算完成，耗时: {time.time() - start_time:.2f} 秒")
        self.logger.info(f"全景图输出尺寸: {self.pano_w} x {self.pano_h}")

    def _make_undistort_maps(self, K, D, wh, model):
        """生成去畸变映射与新的无畸变内参矩阵。"""
        w, h = wh
        if model in ['fisheye', 'equidistant']:
            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D[:4], (w, h), np.eye(3), balance=1.0, new_size=(w, h))
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D[:4], np.eye(3), newK, (w, h), cv2.CV_32FC1)
        else:
            newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0.0, (w, h))
            map1, map2 = cv2.initUndistortRectifyMap(K, D, None, newK, (w, h), cv2.CV_32FC1)
        return newK.astype(np.float32), map1, map2

    def _compute_crop_bounds(self):
        """依据有效映射区域自动计算裁剪范围。"""
        valid_mask = ((self.map_x0 > -1) & (self.map_y0 > -1)) | ((self.map_x1 > -1) & (self.map_y1 > -1))
        if not np.any(valid_mask):
            self.logger.warn('自动裁剪: 未找到有效像素，跳过。')
            return
        rows, cols = np.where(valid_mask)
        r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
        margin = self.config['crop_margin']
        r0 = max(0, r0 - margin)
        c0 = max(0, c0 - margin)
        r1 = min(self.pano_h - 1, r1 + margin)
        c1 = min(self.pano_w - 1, c1 + margin)
        self.crop_bounds = (r0, r1, c0, c1)
        self.logger.info(f'自动裁剪范围: rows {r0}:{r1}, cols {c0}:{c1}')

    def process_frames(self, img0, img1, frame_idx):
        """
        核心处理函数：接收原始图像，返回拼接后的图像。
        """
        # --- 1. 图像预处理：去畸变和缩放 ---
        und0 = cv2.remap(img0, self.ud_map1_0, self.ud_map2_0, cv2.INTER_LINEAR)
        und1 = cv2.remap(img1, self.ud_map1_1, self.ud_map2_1, cv2.INTER_LINEAR)
        
        scale = self.config['compression_scale']
        und0_s = cv2.resize(und0, (int(self.w0 * scale), int(self.h0 * scale)), interpolation=cv2.INTER_AREA)
        und1_s = cv2.resize(und1, (int(self.w1 * scale), int(self.h1 * scale)), interpolation=cv2.INTER_AREA)

        # --- 2. 柱面重投影 ---
        pano0 = cv2.remap(und0_s, self.map_x0, self.map_y0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        pano1 = cv2.remap(und1_s, self.map_x1, self.map_y1, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # --- 3. 动态细化 (可选) ---
        if self.config['enable_refine']:
            pano1_warped = self._refine_overlap(pano0, pano1, frame_idx)
        else:
            pano1_warped = pano1

        # --- 4. 曝光补偿 (可选) ---
        if self.config['enable_exposure_compensation']:
            pano0, pano1_warped = self._compensate_exposure(pano0, pano1_warped)

        # --- 5. 图像融合 ---
        stitched_image = self._blend_images(pano0, pano1_warped)

        # --- 6. 后处理：裁剪 ---
        if self.crop_bounds:
            r0, r1, c0, c1 = self.crop_bounds
            stitched_image = stitched_image[r0:r1+1, c0:c1+1]
            
        return stitched_image

    def _refine_overlap(self, pano0, pano1, frame_idx):
        """特征匹配细化 + 时序平滑。"""
        # 检查是否需要重新计算变换
        recalc = (frame_idx % self.config['refine_update_interval'] == 0) or (self.prev_affine is None and self.prev_homo is None)
        if not recalc:
            return self._apply_cached_transform(pano1)

        try:
            gray0 = cv2.cvtColor(pano0, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(pano1, cv2.COLOR_BGR2GRAY)
            mask0 = (gray0 > 15).astype(np.uint8) # 忽略纯黑区域
            mask1 = (gray1 > 15).astype(np.uint8)
            overlap = cv2.bitwise_and(mask0, mask1)

            if np.sum(overlap) < 5000: # 重叠区域太小
                return self._apply_cached_transform(pano1)

            kps0, des0 = self.orb.detectAndCompute(gray0, overlap)
            kps1, des1 = self.orb.detectAndCompute(gray1, overlap)

            if des0 is None or des1 is None or len(kps0) < self.config['refine_min_matches'] or len(kps1) < self.config['refine_min_matches']:
                return self._apply_cached_transform(pano1)

            matches = self.bf.match(des0, des1)
            matches = sorted(matches, key=lambda m: m.distance)
            good_matches = matches[:max(self.config['refine_min_matches'], int(len(matches) * 0.7))]
            
            if len(good_matches) < self.config['refine_min_matches']:
                return self._apply_cached_transform(pano1)

            pts0 = np.float32([kps0[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            pts1 = np.float32([kps1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            if self.config['use_affine_refine']:
                M, _ = cv2.estimateAffinePartial2D(pts1, pts0, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if M is None: return self._apply_cached_transform(pano1)
                if self.prev_affine is not None:
                    M = self.config['refine_smooth_alpha'] * M + (1 - self.config['refine_smooth_alpha']) * self.prev_affine
                self.prev_affine = M
                self.prev_homo = None
                return cv2.warpAffine(pano1, M, (pano1.shape[1], pano1.shape[0]), flags=cv2.INTER_LINEAR)
            else:
                H, _ = cv2.findHomography(pts1, pts0, cv2.RANSAC, 5.0)
                if H is None: return self._apply_cached_transform(pano1)
                if self.prev_homo is not None:
                    H = self.config['refine_smooth_alpha'] * H + (1 - self.config['refine_smooth_alpha']) * self.prev_homo
                    if abs(H[2, 2]) > 1e-8: H /= H[2, 2]
                self.prev_homo = H
                self.prev_affine = None
                return cv2.warpPerspective(pano1, H, (pano1.shape[1], pano1.shape[0]), flags=cv2.INTER_LINEAR)

        except Exception as e:
            self.logger.warn(f"细化过程中出现异常: {e}")
            return self._apply_cached_transform(pano1)

    def _apply_cached_transform(self, img):
        """复用上一帧的有效变换。"""
        if self.prev_affine is not None:
            return cv2.warpAffine(img, self.prev_affine, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        if self.prev_homo is not None:
            return cv2.warpPerspective(img, self.prev_homo, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        return img

    def _compensate_exposure(self, pano0, pano1_warped):
        """基于重叠区域的亮度差异进行曝光补偿。"""
        try:
            gray0 = cv2.cvtColor(pano0, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(pano1_warped, cv2.COLOR_BGR2GRAY)
            mask0 = (gray0 > 15).astype(np.uint8)
            mask1 = (gray1 > 15).astype(np.uint8)
            overlap = cv2.bitwise_and(mask0, mask1)

            if np.sum(overlap) < 1000:
                return pano0, pano1_warped

            mean0 = cv2.mean(gray0, mask=overlap)[0]
            mean1 = cv2.mean(gray1, mask=overlap)[0]

            if mean1 < 1e-6: return pano0, pano1_warped
            
            gain = mean0 / mean1
            
            # 对增益进行平滑，防止闪烁
            if hasattr(self, 'prev_gain'):
                alpha = self.config['exposure_smooth_alpha']
                gain = alpha * gain + (1 - alpha) * self.prev_gain
            self.prev_gain = gain
            
            pano1_compensated = np.clip(pano1_warped.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            return pano0, pano1_compensated
        except Exception as e:
            self.logger.warn(f"曝光补偿失败: {e}")
            return pano0, pano1_warped

    def _blend_images(self, pano0, pano1_warped):
        """使用多频段融合或线性融合来拼接图像。"""
        mask0 = (cv2.cvtColor(pano0, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
        mask1 = (cv2.cvtColor(pano1_warped, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255

        if self.blender:
            # 使用多频段融合
            # 修正：根据OpenCV 4.x的API，prepare函数需要一个定义最终画布大小的元组
            dst_roi = (0, 0, pano0.shape[1], pano0.shape[0])
            self.blender.prepare(dst_roi)
            
            # feed函数将图像放置在画布的(0,0)位置
            self.blender.feed(pano0.astype(np.int16), mask0, (0,0))
            self.blender.feed(pano1_warped.astype(np.int16), mask1, (0,0))
            
            result, _ = self.blender.blend(None, None)
            return result.astype(np.uint8)
        else:
            # 使用优化的线性融合
            overlap_mask = (mask0 > 0) & (mask1 > 0)
            
            # 基础图像，保留所有非黑像素
            stitched_image = np.maximum(pano0, pano1_warped)
            
            if np.any(overlap_mask):
                # 仅在重叠区进行线性融合
                dist0 = cv2.distanceTransform(mask0, cv2.DIST_L2, 5)
                dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5)
                total_dist = dist0 + dist1
                total_dist[total_dist == 0] = 1.0 # 避免除以零
                
                weight0 = dist0 / total_dist
                weight1 = dist1 / total_dist
                
                w0 = cv2.merge([weight0, weight0, weight0])
                w1 = cv2.merge([weight1, weight1, weight1])

                blended_overlap = pano0.astype(np.float32) * w0 + pano1_warped.astype(np.float32) * w1
                stitched_image[overlap_mask] = np.clip(blended_overlap[overlap_mask], 0, 255).astype(np.uint8)

            return stitched_image


class ImageStitcherNode(Node):
    """
    一个ROS 2节点，用于实时拼接来自两个摄像头的图像。
    采用解耦的Stitcher类实现核心功能。
    """
    def __init__(self):
        super().__init__('image_stitcher_node')
        self.get_logger().info('优化版图像拼接节点已启动...')

        # --- 1. 声明并获取所有ROS 2参数 ---
        self.declare_parameters(
            namespace='',
            parameters=[
                ('config_file_path', '/home/ake/multicam/cam_calib.yaml'),
                # 性能优化：降低默认压缩比例
                ('compression_scale', 0.6),
                ('pano_width_multiplier', 1.5),
                ('ignore_translation', False),
                ('auto_crop', True),
                ('crop_margin', 8),
                ('enable_refine', True),
                ('refine_min_matches', 30),
                ('use_affine_refine', True),
                ('refine_smooth_alpha', 0.1),
                # 性能优化：降低动态细化频率
                ('refine_update_interval', 10),
                # 性能优化：减少特征点数量
                ('refine_orb_nfeatures', 1000),
                ('refine_orb_fast_threshold', 10),
                ('enable_exposure_compensation', True),
                ('exposure_smooth_alpha', 0.05),
                # 性能优化：默认禁用CPU密集型的多频段融合
                ('use_multiband_blender', False),
                ('use_gpu', False), # 强制不使用GPU
            ]
        )
        
        # 将所有参数读入一个字典
        self.params = {
            param.name: self.get_parameter(param.name).get_parameter_value()
            for param in self._parameters.values()
        }
        
        # 将参数值从ParameterValue对象中提取出来
        config = {
            'config_file_path': self.params['config_file_path'].string_value,
            'compression_scale': self.params['compression_scale'].double_value,
            'pano_width_multiplier': self.params['pano_width_multiplier'].double_value,
            'ignore_translation': self.params['ignore_translation'].bool_value,
            'auto_crop': self.params['auto_crop'].bool_value,
            'crop_margin': self.params['crop_margin'].integer_value,
            'enable_refine': self.params['enable_refine'].bool_value,
            'refine_min_matches': self.params['refine_min_matches'].integer_value,
            'use_affine_refine': self.params['use_affine_refine'].bool_value,
            'refine_smooth_alpha': self.params['refine_smooth_alpha'].double_value,
            'refine_update_interval': self.params['refine_update_interval'].integer_value,
            'refine_orb_nfeatures': self.params['refine_orb_nfeatures'].integer_value,
            'refine_orb_fast_threshold': self.params['refine_orb_fast_threshold'].integer_value,
            'enable_exposure_compensation': self.params['enable_exposure_compensation'].bool_value,
            'exposure_smooth_alpha': self.params['exposure_smooth_alpha'].double_value,
            'use_multiband_blender': self.params['use_multiband_blender'].bool_value,
            'use_gpu': self.params['use_gpu'].bool_value,
        }

        self.get_logger().info(f"配置加载完成 (性能优先): {config}")
        if config['use_gpu']:
            self.get_logger().info("CUDA GPU 加速已启用。")
        else:
            self.get_logger().warn("CUDA GPU 加速未启用。")

        self.frame_count = 0
        self.global_frame_idx = 0
        self.bridge = CvBridge()

        # --- 2. 实例化Stitcher核心类 ---
        try:
            self.stitcher = Stitcher(config, self.get_logger())
        except Exception as e:
            self.get_logger().fatal(f"Stitcher初始化失败: {e}\n{traceback.format_exc()}")
            self.destroy_node()
            return

        # --- 3. 初始化ROS订阅器和发布器 ---
        self.stitched_image_publisher = self.create_publisher(Image, '/image_stitched', 10)
        # 新增：创建压缩图像的发布者
        self.compressed_image_publisher = self.create_publisher(CompressedImage, '/image_stitched/compressed', 10)
        
        sub_cam0 = message_filters.Subscriber(self, Image, self.stitcher.cam0_params['rostopic'])
        sub_cam1 = message_filters.Subscriber(self, Image, self.stitcher.cam1_params['rostopic'])
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer([sub_cam0, sub_cam1], queue_size=10, slop=0.1)
        self.time_synchronizer.registerCallback(self.image_callback)
        self.get_logger().info('订阅器已设置，等待图像消息...')

    def image_callback(self, ros_img0_msg, ros_img1_msg):
        """ROS消息回调函数，调用Stitcher进行处理并发布结果。"""
        start_time = time.time()
        try:
            cv_img0 = self.bridge.imgmsg_to_cv2(ros_img0_msg, 'bgr8')
            cv_img1 = self.bridge.imgmsg_to_cv2(ros_img1_msg, 'bgr8')

            # 调用核心处理函数
            stitched_image = self.stitcher.process_frames(cv_img0, cv_img1, self.global_frame_idx)

            # --- 发布原始图像 ---
            stitched_msg = self.bridge.cv2_to_imgmsg(stitched_image, 'bgr8')
            stitched_msg.header = ros_img0_msg.header
            stitched_msg.header.frame_id = 'stitched_camera_frame'
            self.stitched_image_publisher.publish(stitched_msg)

            # --- 新增：发布压缩图像 ---
            compressed_msg = CompressedImage()
            compressed_msg.header = ros_img0_msg.header
            compressed_msg.header.frame_id = 'stitched_camera_frame'
            compressed_msg.format = "jpeg"
            # 将图像编码为JPEG格式
            success, encoded_image = cv2.imencode('.jpg', stitched_image)
            if success:
                compressed_msg.data = encoded_image.tobytes()
                self.compressed_image_publisher.publish(compressed_msg)
            # --- 压缩图像发布结束 ---

            self.global_frame_idx += 1
            self.frame_count += 1
            total_time = time.time() - start_time
            if self.frame_count % 30 == 0:
                fps = 1.0 / total_time if total_time > 0 else 0
                self.get_logger().info(f'FPS: {fps:.1f}, '
                                     f'处理时间: {total_time*1000:.2f}ms, '
                                     f'输出尺寸: {stitched_image.shape[1]}x{stitched_image.shape[0]}')

        except Exception as e:
            self.get_logger().error(f'处理图像时发生错误: {e}\n{traceback.format_exc()}')


def main(args=None):
    rclpy.init(args=args)
    image_stitcher_node = ImageStitcherNode()
    try:
        rclpy.spin(image_stitcher_node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            image_stitcher_node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
