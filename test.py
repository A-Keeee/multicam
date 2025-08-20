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
import traceback

class ImageStitcherNode(Node):
    """
    一个ROS 2节点，用于实时拼接来自两个摄像头的图像。
    采用精确的三维柱面重投影模型，能够根据配置文件自动适应
    'plumb_bob' (径向-切向) 和 'fisheye' (鱼眼) 两种畸变模型。
    """
    def __init__(self):
        super().__init__('image_stitcher_node')
        self.get_logger().info('图像拼接节点已启动...')

        # --- 1. 声明并获取ROS 2参数 ---
        self.declare_parameter('config_file_path', '/home/ake/multicam/cam_calib.yaml')
        self.declare_parameter('compression_scale', 0.8)
        self.declare_parameter('pano_width_multiplier', 1.5) # 全景图宽度乘数
        self.declare_parameter('ignore_translation', False)  
        self.declare_parameter('auto_crop', True)
        self.declare_parameter('crop_margin', 5)  # 像素边距
        self.declare_parameter('enable_refine', True)
        self.declare_parameter('refine_min_matches', 40)
        self.declare_parameter('use_affine_refine', True)
        self.declare_parameter('refine_smooth_alpha', 0.15)  # 新旧形变插值权重
        self.declare_parameter('refine_update_interval', 3)  # 每多少帧尝试一次新的细化
        
        self.config_file = self.get_parameter('config_file_path').get_parameter_value().string_value
        self.compression_scale = self.get_parameter('compression_scale').get_parameter_value().double_value
        self.pano_width_multiplier = self.get_parameter('pano_width_multiplier').get_parameter_value().double_value
        self.ignore_translation = self.get_parameter('ignore_translation').get_parameter_value().bool_value
        # 读取新增参数
        self.auto_crop = self.get_parameter('auto_crop').get_parameter_value().bool_value
        self.crop_margin = int(self.get_parameter('crop_margin').get_parameter_value().integer_value) if self.has_parameter('crop_margin') else 8
        self.crop_bounds = None  # (r0,r1,c0,c1)
        self.enable_refine = self.get_parameter('enable_refine').get_parameter_value().bool_value
        self.refine_min_matches = int(self.get_parameter('refine_min_matches').get_parameter_value().integer_value) if self.has_parameter('refine_min_matches') else 25
        self.use_affine_refine = self.get_parameter('use_affine_refine').get_parameter_value().bool_value
        self.refine_smooth_alpha = self.get_parameter('refine_smooth_alpha').get_parameter_value().double_value
        self.refine_update_interval = int(self.get_parameter('refine_update_interval').get_parameter_value().integer_value) if self.has_parameter('refine_update_interval') else 5
        # 缺失的缓存变量初始化 (修复 AttributeError)
        self.prev_affine = None
        self.prev_homo = None
        self.prev_dyn_w0 = None
        self.prev_dyn_w1 = None
        self.global_frame_idx = 0

        self.get_logger().info(f"配置文件路径: {self.config_file}")
        self.get_logger().info(f"压缩比例: {self.compression_scale}")
        self.get_logger().info(f"全景图宽度乘数: {self.pano_width_multiplier}")

        self.frame_count = 0
        self.bridge = CvBridge()

        # --- 2. 加载和解析相机参数 ---
        self.cam0_params, self.cam1_params = self._load_camera_params()
        
        self.K0, self.D0, (self.w0, self.h0) = self.cam0_params['intrinsics'], self.cam0_params['distortion_coeffs'], self.cam0_params['resolution']
        self.K1, self.D1, (self.w1, self.h1) = self.cam1_params['intrinsics'], self.cam1_params['distortion_coeffs'], self.cam1_params['resolution']

        # --- 3. 进行预计算 ---
        self._prepare_for_reprojection()

        # --- 4. 初始化ROS订阅器和发布器 ---
        self.stitched_image_publisher = self.create_publisher(Image, '/image_stitched', 10)
        sub_cam0 = message_filters.Subscriber(self, Image, self.cam0_params['rostopic'])
        sub_cam1 = message_filters.Subscriber(self, Image, self.cam1_params['rostopic'])
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer([sub_cam0, sub_cam1], queue_size=10, slop=0.1)
        self.time_synchronizer.registerCallback(self.image_callback)
        self.get_logger().info('订阅器已设置，等待图像消息...')

    def _load_camera_params(self):
        """从YAML文件中加载并解析两个相机的参数，包括畸变模型。"""
        if not os.path.exists(self.config_file):
            self.get_logger().error(f"配置文件未找到: {self.config_file}")
            raise FileNotFoundError(f"配置文件未找到: {self.config_file}")
        with open(self.config_file, 'r') as file:
            config = yaml.safe_load(file)
        
        def parse_cam_config(cam_config):
            intrinsics_flat = cam_config['intrinsics']
            distortion_model = cam_config.get('distortion_model', 'plumb_bob')
            self.get_logger().info(f"相机 {cam_config['rostopic']} 使用 '{distortion_model}' 畸变模型。")

            return {
                'intrinsics': np.array([[intrinsics_flat[0], 0, intrinsics_flat[2]], [0, intrinsics_flat[1], intrinsics_flat[3]], [0, 0, 1]], dtype=np.float32),
                'distortion_coeffs': np.array(cam_config['distortion_coeffs'], dtype=np.float32),
                'distortion_model': distortion_model,
                'resolution': cam_config['resolution'],
                'rostopic': cam_config['rostopic'],
                'T_cn_cnm1': np.array(cam_config.get('T_cn_cnm1', np.eye(4)), dtype=np.float32)
            }

        cam0_params = parse_cam_config(config['cam0'])
        cam1_params = parse_cam_config(config['cam1'])
        
        self.get_logger().info(f'成功从 {self.config_file} 加载相机参数')
        return cam0_params, cam1_params

    def _prepare_for_reprojection(self):
        """为三维重投影模式准备所有计算，主要是生成映射表(LUT)。"""
        self.get_logger().info("正在为 'reprojection' 模式进行矢量化预计算...")
        start_time = time.time()

        # 先生成两路去畸变映射并得到无畸变新内参
        self.newK0, self.ud_map1_0, self.ud_map2_0 = self._make_undistort_maps(self.K0, self.D0, (self.w0, self.h0), self.cam0_params['distortion_model'])
        self.newK1, self.ud_map1_1, self.ud_map2_1 = self._make_undistort_maps(self.K1, self.D1, (self.w1, self.h1), self.cam1_params['distortion_model'])
        
        scale = self.compression_scale
        self.pano_h = int(self.h0 * scale)
        self.pano_w = int(self.w0 * scale * self.pano_width_multiplier)
        
        # 缩放无畸变内参
        K0_s = self.newK0.copy(); K0_s[:2, :] *= scale
        K1_s = self.newK1.copy(); K1_s[:2, :] *= scale
        
        # 获取从cam0到cam1的变换矩阵
        T_01 = self.cam1_params['T_cn_cnm1']
        R_01 = T_01[0:3, 0:3].copy()
        if self.ignore_translation:
            t_01 = np.zeros(3, dtype=np.float32)
        else:
            t_01 = T_01[0:3, 3]
        
        # --- 生成柱面全景方向向量 ---
        u_pano, v_pano = np.meshgrid(np.arange(self.pano_w), np.arange(self.pano_h))
        f = K0_s[0, 0]
        cx_pano = self.pano_w / 2.0
        cy_pano = self.pano_h / 2.0
        theta = (u_pano - cx_pano) / f
        h = (v_pano - cy_pano)
        p_cam0 = np.stack([np.sin(theta), h / f, np.cos(theta)], axis=-1)  # 归一化方向
        # 归一化（保持单位方向）
        norm = np.linalg.norm(p_cam0, axis=-1, keepdims=True)
        p_cam0 /= np.clip(norm, 1e-8, None)
        
        # cam1 方向（仅旋转 + 可选平移对远景不敏感，理论上平移已忽略）
        if self.ignore_translation:
            p_cam1 = (p_cam0 @ R_01.T)
        else:
            # 将方向延伸成点再加平移（简化：假设深度=1）
            p_cam1 = (p_cam0 @ R_01.T) + t_01
        
        # 投影到无畸变图像平面 (pinhole) —— 不再调用带畸变的projectPoints
        def project_unit_dirs(p_dirs, K):
            px = p_dirs[..., 0]
            py = p_dirs[..., 1]
            pz = p_dirs[..., 2]
            valid = pz > 1e-6
            u = K[0, 0] * (px / pz) + K[0, 2]
            v = K[1, 1] * (py / pz) + K[1, 2]
            u[~valid] = -1
            v[~valid] = -1
            return u.astype(np.float32), v.astype(np.float32)
        
        u0, v0 = project_unit_dirs(p_cam0, K0_s)
        u1, v1 = project_unit_dirs(p_cam1, K1_s)
        
        self.map_x0, self.map_y0 = u0, v0
        self.map_x1, self.map_y1 = u1, v1
        
        self._create_reprojection_blender()
        # 生成裁剪范围
        if self.auto_crop:
            self._compute_crop_bounds()

        self.get_logger().info(f"重投影预计算完成，耗时: {time.time() - start_time:.2f} 秒")
        self.get_logger().info(f"全景图输出尺寸: {self.pano_w} x {self.pano_h}")

    def _make_undistort_maps(self, K, D, wh, model):
        """生成去畸变映射与新的无畸变内参矩阵。"""
        w, h = wh
        if model in ['fisheye', 'equidistant']:
            # 估计新的相机矩阵（保留全部视场）
            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D[:4], (w, h), np.eye(3), balance=1.0, new_size=(w, h))
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D[:4], np.eye(3), newK, (w, h), cv2.CV_32FC1)
        else:
            newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0.0, (w, h))
            map1, map2 = cv2.initUndistortRectifyMap(K, D, None, newK, (w, h), cv2.CV_32FC1)
        return newK.astype(np.float32), map1, map2

    def _create_maps_from_points(self, p_2d, w, h):
        """根据投影点生成映射表，并标记无效点。"""
        map_x = p_2d[..., 0].astype(np.float32)
        map_y = p_2d[..., 1].astype(np.float32)

        invalid_mask = (map_x < 0) | (map_x >= w) | (map_y < 0) | (map_y >= h)
        map_x[invalid_mask] = -1
        map_y[invalid_mask] = -1
        return map_x, map_y

    def _create_reprojection_blender(self):
        """为重投影模式创建融合权重。"""
        mask0 = (self.map_x0 > -1).astype(np.uint8) * 255
        mask1 = (self.map_x1 > -1).astype(np.uint8) * 255
        
        dist0 = cv2.distanceTransform(mask0, cv2.DIST_L2, 5)
        dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5)
        
        total_dist = dist0 + dist1
        total_dist[total_dist == 0] = 1.0

        weight0 = dist0 / total_dist
        weight1 = dist1 / total_dist

        self.blender_weight_0 = cv2.merge([weight0, weight0, weight0]).astype(np.float32)
        self.blender_weight_1 = cv2.merge([weight1, weight1, weight1]).astype(np.float32)
        self.get_logger().info("重投影融合权重已生成。")

    def _compute_crop_bounds(self):
        """依据有效映射区域自动计算裁剪范围，去除大面积黑边。"""
        valid0 = (self.map_x0 > -1) & (self.map_y0 > -1)
        valid1 = (self.map_x1 > -1) & (self.map_y1 > -1)
        union = (valid0 | valid1)
        if not np.any(union):
            self.get_logger().warn('自动裁剪: 未找到有效像素，跳过。')
            return
        rows = np.where(union.any(axis=1))[0]
        cols = np.where(union.any(axis=0))[0]
        r0, r1 = rows[0], rows[-1]
        c0, c1 = cols[0], cols[-1]
        # 加 margin 并夹紧
        r0 = max(0, r0 - self.crop_margin)
        c0 = max(0, c0 - self.crop_margin)
        r1 = min(self.pano_h - 1, r1 + self.crop_margin)
        c1 = min(self.pano_w - 1, c1 + self.crop_margin)
        self.crop_bounds = (r0, r1, c0, c1)
        self.get_logger().info(f'自动裁剪范围: rows {r0}:{r1}, cols {c0}:{c1}')

    def _refine_overlap(self, pano0, pano1):
        """特征匹配细化 + 自适应缝合权重 + 时序平滑。
        返回: (pano1_aligned, dyn_w0, dyn_w1) 或 (回退)."""
        try:
            gray0 = cv2.cvtColor(pano0, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(pano1, cv2.COLOR_BGR2GRAY)
            mask0 = (gray0 > 0).astype(np.uint8)
            mask1 = (gray1 > 0).astype(np.uint8)
            overlap = cv2.bitwise_and(mask0, mask1)
            if overlap.sum() < 5000:
                # 重叠太少, 直接复用上一帧稳定结果
                if self.prev_affine is not None or self.prev_homo is not None:
                    pano1_warp = self._apply_cached_transform(pano1)
                    return pano1_warp, self.prev_dyn_w0, self.prev_dyn_w1
                return pano1, None, None

            # 仅按间隔尝试重新估计; 否则复用缓存减少抖动
            recalc = (self.global_frame_idx % self.refine_update_interval == 0) or (self.prev_affine is None and self.prev_homo is None)
            if not recalc:
                pano1_warp = self._apply_cached_transform(pano1)
                return pano1_warp, self.prev_dyn_w0, self.prev_dyn_w1

            ys, xs = np.where(overlap > 0)
            y0, y1 = ys.min(), ys.max(); x0, x1 = xs.min(), xs.max()
            roi0 = gray0[y0:y1+1, x0:x1+1]
            roi1 = gray1[y0:y1+1, x0:x1+1]
            orb = cv2.ORB_create(nfeatures=4000, fastThreshold=7)
            kps0, des0 = orb.detectAndCompute(roi0, None)
            kps1, des1 = orb.detectAndCompute(roi1, None)
            if des0 is None or des1 is None or len(kps0) < 10 or len(kps1) < 10:
                pano1_warp = self._apply_cached_transform(pano1)
                return pano1_warp, self.prev_dyn_w0, self.prev_dyn_w1
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des0, des1)
            if len(matches) < self.refine_min_matches:
                pano1_warp = self._apply_cached_transform(pano1)
                return pano1_warp, self.prev_dyn_w0, self.prev_dyn_w1
            matches = sorted(matches, key=lambda m: m.distance)
            top = matches[:max(self.refine_min_matches, int(len(matches)*0.6))]
            pts0 = np.float32([kps0[m.queryIdx].pt for m in top])
            pts1 = np.float32([kps1[m.trainIdx].pt for m in top])
            pts0[:,0] += x0; pts0[:,1] += y0
            pts1[:,0] += x0; pts1[:,1] += y0

            # 估计形变
            if self.use_affine_refine:
                M, inliers = cv2.estimateAffinePartial2D(pts1, pts0, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                if M is None:
                    pano1_warp = self._apply_cached_transform(pano1)
                    return pano1_warp, self.prev_dyn_w0, self.prev_dyn_w1
                # 平滑仿射 (直接对矩阵元素 EMA)
                if self.prev_affine is not None:
                    alpha = self.refine_smooth_alpha
                    M = (1-alpha)*self.prev_affine + alpha*M
                pano1_warp = cv2.warpAffine(pano1, M, (pano1.shape[1], pano1.shape[0]), flags=cv2.INTER_LINEAR)
                self.prev_affine = M
                self.prev_homo = None
            else:
                H, inliers = cv2.findHomography(pts1, pts0, cv2.RANSAC, 3.0)
                if H is None:
                    pano1_warp = self._apply_cached_transform(pano1)
                    return pano1_warp, self.prev_dyn_w0, self.prev_dyn_w1
                if self.prev_homo is not None:
                    alpha = self.refine_smooth_alpha
                    # 简单线性插值再归一化
                    H = (1-alpha)*self.prev_homo + alpha*H
                    if abs(H[2,2]) > 1e-8:
                        H /= H[2,2]
                pano1_warp = cv2.warpPerspective(pano1, H, (pano1.shape[1], pano1.shape[0]), flags=cv2.INTER_LINEAR)
                self.prev_homo = H
                self.prev_affine = None

            # 自适应权重 (与之前逻辑相同)
            g0a = gray0
            g1a = cv2.cvtColor(pano1_warp, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(g0a, g1a).astype(np.float32)
            diff[overlap == 0] = 0
            nonzero = diff[overlap>0]
            if nonzero.size == 0:
                return pano1_warp, None, None
            med = np.median(nonzero)
            mask_pref0 = (diff <= med).astype(np.float32)
            mask_pref1 = 1.0 - mask_pref0
            ksize = max(5, (min(pano0.shape[0], pano0.shape[1])//80)*2+1)
            w0 = cv2.GaussianBlur(mask_pref0, (ksize, ksize), 0)
            w1 = cv2.GaussianBlur(mask_pref1, (ksize, ksize), 0)
            s = w0 + w1 + 1e-6
            w0 /= s; w1 /= s
            w0_3 = cv2.merge([w0,w0,w0]).astype(np.float32)
            w1_3 = cv2.merge([w1,w1,w1]).astype(np.float32)
            self.prev_dyn_w0 = w0_3
            self.prev_dyn_w1 = w1_3
            return pano1_warp, w0_3, w1_3
        except Exception:
            # 失败时复用缓存
            if self.prev_affine is not None or self.prev_homo is not None:
                pano1_warp = self._apply_cached_transform(pano1)
                return pano1_warp, self.prev_dyn_w0, self.prev_dyn_w1
            return pano1, None, None

    def _apply_cached_transform(self, img):
        """对 pano1 复用上一帧形变 (减少抖动)。"""
        if self.prev_affine is not None:
            return cv2.warpAffine(img, self.prev_affine, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        if self.prev_homo is not None:
            return cv2.warpPerspective(img, self.prev_homo, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        return img

    def image_callback(self, ros_img0_msg, ros_img1_msg):
        """根据预计算的映射表，对图像进行重映射和融合。"""
        start_time = time.time()
        try:
            cv_img0 = self.bridge.imgmsg_to_cv2(ros_img0_msg, 'bgr8')
            cv_img1 = self.bridge.imgmsg_to_cv2(ros_img1_msg, 'bgr8')
            # 先去畸变
            und0 = cv2.remap(cv_img0, self.ud_map1_0, self.ud_map2_0, cv2.INTER_LINEAR)
            und1 = cv2.remap(cv_img1, self.ud_map1_1, self.ud_map2_1, cv2.INTER_LINEAR)
            scale = self.compression_scale
            und0_s = cv2.resize(und0, (int(self.w0*scale), int(self.h0*scale)), interpolation=cv2.INTER_LINEAR)
            und1_s = cv2.resize(und1, (int(self.w1*scale), int(self.h1*scale)), interpolation=cv2.INTER_LINEAR)
            # 使用预生成map投影（map_x/y 已基于无畸变内参）
            pano0 = cv2.remap(und0_s, self.map_x0, self.map_y0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            pano1 = cv2.remap(und1_s, self.map_x1, self.map_y1, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # --- 动态细化 ---
            if self.enable_refine:
                pano1, dyn_w0, dyn_w1 = self._refine_overlap(pano0, pano1)
            else:
                dyn_w0 = dyn_w1 = None

            # --- 权重选择 ---
            if dyn_w0 is not None:
                w0 = dyn_w0
                w1 = dyn_w1
            else:
                w0 = self.prev_dyn_w0 if self.prev_dyn_w0 is not None else self.blender_weight_0
                w1 = self.prev_dyn_w1 if self.prev_dyn_w1 is not None else self.blender_weight_1

            # --- 鲁棒的图像合成 ---
            # 步骤 1: 使用 np.maximum 创建一个包含所有非黑像素的基础图像。
            # 这可以确保 pano1 中非重叠部分的像素不会因变换或权重问题而丢失，但会在重叠区产生硬接缝。
            stitched_image = np.maximum(pano0, pano1)

            # 步骤 2: 识别出两张图真正重叠的区域。
            # 注意: 这里的 pano1 已经是经过 refine (warp) 之后的版本。
            gray0 = cv2.cvtColor(pano0, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(pano1, cv2.COLOR_BGR2GRAY)
            overlap_mask = (gray0 > 0) & (gray1 > 0)

            # 步骤 3: 仅在重叠区域内计算平滑融合的像素值。
            blended_overlap = pano0.astype(np.float32) * w0 + pano1.astype(np.float32) * w1
            
            # 步骤 4: 将平滑融合的结果应用到重叠区域，替换掉硬接缝。
            # 使用 [overlap_mask] 作为索引，可以高效地只操作目标像素。
            stitched_image[overlap_mask] = np.clip(blended_overlap[overlap_mask], 0, 255).astype(np.uint8)

            # --- 后处理与发布 ---
            # 自动裁剪
            if self.crop_bounds is not None:
                r0, r1, c0, c1 = self.crop_bounds
                stitched_image = stitched_image[r0:r1+1, c0:c1+1]

            self.global_frame_idx += 1
            self.frame_count += 1
            total_time = time.time() - start_time
            if self.frame_count % 30 == 0:
                fps = 1.0 / total_time if total_time > 0 else 0
                self.get_logger().info(f'FPS: {fps:.1f}, '
                                     f'处理时间: {total_time*1000:.2f}ms, '
                                     f'输出尺寸: {stitched_image.shape[1]}x{stitched_image.shape[0]}')

            stitched_msg = self.bridge.cv2_to_imgmsg(stitched_image, 'bgr8')
            stitched_msg.header = ros_img0_msg.header
            stitched_msg.header.frame_id = 'stitched_camera_frame'
            self.stitched_image_publisher.publish(stitched_msg)

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