# 卫星相机视场计算
# sim/satellite_fov_calculation.py 的头部导入区域
import math
import numpy as np
from typing import List
from pymap3d import geodetic2ecef

EARTH_RADIUS = 6371000.0


class SatelliteFovCalculation:
    # 将大地坐标转换为地心地固坐标
    def ecef_from_geodetic(self, lon: float, lat: float, alt: float) -> np.ndarray:
        x, y, z = geodetic2ecef(lat, lon, alt)
        return np.array([x, y, z])

    # 计算向量的单位向量
    def normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm == 0:
            raise ValueError("零向量无法归一化")

        return v / norm

    # 建立相机视场坐标系的旋转矩阵
    def build_camera_frame_from_axis(self, axis_ecef: np.ndarray, ref_vec: np.ndarray = None) -> np.ndarray:
        u = axis_ecef
        x_axis = None

        if ref_vec is not None:
            ref = ref_vec - np.dot(ref_vec, u) * u
            norm_ref = np.linalg.norm(ref)
            if norm_ref > 1e-6:
                x_axis = ref / norm_ref

        if x_axis is None:
            ref = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(u, ref)) > 0.999:
                ref = np.array([1.0, 0.0, 0.0])
            x_axis = self.normalize(np.cross(u, ref))

        y_axis = np.cross(u, x_axis)
        r_matrix = np.column_stack((x_axis, y_axis, u))

        return r_matrix

    # 坐标旋转
    def apply_rotation(self, R: np.ndarray, v: np.ndarray) -> np.ndarray:
        return R.T @ v

    # 判断目标与卫星之间是否被地球遮挡
    def is_earth_occluded(self, sat_ecef: np.ndarray, tgt_ecef: np.ndarray) -> bool:
        d = tgt_ecef - sat_ecef
        dist_st = np.linalg.norm(d)
        if dist_st == 0:
            return True

        a = np.dot(d, d)
        b = 2 * np.dot(sat_ecef, d)
        c = np.dot(sat_ecef, sat_ecef) - EARTH_RADIUS ** 2
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return False

        sqrt_disc = math.sqrt(discriminant)

        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        return (0 <= t1 <= 1) or (0 <= t2 <= 1)

    # 向量旋转-罗德里格斯旋转公式
    def rotate_vector(self, v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        return v * cos_a + np.cross(axis, v) * sin_a + axis * np.dot(axis, v) * (1 - cos_a)

    # 找出在相机视场范围内的目标
    def find_targets_in_fov(self,
                            sat_current_geo_pos: np.ndarray,
                            fov_az: float,
                            fov_el: float,
                            center_target_geo_pos: np.ndarray,
                            max_pointing_angle: float,
                            targets: List[np.ndarray],
                            sat_last_geo_pos: np.ndarray = None) -> List[int]:
        # Step 1：转换为ECEF坐标
        sat_current_ecef = self.ecef_from_geodetic(*sat_current_geo_pos)
        center_ecef = self.ecef_from_geodetic(*center_target_geo_pos)

        # Step 2：计算天底方向（卫星指向地心）
        nadir_dir = self.normalize(-sat_current_ecef)

        # Step 3：计算期望指向方向（卫星指向中心目标）
        desired_dir = self.normalize(center_ecef - sat_current_ecef)

        # Step 4：计算期望方向与天底方向夹角
        cos_theta = np.clip(np.dot(desired_dir, nadir_dir), -1.0, 1.0)
        theta = math.acos(cos_theta)

        # Step 5：确定实际光轴方向
        max_angle = math.radians(max_pointing_angle)
        if theta <= max_angle:
            actual_axis = desired_dir
        else:
            rot_axis = np.cross(nadir_dir, desired_dir)
            axis_norm = np.linalg.norm(rot_axis)
            if axis_norm < 1e-12:
                if theta < 1e-6:
                    actual_axis = nadir_dir
                else:
                    if abs(nadir_dir[0]) < 0.9:
                        ref_axis = np.array([1.0, 0.0, 0.0])
                    else:
                        ref_axis = np.array([0.0, 1.0, 0.0])

                rot_axis = self.normalize(np.cross(nadir_dir, ref_axis))
            else:
                rot_axis = rot_axis / axis_norm

            actual_axis = self.normalize(self.rotate_vector(nadir_dir, rot_axis, max_angle))

        # Step 6：建立相机坐标系旋转矩阵
        ref_vec = None
        if sat_last_geo_pos is not None:
            sat_last_ecef = self.ecef_from_geodetic(*sat_last_geo_pos)
            ref_vec = sat_current_ecef - sat_last_ecef
            norm_ref_vec = np.linalg.norm(ref_vec)
            if norm_ref_vec > 1e-12:
                ref_vec = ref_vec / norm_ref_vec

        r_matrix = self.build_camera_frame_from_axis(actual_axis, ref_vec)

        # Step 7：计算目标是否在视场角内
        half_az = math.radians(fov_az / 2.0)
        half_el = math.radians(fov_el / 2.0)

        in_fov_indices = []
        for i, tgt in enumerate(targets):
            tgt_ecef = self.ecef_from_geodetic(*tgt)
            if self.is_earth_occluded(sat_current_ecef, tgt_ecef):
                continue

            vec_ecef = tgt_ecef - sat_current_ecef
            vec_cam = self.apply_rotation(r_matrix, vec_ecef)

            x, y, z = vec_cam
            if z <= 0:
                continue

            azi = math.atan2(x, z)
            ele = math.atan2(y, z)
            if abs(azi) <= half_az and abs(ele) <= half_el:
                in_fov_indices.append(i)

        return in_fov_indices

    # 判断目标是否能够被低轨卫星对空相机探测
    def judge_target_can_be_detected_by_low_orbit_satellite_air_camera(self,
                                                                       sat_current_geo_pos: np.ndarray,
                                                                       ele_min: float,
                                                                       ele_max: float,
                                                                       target_geo_pos: np.ndarray,
                                                                       sat_last_geo_pos: np.ndarray = None
                                                                       ) -> bool:
        # Step 1：转换为ECEF坐标
        sat_current_ecef = self.ecef_from_geodetic(*sat_current_geo_pos)
        target_current_ecef = self.ecef_from_geodetic(*target_geo_pos)
        center_ecef = np.array([0.0, 0.0, 0.0])

        # Step 2：默认相机指向地心，计算相机光轴方向
        actual_axis = self.normalize(center_ecef - sat_current_ecef)

        # Step 3：建立相机坐标系旋转矩阵
        ref_vec = None
        if sat_last_geo_pos is not None:
            sat_last_ecef = self.ecef_from_geodetic(*sat_last_geo_pos)
            ref_vec = sat_current_ecef - sat_last_ecef
            norm_ref_vec = np.linalg.norm(ref_vec)
            if norm_ref_vec > 1e-12:
                ref_vec = ref_vec / norm_ref_vec

        r_matrix = self.build_camera_frame_from_axis(actual_axis, ref_vec)

        # Step 4：计算目标是否在视场角内
        if self.is_earth_occluded(sat_current_ecef, target_geo_pos):
            return False
        else:
            vec_ecef = target_current_ecef - sat_current_ecef
            vec_cam = self.apply_rotation(r_matrix, vec_ecef)

            x, y, z = vec_cam
            ele_radians = math.atan2(y, z)
            ele_degrees = np.degrees(ele_radians)
            if 90.0 + ele_min <= abs(ele_degrees) <= 90.0 + ele_max:
                return True
            else:
                return False

    # 判断目标是否能够被卫星对地相机探测
    def judge_target_can_be_detected_by_satellite_ground_camera(self,
                                                                sat_current_geo_pos: np.ndarray,
                                                                azi_min: float,
                                                                azi_max: float,
                                                                ele_min: float,
                                                                ele_max: float,
                                                                target_geo_pos: np.ndarray,
                                                                sat_last_geo_pos: np.ndarray = None
                                                                ) -> bool:
        # Step 1：转换为ECEF坐标
        sat_current_ecef = self.ecef_from_geodetic(*sat_current_geo_pos)
        target_current_ecef = self.ecef_from_geodetic(*target_geo_pos)
        center_ecef = np.array([0.0, 0.0, 0.0])

        # Step 2：默认相机指向地心，计算相机光轴方向
        actual_axis = self.normalize(center_ecef - sat_current_ecef)

        # Step 3：建立相机坐标系旋转矩阵
        ref_vec = None
        if sat_last_geo_pos is not None:
            sat_last_ecef = self.ecef_from_geodetic(*sat_last_geo_pos)
            ref_vec = sat_current_ecef - sat_last_ecef
            norm_ref_vec = np.linalg.norm(ref_vec)
            if norm_ref_vec > 1e-12:
                ref_vec = ref_vec / norm_ref_vec

        r_matrix = self.build_camera_frame_from_axis(actual_axis, ref_vec)

        # Step 4：计算目标是否在视场角内
        if self.is_earth_occluded(sat_current_ecef, target_geo_pos):
            return False
        else:
            vec_ecef = target_current_ecef - sat_current_ecef
            vec_cam = self.apply_rotation(r_matrix, vec_ecef)

            x, y, z = vec_cam
            if z <= 0:
                return False

            azi_radians = math.atan2(x, z)
            ele_radians = math.atan2(y, z)
            azi_degrees = np.degrees(azi_radians)
            ele_degrees = np.degrees(ele_radians)

            if azi_min <= azi_degrees <= azi_max and ele_min <= ele_degrees <= ele_max:
                return True
            else:
                return False
