"""Target localisation: bearing from a tracked pixel + camera pose + ground-plane assumption. No ROS."""
from __future__ import annotations                                   # forward type hints
import numpy as np                                                   # arrays


def pixel_to_world_on_plane(px: np.ndarray, camera, T_w_c: np.ndarray, plane_z: float):   # ray-plane intersection
    n = camera.undistort_to_normalized(np.asarray(px, dtype=np.float64).reshape(1, 2))[0]   # normalized ray (x/z, y/z)
    d_c = np.array([n[0], n[1], 1.0])                                # ray direction in camera frame
    d_w = T_w_c[:3, :3] @ d_c                                        # ray direction in world frame
    c_w = T_w_c[:3, 3]                                               # camera centre in world frame
    if abs(d_w[2]) < 1e-6:                                           # ray parallel to the plane
        return None                                                  # no intersection
    lam = (plane_z - c_w[2]) / d_w[2]                                # distance along the ray
    if lam <= 0:                                                     # intersection behind the camera
        return None                                                  # invalid
    return c_w + lam * d_w                                           # world point on the plane
