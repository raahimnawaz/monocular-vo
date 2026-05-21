"""Keyframe data structure + selection heuristics.

A keyframe is a frame the back-end optimization "remembers." Keeping every
frame would blow up the factor graph; keyframes are picked sparsely enough
to be efficient and densely enough that loop closures can find them.

Selection heuristic (any one triggers a new keyframe):
    1. translation since last keyframe exceeds `min_translation_m`
    2. rotation since last keyframe exceeds `min_rotation_deg`
    3. number of inlier matches between current frame and last keyframe
       drops below `min_inliers`
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Keyframe:
    index: int  # original frame index in the video
    keyframe_id: int
    R_world: np.ndarray  # (3, 3) rotation world -> camera
    t_world: np.ndarray  # (3,) camera origin in world frame
    descriptors: np.ndarray  # (N, 32) ORB descriptors
    keypoints_uv: np.ndarray  # (N, 2) keypoint pixel coords
    depth: np.ndarray | None = None  # (H, W) metric depth at this frame


@dataclass
class KeyframeSelector:
    min_translation_m: float = 0.30
    min_rotation_deg: float = 10.0
    min_inliers: int = 200
    max_frames_between: int = 30

    last_keyframe: Keyframe | None = field(default=None, init=False)
    frames_since_last: int = field(default=0, init=False)

    def should_insert(
        self,
        R_world: np.ndarray,
        t_world: np.ndarray,
        num_inliers: int,
    ) -> bool:
        self.frames_since_last += 1
        if self.last_keyframe is None:
            return True
        delta_t = np.linalg.norm(t_world - self.last_keyframe.t_world)
        if delta_t >= self.min_translation_m:
            return True
        R_rel = self.last_keyframe.R_world.T @ R_world
        rvec, _ = cv2.Rodrigues(R_rel)
        angle_deg = float(np.degrees(np.linalg.norm(rvec)))
        if angle_deg >= self.min_rotation_deg:
            return True
        if num_inliers < self.min_inliers:
            return True
        if self.frames_since_last >= self.max_frames_between:
            return True
        return False

    def insert(self, kf: Keyframe) -> None:
        self.last_keyframe = kf
        self.frames_since_last = 0
