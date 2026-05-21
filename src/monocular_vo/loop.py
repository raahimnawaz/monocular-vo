"""Loop closure: detect when a keyframe revisits a previously-seen scene.

Two stages:

1. **Detection (cheap):** match ORB descriptors between the current keyframe
   and a candidate prior keyframe; the score is the number of Lowe-ratio
   matches. Above a threshold, the candidate is a closure proposal.

2. **Geometric verification (slower):** estimate the SE(3) transform between
   the two keyframes via depth-PnP-RANSAC, the same procedure as the VO
   front-end. If RANSAC finds enough inliers, accept the closure and emit a
   `LoopClosure` with the relative pose.

Candidates are skipped if they're temporally close to the current keyframe
(within `temporal_skip` keyframes) to avoid trivial "closures" with the
immediate past.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .keyframe import Keyframe
from .vo import StepResult, backproject, estimate_relative_pose


@dataclass
class LoopClosure:
    from_id: int  # current keyframe id
    to_id: int  # earlier keyframe id (the one being looped back to)
    R: np.ndarray  # rotation from `from` to `to`
    t: np.ndarray  # translation in `from` frame (meters)
    num_inliers: int


@dataclass
class LoopDetector:
    K: np.ndarray
    descriptor_ratio: float = 0.75
    candidate_match_threshold: int = 100
    temporal_skip: int = 10
    verification_inlier_threshold: int = 80
    max_relative_translation_m: float | None = 3.0
    """Reject closure candidates whose PnP-recovered translation is larger than this.

    Real loop closures are between keyframes that are physically near each other;
    rejecting large-baseline matches filters out a major class of false positives
    in repetitive scenes (hallways, identical room corners, etc.). Set to None to
    disable.
    """

    def detect(
        self,
        current: Keyframe,
        all_keyframes: list[Keyframe],
    ) -> LoopClosure | None:
        """Search prior keyframes for a loop-closure candidate.

        Returns the strongest verified closure (most inliers) or None.
        """
        if current.depth is None:
            return None  # cannot do depth-PnP verification without depth

        best: tuple[int, StepResult] | None = None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        eligible = [kf for kf in all_keyframes if kf.keyframe_id < current.keyframe_id - self.temporal_skip]
        if not eligible:
            return None

        # Stage 1: descriptor-match each eligible keyframe
        candidates: list[tuple[Keyframe, list[cv2.DMatch]]] = []
        for kf in eligible:
            if len(kf.descriptors) == 0 or len(current.descriptors) == 0:
                continue
            raw = bf.knnMatch(kf.descriptors, current.descriptors, k=2)
            good: list[cv2.DMatch] = []
            for pair in raw:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < self.descriptor_ratio * n.distance:
                    good.append(m)
            if len(good) >= self.candidate_match_threshold:
                candidates.append((kf, good))

        if not candidates:
            return None

        # Stage 2: geometric verification on the top candidates by descriptor score
        candidates.sort(key=lambda x: len(x[1]), reverse=True)
        for kf, matches in candidates[:5]:
            uv_kf = np.array([kf.keypoints_uv[m.queryIdx] for m in matches], dtype=np.float64)
            uv_cur = np.array([current.keypoints_uv[m.trainIdx] for m in matches], dtype=np.float64)
            pts3d_kf = backproject(uv_kf, kf.depth, self.K)
            result = estimate_relative_pose(pts3d_kf, uv_cur, self.K)
            if result.num_inliers < self.verification_inlier_threshold:
                continue
            if (
                self.max_relative_translation_m is not None
                and float(np.linalg.norm(result.t)) > self.max_relative_translation_m
            ):
                continue
            if best is None or result.num_inliers > best[1].num_inliers:
                best = (kf.keyframe_id, result)

        if best is None:
            return None
        kf_id, res = best
        return LoopClosure(
            from_id=current.keyframe_id,
            to_id=kf_id,
            R=res.R,
            t=res.t,
            num_inliers=res.num_inliers,
        )
