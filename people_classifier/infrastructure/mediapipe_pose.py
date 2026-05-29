import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from domain.entities import BoundingBox, HumanBodyStats, Keypoint
from domain.interfaces import IPoseEstimator

logger = logging.getLogger(__name__)

# MediaPipe landmark index → readable name (same indices as legacy API)
_LANDMARK_NAMES: Dict[int, str] = {
    11: "left_shoulder",
    12: "right_shoulder",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    15: "left_wrist",
    16: "right_wrist",
}

_PADDING_RATIO = 0.15
_DEFAULT_MODEL = str(Path(__file__).resolve().parent.parent / "pose_landmarker_full.task")


class MediaPipePoseEstimator(IPoseEstimator):
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_path: str = _DEFAULT_MODEL,
    ) -> None:
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=5,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def estimate_pose(
        self, image_data: bytes, bbox: Optional[BoundingBox] = None
    ) -> Optional[HumanBodyStats]:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Failed to decode image for pose estimation")
            return None

        h, w = img.shape[:2]
        crop = self._crop(img, bbox, w, h) if bbox is not None else img

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]
        keypoints: Dict[str, Keypoint] = {}
        for idx, name in _LANDMARK_NAMES.items():
            lm = landmarks[idx]
            keypoints[name] = Keypoint(x=lm.x, y=lm.y, visibility=lm.visibility)

        effective_bbox = bbox if bbox is not None else BoundingBox(0, 0, float(w), float(h), 1.0)
        return HumanBodyStats(bounding_box=effective_bbox, keypoints=keypoints, is_full_body=False)

    @staticmethod
    def _crop(img: np.ndarray, bbox: BoundingBox, w: int, h: int) -> np.ndarray:
        pad_x = bbox.width * _PADDING_RATIO
        pad_y = bbox.height * _PADDING_RATIO
        x1 = max(0, int(bbox.x1 - pad_x))
        y1 = max(0, int(bbox.y1 - pad_y))
        x2 = min(w, int(bbox.x2 + pad_x))
        y2 = min(h, int(bbox.y2 + pad_y))
        return img[y1:y2, x1:x2]

    def close(self) -> None:
        self._landmarker.close()
