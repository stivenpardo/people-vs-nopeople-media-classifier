import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from domain.entities import ClassificationLabel, ClassificationResult, MediaType
from domain.interfaces import IPersonDetector, IPoseEstimator, IVideoProcessor

logger = logging.getLogger(__name__)


class ClassifierService:
    def __init__(
        self,
        detector: IPersonDetector,
        pose_estimator: IPoseEstimator,
        video_processor: IVideoProcessor,
        min_bbox_area_ratio: float = 0.05,
        min_bbox_aspect_ratio: float = 1.2,
        full_body_keypoint_threshold: float = 0.5,
        required_keypoints: Optional[List[str]] = None,
        video_sample_fps: float = 1.0,
    ) -> None:
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.video_processor = video_processor
        self.min_bbox_area_ratio = min_bbox_area_ratio
        self.min_bbox_aspect_ratio = min_bbox_aspect_ratio
        self.keypoint_threshold = full_body_keypoint_threshold
        self.required_keypoints: List[str] = required_keypoints or [
            "left_ankle", "right_ankle",
            "left_shoulder", "right_shoulder",
            "left_hip", "right_hip",
        ]
        self.video_sample_fps = video_sample_fps

    def classify(
        self,
        file_path: Path,
        media_type: MediaType,
        image_bytes: Optional[bytes] = None,
    ) -> ClassificationResult:
        if media_type == MediaType.IMAGE:
            return self._classify_image(file_path, image_bytes)
        return self._classify_video(file_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify_image(
        self, file_path: Path, image_bytes: Optional[bytes] = None
    ) -> ClassificationResult:
        if image_bytes is None:
            with open(file_path, "rb") as f:
                image_bytes = f.read()

        # Decode once — reused for area calculations and pose estimation
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Cannot decode image: %s", file_path)
            return self._no_people(file_path, "image_decode_failed")

        img_h, img_w = img.shape[:2]
        persons = self.detector.detect_persons(image_bytes)

        if not persons:
            return self._no_people(file_path, "no_person_detected")

        for person in persons:
            bbox = person.bounding_box

            # --- Geometric filters ---
            area_ratio = bbox.area_ratio(img_w, img_h)
            if area_ratio < self.min_bbox_area_ratio:
                logger.debug("%s: person rejected — area_ratio %.3f < %.3f", file_path.name, area_ratio, self.min_bbox_area_ratio)
                continue

            aspect = bbox.aspect_ratio
            if aspect < self.min_bbox_aspect_ratio:
                logger.debug("%s: person rejected — aspect_ratio %.2f < %.2f", file_path.name, aspect, self.min_bbox_aspect_ratio)
                continue

            # --- Pose & keypoint visibility filter ---
            pose = self.pose_estimator.estimate_pose(image_bytes, bbox=bbox)
            if pose is None or not pose.keypoints:
                logger.debug("%s: no pose detected for bbox", file_path.name)
                continue

            missing = [
                kp for kp in self.required_keypoints
                if pose.keypoints.get(kp) is None
                or pose.keypoints[kp].visibility < self.keypoint_threshold
            ]
            if missing:
                logger.debug("%s: missing/low-visibility keypoints: %s", file_path.name, missing)
                continue

            return ClassificationResult(
                file_path=file_path,
                label=ClassificationLabel.PEOPLE,
                confidence=bbox.confidence,
                details={
                    "reason": "full_body_detected",
                    "area_ratio": round(area_ratio, 3),
                    "aspect_ratio": round(aspect, 2),
                },
            )

        return self._no_people(file_path, "no_full_body_passed_checks")

    def _classify_video(self, file_path: Path) -> ClassificationResult:
        frames = self.video_processor.extract_frames(file_path, self.video_sample_fps)
        if not frames:
            return self._no_people(file_path, "no_frames_extracted")

        for frame_bytes in frames:
            result = self._classify_image(file_path, image_bytes=frame_bytes)
            if result.label == ClassificationLabel.PEOPLE:
                result.details["source"] = "video_frame"
                return result

        return self._no_people(file_path, "no_full_body_in_any_frame")

    @staticmethod
    def _no_people(file_path: Path, reason: str) -> ClassificationResult:
        return ClassificationResult(
            file_path=file_path,
            label=ClassificationLabel.NO_PEOPLE,
            confidence=0.0,
            details={"reason": reason},
        )
