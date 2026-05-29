import logging
from pathlib import Path
from typing import List

import cv2

from domain.interfaces import IVideoProcessor

logger = logging.getLogger(__name__)


class OpenCVVideoProcessor(IVideoProcessor):
    def extract_frames(self, video_path: Path, sample_fps: float = 1.0) -> List[bytes]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0
        frame_interval = max(1, int(round(video_fps / sample_fps)))

        frames: List[bytes] = []
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if ok:
                        frames.append(buf.tobytes())
                frame_count += 1
        finally:
            cap.release()

        logger.debug("Extracted %d frames from %s", len(frames), video_path.name)
        return frames
