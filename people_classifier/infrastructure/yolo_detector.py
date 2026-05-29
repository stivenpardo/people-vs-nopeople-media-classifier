import logging
import threading
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

from domain.entities import BoundingBox, HumanBodyStats
from domain.interfaces import IPersonDetector

logger = logging.getLogger(__name__)

_PERSON_CLASS_ID = 0


class YOLOPersonDetector(IPersonDetector):
    def __init__(self, confidence_threshold: float = 0.6, device: str = "cpu") -> None:
        self._model = YOLO("yolov8n.pt")
        self._conf = confidence_threshold
        self._device = device
        # ultralytics fuses Conv+BN on the first inference call; concurrent threads
        # racing through that fuse step cause AttributeError('bn'). Serialize calls.
        self._lock = threading.Lock()

    def detect_persons(self, image_data: bytes) -> List[HumanBodyStats]:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Failed to decode image for person detection")
            return []

        with self._lock:
            results = self._model(
                img,
                conf=self._conf,
                device=self._device,
                classes=[_PERSON_CLASS_ID],
                verbose=False,
            )[0]

        persons: List[HumanBodyStats] = []
        for box in results.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)
            persons.append(HumanBodyStats(bounding_box=bbox, keypoints={}, is_full_body=False))

        return persons
