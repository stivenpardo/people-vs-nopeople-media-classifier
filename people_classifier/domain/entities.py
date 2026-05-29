from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from enum import Enum


class MediaType(Enum):
    IMAGE = "image"
    VIDEO = "video"


class ClassificationLabel(Enum):
    PEOPLE = "people"
    NO_PEOPLE = "nopeople"


@dataclass
class MediaFile:
    path: Path
    media_type: MediaType
    size_bytes: int = 0


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def aspect_ratio(self) -> float:
        """height / width — >1.0 means taller than wide (standing person)."""
        return self.height / self.width if self.width > 0 else 0.0

    def area_ratio(self, image_width: int, image_height: int) -> float:
        total = image_width * image_height
        return (self.width * self.height) / total if total > 0 else 0.0


@dataclass
class Keypoint:
    x: float
    y: float
    visibility: float  # MediaPipe visibility score 0–1


@dataclass
class HumanBodyStats:
    bounding_box: BoundingBox
    keypoints: Dict[str, Keypoint]
    is_full_body: bool
    rejection_reason: Optional[str] = None


@dataclass
class ClassificationResult:
    file_path: Path
    label: ClassificationLabel
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)
