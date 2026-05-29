from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

from domain.entities import BoundingBox, HumanBodyStats, MediaFile, MediaType


class IPersonDetector(ABC):
    @abstractmethod
    def detect_persons(self, image_data: bytes) -> List[HumanBodyStats]:
        """Return detected persons with bounding boxes. Keypoints may be empty."""
        pass


class IPoseEstimator(ABC):
    @abstractmethod
    def estimate_pose(
        self, image_data: bytes, bbox: Optional[BoundingBox] = None
    ) -> Optional[HumanBodyStats]:
        """
        Estimate pose for one person.
        When bbox is provided, crops to that region before running MediaPipe
        so multi-person images are handled correctly.
        Returns None if no pose detected.
        """
        pass


class IVideoProcessor(ABC):
    @abstractmethod
    def extract_frames(self, video_path: Path, sample_fps: float) -> List[bytes]:
        """Return JPEG-encoded frame bytes sampled at sample_fps."""
        pass


class IFileSystem(ABC):
    @abstractmethod
    def read_bytes(self, path: Path) -> bytes:
        pass

    @abstractmethod
    def list_media_files(self, directory: Path) -> List[MediaFile]:
        """Walk directory and return MediaFile objects for all supported media."""
        pass

    @abstractmethod
    def copy_file(self, src: Path, dst: Path) -> None:
        pass

    @abstractmethod
    def move_file(self, src: Path, dst: Path) -> None:
        pass

    @abstractmethod
    def ensure_dir(self, path: Path) -> None:
        pass
