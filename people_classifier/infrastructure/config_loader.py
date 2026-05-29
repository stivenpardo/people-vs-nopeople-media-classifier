from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field, field_validator


class AppConfig(BaseModel):
    input_folder: Path
    output_folder: Path
    operation: str = "copy"
    confidence_threshold: float = Field(0.6, ge=0.0, le=1.0)
    full_body_keypoint_threshold: float = Field(0.5, ge=0.0, le=1.0)
    min_body_parts: List[str] = [
        "left_ankle", "right_ankle",
        "left_shoulder", "right_shoulder",
        "left_hip", "right_hip",
    ]
    min_human_bbox_area_ratio: float = Field(0.05, ge=0.0, le=1.0)
    min_bbox_aspect_ratio: float = Field(1.2, ge=0.0)
    video_sample_fps: float = Field(1.0, gt=0.0)
    device: str = "cpu"
    num_workers: int = Field(4, ge=1)
    log_level: str = "INFO"

    @field_validator("operation")
    @classmethod
    def _validate_operation(cls, v: str) -> str:
        if v not in ("copy", "move"):
            raise ValueError("operation must be 'copy' or 'move'")
        return v

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        if v not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        return v

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
