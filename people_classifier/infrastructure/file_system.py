import shutil
from pathlib import Path
from typing import List

from domain.entities import MediaFile, MediaType
from domain.interfaces import IFileSystem

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".3gp"}


class LocalFileSystem(IFileSystem):
    def read_bytes(self, path: Path) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def list_media_files(self, directory: Path) -> List[MediaFile]:
        files: List[MediaFile] = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext in _IMAGE_EXT:
                media_type = MediaType.IMAGE
            elif ext in _VIDEO_EXT:
                media_type = MediaType.VIDEO
            else:
                continue
            files.append(MediaFile(path=path, media_type=media_type, size_bytes=path.stat().st_size))
        return files

    def copy_file(self, src: Path, dst: Path) -> None:
        shutil.copy2(src, dst)

    def move_file(self, src: Path, dst: Path) -> None:
        shutil.move(str(src), str(dst))

    def ensure_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
