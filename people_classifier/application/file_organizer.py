import hashlib
import logging
from pathlib import Path
from typing import Dict, Set

from domain.entities import ClassificationLabel, ClassificationResult
from domain.interfaces import IFileSystem

logger = logging.getLogger(__name__)

_HASH_CHUNK = 1 << 20  # 1 MiB


class FileOrganizer:
    def __init__(self, fs: IFileSystem, operation: str = "copy") -> None:
        assert operation in ("copy", "move"), "operation must be 'copy' or 'move'"
        self.fs = fs
        self.operation = operation
        self._seen_hashes: Set[str] = set()

    def organize(
        self, results: Dict[Path, ClassificationResult], output_root: Path
    ) -> None:
        people_dir = output_root / ClassificationLabel.PEOPLE.value
        nopeople_dir = output_root / ClassificationLabel.NO_PEOPLE.value
        self.fs.ensure_dir(people_dir)
        self.fs.ensure_dir(nopeople_dir)

        duplicates = 0
        for src_path, res in results.items():
            content_hash = self._sha256(src_path)
            if content_hash in self._seen_hashes:
                logger.info("Skipping duplicate %s", src_path.name)
                duplicates += 1
                continue
            self._seen_hashes.add(content_hash)

            dst_dir = people_dir if res.label == ClassificationLabel.PEOPLE else nopeople_dir
            dst_path = self._unique_dst(dst_dir, src_path.name)

            try:
                if self.operation == "copy":
                    self.fs.copy_file(src_path, dst_path)
                else:
                    self.fs.move_file(src_path, dst_path)
                logger.info("%sd %s → %s", self.operation.capitalize(), src_path.name, dst_path)
            except Exception as exc:
                logger.error("Failed to %s %s: %s", self.operation, src_path, exc)

        if duplicates:
            logger.info("Skipped %d duplicate file(s)", duplicates)

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(_HASH_CHUNK):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _unique_dst(dst_dir: Path, filename: str) -> Path:
        candidate = dst_dir / filename
        if not candidate.exists():
            return candidate
        stem, suffix = Path(filename).stem, Path(filename).suffix
        counter = 1
        while True:
            candidate = dst_dir / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1
