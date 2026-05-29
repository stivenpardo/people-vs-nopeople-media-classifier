import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow running as `python interfaces/cli.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from application.classifier_service import ClassifierService
from application.file_organizer import FileOrganizer
from domain.entities import ClassificationLabel, ClassificationResult, MediaFile
from infrastructure.config_loader import AppConfig
from infrastructure.file_system import LocalFileSystem
from infrastructure.mediapipe_pose import MediaPipePoseEstimator
from infrastructure.video_processor import OpenCVVideoProcessor
from infrastructure.yolo_detector import YOLOPersonDetector


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _classify_one(
    media_file: MediaFile, classifier: ClassifierService
) -> ClassificationResult:
    try:
        return classifier.classify(media_file.path, media_file.media_type)
    except Exception as exc:
        logging.getLogger(__name__).error("Error classifying %s: %s", media_file.path, exc)
        return ClassificationResult(
            file_path=media_file.path,
            label=ClassificationLabel.NO_PEOPLE,
            confidence=0.0,
            details={"error": str(exc)},
        )


def main(config_path: str = "config.yaml") -> None:
    cfg = AppConfig.from_yaml(config_path)
    _setup_logging(cfg.log_level)
    logger = logging.getLogger(__name__)

    # --- Wire up the object graph ---
    fs = LocalFileSystem()
    detector = YOLOPersonDetector(
        confidence_threshold=cfg.confidence_threshold,
        device=cfg.device,
    )
    pose_est = MediaPipePoseEstimator(
        min_detection_confidence=cfg.full_body_keypoint_threshold,
    )
    video_proc = OpenCVVideoProcessor()

    classifier = ClassifierService(
        detector=detector,
        pose_estimator=pose_est,
        video_processor=video_proc,
        min_bbox_area_ratio=cfg.min_human_bbox_area_ratio,
        min_bbox_aspect_ratio=cfg.min_bbox_aspect_ratio,
        full_body_keypoint_threshold=cfg.full_body_keypoint_threshold,
        required_keypoints=cfg.min_body_parts,
        video_sample_fps=cfg.video_sample_fps,
    )
    organizer = FileOrganizer(fs, cfg.operation)

    # --- Discover media files ---
    media_files = fs.list_media_files(cfg.input_folder)
    logger.info("Found %d media files in %s", len(media_files), cfg.input_folder)

    if not media_files:
        logger.warning("No supported media files found. Exiting.")
        return

    # --- Parallel classification ---
    results: dict[Path, ClassificationResult] = {}
    with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
        future_map = {
            executor.submit(_classify_one, mf, classifier): mf
            for mf in media_files
        }
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Classifying"):
            mf = future_map[future]
            results[mf.path] = future.result()

    # --- Organise output ---
    organizer.organize(results, cfg.output_folder)

    # --- Summary ---
    people = sum(1 for r in results.values() if r.label == ClassificationLabel.PEOPLE)
    no_people = len(results) - people
    logger.info("Done. people=%d  nopeople=%d  total=%d", people, no_people, len(results))
    print(f"\nSummary  →  people: {people}  |  nopeople: {no_people}  |  total: {len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify media files into people / nopeople.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML (default: config.yaml)")
    args = parser.parse_args()
    main(args.config)
