"""
Local entry point for drone detection using the pre-trained YOLOv8x weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "weight" / "best.pt"
DEFAULT_VIDEO = ROOT / "test" / "pexels-joseph-redfield-8459631 (1080p).mp4"
DEFAULT_IMAGE = ROOT / "test_batch" / "val_batch0_labels.jpg"
HF_REPO = "doguilmak/Drone-Detection-YOLOv8x"
HF_WEIGHT_FILE = "weight/best.pt"


def get_weights(weights_path: Path) -> Path:
    if weights_path.exists():
        return weights_path

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading weights to {weights_path} ...")
    downloaded = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_WEIGHT_FILE,
        local_dir=weights_path.parent.parent,
    )
    return Path(downloaded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run drone detection with YOLOv8x.")
    parser.add_argument(
        "--source",
        type=str,
        default=str(DEFAULT_VIDEO),
        help="Image, video, or directory path for detection/tracking. "
        "Use a webcam index like 0 to stream from a local camera.",
    )
    parser.add_argument(
        "--webcam",
        type=int,
        default=None,
        help="Shortcut for --source using webcam index (e.g. --webcam 0).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to the trained model weights.",
    )
    parser.add_argument(
        "--mode",
        choices=("detect", "track"),
        default="track",
        help="Use detect for images, track for videos.",
    )
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold.")
    parser.add_argument(
        "--project",
        type=Path,
        default=ROOT / "runs" / "detect",
        help="Directory where outputs are saved.",
    )
    parser.add_argument("--name", default="predict", help="Run name inside project folder.")
    parser.add_argument("--show", action="store_true", help="Display results in a window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    is_webcam = args.webcam is not None
    if is_webcam:
        source: Path | int = args.webcam
    else:
        source = Path(args.source).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source}")

    weights = get_weights(args.weights.resolve())
    model = YOLO(str(weights))

    common_kwargs = {
        "source": source if is_webcam else str(source),
        "conf": args.conf,
        "iou": args.iou,
        "project": str(args.project),
        "name": args.name,
        "save": True,
        "show": args.show,
    }

    is_video = is_webcam or (
        isinstance(source, Path) and source.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    )
    if is_video:
        common_kwargs["stream"] = True

    if args.mode == "track":
        results = model.track(**common_kwargs)
    else:
        results = model.predict(**common_kwargs)

    save_dir = None
    for result in results:
        save_dir = Path(result.save_dir)
    if save_dir is None:
        save_dir = args.project / args.name
    print(f"Done. Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
