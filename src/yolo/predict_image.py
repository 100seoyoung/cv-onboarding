from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    image_path = Path("images/test.jpg")

    if not image_path.exists():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    model = YOLO("yolo26n.pt")

    results = model.predict(
        source=str(image_path),
        conf=0.25,
        device="cuda",
        save=True,
    )

    for box in results[0].boxes:
        class_id = int(box.cls.item())
        class_name = model.names[class_id]
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print(
            f"{class_name:12s} "
            f"conf={confidence:.2f} "
            f"box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
        )


if __name__ == "__main__":
    main()