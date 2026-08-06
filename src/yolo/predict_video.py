import argparse
import time
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO 영상 객체 탐지 실험"
    )

    parser.add_argument(
        "--source",
        default="images/people-walking.mp4",
        help="입력 영상 경로",
    )
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="사용할 YOLO 모델",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="입력 이미지 크기",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="실행 장치",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="박스가 그려진 결과 영상 저장",
    )

    args = parser.parse_args()

    video_path = Path(args.source)

    if not video_path.exists():
        raise FileNotFoundError(
            f"영상 파일을 찾을 수 없습니다: {video_path}"
        )

    model = YOLO(args.model)

    run_name = (
        f"{Path(args.model).stem}"
        f"_conf{str(args.conf).replace('.', 'p')}"
        f"_img{args.imgsz}"
        f"_{args.device}"
    )

    start_time = time.time()
    frame_count = 0
    total_boxes = 0
    total_persons = 0

    results = model.predict(
        source=str(video_path),
        stream=True,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
        save=args.save,
        project="runs/experiments",
        name=run_name,
        exist_ok=True,
    )

    for result in results:
        frame_count += 1
        boxes = result.boxes
        total_boxes += len(boxes)

        person_count = sum(
            1
            for box in boxes
            if model.names[int(box.cls.item())] == "person"
        )

        total_persons += person_count

        print(
            f"frame {frame_count:4d}: "
            f"전체 객체 {len(boxes):2d}개, "
            f"person {person_count:2d}명"
        )

    elapsed_time = time.time() - start_time

    if frame_count == 0:
        raise RuntimeError("처리된 프레임이 없습니다.")

    fps = frame_count / elapsed_time
    average_boxes = total_boxes / frame_count
    average_persons = total_persons / frame_count

    print("\n===== 실험 결과 =====")
    print(f"모델: {args.model}")
    print(f"Confidence: {args.conf}")
    print(f"이미지 크기: {args.imgsz}")
    print(f"실행 장치: {args.device}")
    print(f"총 프레임 수: {frame_count}")
    print(f"총 검출 박스 수: {total_boxes}")
    print(f"프레임당 평균 박스 수: {average_boxes:.2f}")
    print(f"프레임당 평균 사람 수: {average_persons:.2f}")
    print(f"총 실행 시간: {elapsed_time:.2f}초")
    print(f"평균 처리 속도: {fps:.1f} FPS")

    if args.save:
        print(f"결과 저장 폴더: runs/experiments/{run_name}")


if __name__ == "__main__":
    main()