import cv2
from ultralytics import YOLO


def main() -> None:
    # YOLO nano 모델 불러오기
    model = YOLO("yolo26n.pt")

    # 0번 웹캠 열기
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "웹캠을 열 수 없습니다. "
            "ls /dev/video* 명령으로 장치를 확인하세요."
        )

    while True:
        # 웹캠에서 한 프레임 읽기
        ok, frame = cap.read()

        if not ok:
            print("웹캠 영상을 읽지 못했습니다.")
            break

        # 현재 프레임에 YOLO 객체 탐지 실행
        results = model(
            frame,
            conf=0.25,
            device="cuda",
            verbose=False,
        )

        # 객체 이름과 박스가 표시된 화면 생성
        annotated_frame = results[0].plot()

        # 화면 출력
        cv2.imshow(
            "YOLO webcam - press q to quit",
            annotated_frame,
        )

        # q를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
