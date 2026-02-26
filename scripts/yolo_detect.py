from pathlib import Path
import cv2
import time
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent

# Input options: "webcam", "image", "video"
INPUT_TYPE = "webcam"  # change to "video" or "webcam"

# Paths for image/video input (relative to project)
IMAGE_PATH = BASE_DIR / "images" / "phone.jpg"

#just create a video folder because i dont have sample videos
VIDEO_PATH = BASE_DIR / "videos" / "test_video.mp4"    
OUTPUT_VIDEO_PATH = BASE_DIR / "videos" / "output.mp4"  

# Path to YOLO model
MODEL_PATH = BASE_DIR / "models" / "yolo26n.pt"

CONFIDENCE_THRESHOLD = 0.5


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# Load model
model = YOLO(str(MODEL_PATH))


def draw_boxes(frame, results):
    for box, cls_id, conf in zip(
        results[0].boxes.xyxy,
        results[0].boxes.cls,
        results[0].boxes.conf
    ):
        x1, y1, x2, y2 = map(int, box)
        class_name = model.names[int(cls_id)]
        confidence = float(conf)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{class_name} {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    return frame


if INPUT_TYPE == "image":
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")

    frame = cv2.imread(str(IMAGE_PATH))
    start_time = time.time()
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    frame = draw_boxes(frame, results)
    end_time = time.time()
    print(f"Inference Time: {end_time - start_time:.4f} seconds")
    cv2.imshow("YOLO Image Inference", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


elif INPUT_TYPE == "video":
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found at {VIDEO_PATH}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        str(OUTPUT_VIDEO_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_input,
        (width, height)
    )

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        frame = draw_boxes(frame, results)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow("YOLO Video Inference", frame)

        if cv2.waitKey(1) == 27: 
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved at {OUTPUT_VIDEO_PATH}")

elif INPUT_TYPE == "webcam":
    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        frame = draw_boxes(frame, results)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO Webcam Inference", frame)
        if cv2.waitKey(1) == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()


elif INPUT_TYPE == "raspi_cam":
    # For Raspberry Pi Camera v2 / HQ camera using OpenCV VideoCapture
    # Make sure libcamera is installed and camera is enabled
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # 0 is default camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # you can set resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    prev_time = time.time()

    if not cap.isOpened():
        raise RuntimeError("Could not open Raspberry Pi camera")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        frame = draw_boxes(frame, results)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO Raspberry Pi Camera", frame)
        if cv2.waitKey(1) == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    raise ValueError("INPUT_TYPE must be 'image', 'video', 'webcam', 'raspi_cam'")