import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch

video_path = "dataset_video.mp4"
model_path = "best.pt"
output_file = "tracker.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO(model_path).to(device)

trackers = {
    0: DeepSort(max_age=30, n_init=3, max_iou_distance=0.7, max_cosine_distance=0.3),
    1: DeepSort(max_age=30, n_init=3, max_iou_distance=0.7, max_cosine_distance=0.3),
    2: DeepSort(max_age=30, n_init=3, max_iou_distance=0.7, max_cosine_distance=0.3),
}

id_offsets = {
    0: 0,
    1: 10000,
    2: 20000
}

cap = cv2.VideoCapture(video_path)

frame_id = 1

with open(output_file, "w") as f:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        class_detections = {0: [], 1: [], 2: []}

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if cls not in class_detections:
                continue

            if conf < 0.4:
                continue

            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                continue

            class_detections[cls].append(([x1, y1, w, h], conf, cls))

        for cls in class_detections:

            detections = class_detections[cls]

            if len(detections) == 0:
                continue

            tracks = trackers[cls].update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = int(track.track_id) + id_offsets[cls]

                l, t_, r, b = track.to_ltrb()

                w = r - l
                h = b - t_

                if w <= 0 or h <= 0:
                    continue

                f.write(f"{frame_id},{track_id},{int(l)},{int(t_)},{int(w)},{int(h)},1,-1,-1,-1\n")

        frame_id += 1

cap.release()

print("✅ Tracker output hazır (optimize edilmiş)")