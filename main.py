import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd
from ultralytics import YOLO
import cvzone
import threading
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import torch


class VideoApp:
    def __init__(self, root, video_source, model_path, coco_path):
        self.root = root
        self.root.title("Live Cam")
        self.root.geometry("1100x580")

        self.video_source = video_source
        self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
        self.fps = self.cap.set(cv2.CAP_PROP_FPS, 144)

        self.canvas = tk.Canvas(self.root, width=1020, height=500)
        self.canvas.pack()

        self.btn_play = ttk.Button(self.root, text="Play", command=self.play)
        self.btn_play.place(x=100, y=515)

        self.btn_pause = ttk.Button(self.root, text="Pause", command=self.pause)
        self.btn_pause.place(x=200, y=515)

        self.btn_stop = ttk.Button(self.root, text="Stop", command=self.stop)
        self.btn_stop.place(x=300, y=515)

        self.coord_label = tk.Label(self.root, text="X: 0, Y: 0", fg="black")
        self.coord_label.place(x=50, y=545)

        self.is_playing = False
        self.model = YOLO('best300.pt').to("cuda")
        self.tracker = DeepSort(max_age=30)

        self.class_list = self.load_class_list(coco_path)

        self.current_frame = None
        self.tracked_ids = {}

        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        self.process_thread = threading.Thread(target=self.process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()

        self.canvas.bind("<Motion>", self.mouse_move)

    def load_class_list(self, coco_path):
        try:
            with open(coco_path, 'r') as coco_file:
                class_list = [line.strip() for line in coco_file.readlines() if line.strip()]
            return class_list
        except FileNotFoundError:
            print(f"Error: Objects file not found at '{coco_path}'")
            return []

    def play(self):
        self.is_playing = True

    def pause(self):
        self.is_playing = False

    def stop(self):
        self.is_playing = False
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def capture_frames(self):
        while self.cap.isOpened():
            if self.is_playing:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.current_frame = frame
            else:
                self.current_frame = None

    def process_frames(self):
        while True:
            if self.is_playing and self.current_frame is not None:
                frame = self.current_frame.copy()

                frame = cv2.resize(frame, (1020, 500))

                results = self.model.predict(frame, device="cuda")
                a = results[0].boxes.data
                px = pd.DataFrame(a.cpu().numpy()).astype("float") if torch.cuda.is_available() else pd.DataFrame(
                    a).astype("float")

                frame_height, frame_width, _ = frame.shape
                person_list = []

                for index, row in px.iterrows():
                    x1 = int(row[0])
                    y1 = int(row[1])
                    x2 = int(row[2])
                    y2 = int(row[3])
                    confidence = float(row[4])
                    class_id = int(row[5])

                    if class_id < len(self.class_list) and self.class_list[
                        class_id] in ["Phone", "Cigarette", "Hardhat"]:
                        # YOLO'nun çizdiği kutuları ekrana çizdiriyoruz
                        label = self.class_list[class_id]
                        color = (0, 255, 0)

                        if label == "Phone":
                            color = (0, 255, 0)
                        elif label == "Cigarette":
                            color = (0, 0, 255)
                        elif label == "Hardhat":
                            color = (255, 255, 0)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cvzone.putTextRect(frame, f'{label} {index}', (x1, y1), 0.7, 1, colorR=color, colorT=(0, 0, 0))

                        person_list.append(([x1, y1, x2, y2], confidence, 0))

                        # DeepSORT Takip Algoritması
                tracks = self.tracker.update_tracks(person_list, frame=frame)
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)

                    if track_id not in self.tracked_ids:
                        self.tracked_ids[track_id] = True

                lost_tracks = [track for track in tracks if not track.is_confirmed()]
                for lost_track in lost_tracks:
                    track_id = lost_track.track_id
                    if track_id in self.tracked_ids:
                        del self.tracked_ids[track_id]

                self.root.after(0, self.update_gui)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.canvas.photo = photo

            self.root.update_idletasks()

    def mouse_move(self, event):
        x, y = event.x, event.y
        self.coord_label.config(text=f"X: {x}, Y: {y}")

    def update_gui(self):
        self.root.after(0, self.update_gui)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root, video_source="model_test_video.mkv",
                   model_path='best300.pt',
                   coco_path='objects')
    root.mainloop()
