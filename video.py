import cv2
import os

image_folder = "dataset/train/images"
output_video = "dataset_video.mp4"
fps = 30

images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")])

first_frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_frame.shape

video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

for img_name in images:
    img_path = os.path.join(image_folder, img_name)
    frame = cv2.imread(img_path)
    video.write(frame)

video.release()
print(f"Video oluşturuldu: {output_video}")