import os
import cv2

# Dataset klasör yolları
image_folder = "dataset/train/images"
label_folder = "dataset/train/labels"
output_file = "gt.txt"

# IOU eşik değeri (isteğe göre ayarlanabilir)
IOU_THRESHOLD = 0.3


def iou(boxA, boxB):
    # box = [x, y, w, h]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou_value = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou_value


with open(output_file, "w") as f_out:
    # Tüm resimler için döngü
    images = sorted(os.listdir(image_folder))
    for img_name in images:
        frame_id = int(os.path.splitext(img_name)[0])  # frame id = resim adı
        img_path = os.path.join(image_folder, img_name)
        height, width = cv2.imread(img_path).shape[:2]

        label_path = os.path.join(label_folder, img_name.replace(".jpg", ".txt"))
        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                # YOLO formatı: class x_center y_center w h (norm)
                cls, x_center, y_center, w, h = map(float, line.strip().split())
                x = (x_center - w / 2) * width
                y = (y_center - h / 2) * height
                w *= width
                h *= height
                tid = int(cls)  # ground truth için sınıf ID kullanabiliriz
                f_out.write(f"{frame_id},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

print(f"Ground truth oluşturuldu: {output_file}")