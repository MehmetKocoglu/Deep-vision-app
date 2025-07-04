import os
import cv2
import random
import numpy as np
import albumentations as A
from shutil import copyfile


# Augmentasyon fonksiyonu
def augment_image(image, label, output_image_path, output_label_path):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),  # Yatay çevirme
        A.RandomBrightnessContrast(p=0.2),  # Parlaklık ve kontrast değişimi
        A.RandomGamma(p=0.2),  # Gamma düzeltmesi
        A.Rotate(limit=30, p=0.5),  # Döndürme
        A.Resize(640, 640, p=1.0)  # Boyutlandırma
    ])

    augmented = transform(image=image)
    augmented_image = augmented['image']

    cv2.imwrite(output_image_path, augmented_image)
    copyfile(label, output_label_path)


# Augmentasyon işlemi için verileri yükleme
def augment_class_images(input_image_folder, input_label_folder, output_image_folder, output_label_folder,
                         num_augmentations=5):
    image_files = os.listdir(input_image_folder)

    for image_file in image_files:
        image_path = os.path.join(input_image_folder, image_file)
        label_path = os.path.join(input_label_folder, image_file.replace('.jpg', '.txt'))

        if not os.path.exists(label_path):
            continue

        for i in range(num_augmentations):
            augmented_image_path = os.path.join(output_image_folder, f"aug_{i}_{image_file}")
            augmented_label_path = os.path.join(output_label_folder, f"aug_{i}_{image_file.replace('.jpg', '.txt')}")

            image = cv2.imread(image_path)
            augment_image(image, label_path, augmented_image_path, augmented_label_path)



# augment_class_images("C://Users//memoc//Desktop//train//images", "C://Users//memoc//Desktop//train//labels", "C://Users//memoc//Desktop//train//cigarette_aug_images", "C://Users//memoc//Desktop//train//cigarette_aug_labels")
augment_class_images("C://Users//memoc//Desktop//cigarette//train//images", "C://Users//memoc//Desktop//cigarette//train//labels", "C://Users//memoc//Desktop//cigarette//cigarette_aug_images", "C://Users//memoc//Desktop//cigarette//cigarette_aug_labels")