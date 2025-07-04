from PIL import Image
import os
import math


folder_path = "fotograf"
image_files = [
    "img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg",
    "img5.jpg", "img6.jpg", "img7.jpg", "img8.jpg",
    "img9.jpg", "img10.jpg", "img11.jpg", "img12.jpg",
    "img13.jpg", "img14.jpg", "img15.jpg", "img16.jpg"
]

images = [Image.open(os.path.join(folder_path, img)) for img in image_files]

w, h = images[0].size
images = [img.resize((w, h)) for img in images]

grid_size = int(math.ceil(math.sqrt(len(images))))

new_img = Image.new('RGB', (w * grid_size, h * grid_size), color=(255, 255, 255))

for index, img in enumerate(images):
    x = (index % grid_size) * w
    y = (index // grid_size) * h
    new_img.paste(img, (x, y))

output_path = "birlesik_fotograf.png"
new_img.save(output_path)
print(f"Birleşik fotoğraf kaydedildi: {output_path}")