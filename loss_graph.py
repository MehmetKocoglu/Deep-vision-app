import matplotlib.pyplot as plt
import matplotlib as mpl
import re

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 15

file_paths = {
    "main300.txt": "Tümü",
    "hardhat300.txt": "Baret",
    "phone300.txt": "Telefon",
    "cigarette300.txt": "Sigara",
}

def extract_losses(file_path):
    box_losses, cls_losses, dfl_losses = [], [], []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        if re.match(r"^\s*\d+/\d+", line):
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            if len(numbers) >= 6:
                try:
                    box_losses.append(float(numbers[3]))
                    cls_losses.append(float(numbers[4]))
                    dfl_losses.append(float(numbers[5]))
                except:
                    continue
    return box_losses, cls_losses, dfl_losses

# Gruplama
models_300 = {name: extract_losses(path) for path, name in file_paths.items() if "300" in path}

# 1. Box Loss - 300 Epoch
plt.figure(figsize=(10, 5))
for name, losses in models_300.items():
    plt.plot(losses[0], label=name)
plt.title("Konumlandırma Kaybı")
plt.xlabel("Epoch Sayısı")
plt.ylabel("Kayıp")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Class Loss - 300 Epoch
plt.figure(figsize=(10, 5))
for name, losses in models_300.items():
    plt.plot(losses[1], label=name)
plt.title("Sınıflandırma Kaybı")
plt.xlabel("Epoch Sayısı")
plt.ylabel("Kayıp")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. DFL Loss - 300 Epoch
plt.figure(figsize=(10, 5))
for name, losses in models_300.items():
    plt.plot(losses[2], label=name)
plt.title("Dağılım Odaklı Kayıp")
plt.xlabel("Epoch Sayısı")
plt.ylabel("Kayıp")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
