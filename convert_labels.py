import os

# Mapping tablosu (hangisi hangi ID'ye dönüşecek?)
baret_mapping = {
    0: 0,  # helmet
}

telefon_mapping = {
    1: 0  # phone
}

sigara_mapping = {
    2: 0  # cigarette
}

def update_labels_recursive(base_dir, mapping):
    print(f"🔄 İşleniyor: {base_dir}")
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    old_id = int(parts[0])
                    new_id = mapping.get(old_id, old_id)
                    parts[0] = str(new_id)
                    new_lines.append(" ".join(parts) + "\n")

                with open(file_path, "w") as f:
                    f.writelines(new_lines)
    print(f"✅ Tamamlandı: {base_dir}\n")

# Her sınıf için uygula
update_labels_recursive(r"C:\\Users\\memoc\\Desktop\\hardhat\\valid\\labels", baret_mapping)
update_labels_recursive(r"C:\\Users\\memoc\\Desktop\\phone\\train\\labels", telefon_mapping)
update_labels_recursive(r"C:\\Users\\memoc\\Desktop\\cigarette\\train\\labels", sigara_mapping)