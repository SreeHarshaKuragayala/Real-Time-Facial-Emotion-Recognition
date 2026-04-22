import os
import cv2
from tqdm import tqdm

# Update these paths
base_dirs = ['F:/Python/Training/DATASET/train', 'F:/Python/Training/DATASET/test']
img_size = 224

for base_dir in base_dirs:
    for emotion_folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, emotion_folder)
        if not os.path.isdir(folder_path):
            continue
        for img_name in tqdm(os.listdir(folder_path), desc=f"Processing {emotion_folder}"):
            img_path = os.path.join(folder_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, (img_size, img_size))  # Resize
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # Convert to RGB
                cv2.imwrite(img_path, img)
            except Exception as e:
                print(f"Failed {img_path}: {e}")
