import os
import shutil
from sklearn.model_selection import train_test_split

# 以8:1:1分割成訓練集、驗證集、測試集
def create_sub_dataset(original_images_root, original_labels_root, new_train_images_root, new_train_labels_root, valid_images_root, valid_labels_root, test_images_root, test_labels_root, test_size=0.1, valid_size=0.1):
    # 圖片以及標籤的位址
    original_images = sorted([os.path.join(original_images_root, f) for f in os.listdir(original_images_root) if f.endswith('.jpg')])
    original_labels = sorted([os.path.join(original_labels_root, f) for f in os.listdir(original_labels_root) if f.endswith('.png')])

    # 只保留有對應標籤的圖片
    images = [img for img in original_images if os.path.exists(os.path.join(original_labels_root, os.path.basename(img).replace('.jpg', '.png')))]
    labels = [lbl for lbl in original_labels if os.path.exists(os.path.join(original_images_root, os.path.basename(lbl).replace('.png', '.jpg')))]

    # 總的測試集比例是 test_size
    # 剩下的資料用來分割成訓練集和驗證集
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size, random_state=42)

    # 剩下的資料中，再以 valid_size / (1 - test_size) 的比例來分割驗證集和訓練集
    valid_size_adjusted = valid_size / (1 - test_size)
    train_images, valid_images, train_labels, valid_labels = train_test_split(train_images, train_labels, test_size=valid_size_adjusted, random_state=42)

    # 創建新的訓練集、驗證集與測試集資料夾
    os.makedirs(new_train_images_root, exist_ok=True)
    os.makedirs(new_train_labels_root, exist_ok=True)
    os.makedirs(valid_images_root, exist_ok=True)
    os.makedirs(valid_labels_root, exist_ok=True)
    os.makedirs(test_images_root, exist_ok=True)
    os.makedirs(test_labels_root, exist_ok=True)

    # 將訓練集資料複製到新的資料夾
    for image in train_images:
        shutil.copy(image, os.path.join(new_train_images_root, os.path.basename(image)))
    for label in train_labels:
        shutil.copy(label, os.path.join(new_train_labels_root, os.path.basename(label)))

    # 將驗證集資料轉移
    for image in valid_images:
        shutil.copy(image, os.path.join(valid_images_root, os.path.basename(image)))
    for label in valid_labels:
        shutil.copy(label, os.path.join(valid_labels_root, os.path.basename(label)))
    
    # 將測試集資料轉移
    for image in test_images:
        shutil.copy(image, os.path.join(test_images_root, os.path.basename(image)))
    for label in test_labels:
        shutil.copy(label, os.path.join(test_labels_root, os.path.basename(label)))

original_images_root = "/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Training_dataset_original/images"  # 原始訓練集圖片
original_labels_root = "/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Training_dataset_original/labels"  # 原始訓練集標籤
new_train_images_root = "/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Training_dataset/images"  # 新的訓練集圖片
new_train_labels_root = "/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Training_dataset/labels"  # 新的訓練集標籤
valid_images_root = "/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Validation_dataset/images"  # 驗證集圖片
valid_labels_root = "/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Validation_dataset/labels"  # 驗證集標籤
test_images_root = "/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Testing_dataset/images"  # 測試集圖片
test_labels_root = "/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Testing_dataset/labels"  # 測試集標籤

create_sub_dataset(original_images_root, original_labels_root, new_train_images_root, new_train_labels_root, valid_images_root, valid_labels_root, test_images_root, test_labels_root)
