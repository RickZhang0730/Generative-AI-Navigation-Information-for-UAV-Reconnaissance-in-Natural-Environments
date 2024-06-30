import os
from torch.utils.data import DataLoader
from Data_augmentation.dataset_augmentation import GeneralDataset


# 指定 training set 和 validation set 的路徑
tr_root = "/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Training_dataset_divided_expand_combined"
val_root = "/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Validation_dataset"

# 使用 DataLoader 來包裝 training set 和 validation set
# - mode: 設置數據集的模式，設定為 "train "則會做資料擴增
tr_datastet = GeneralDataset(image_root=os.path.join(tr_root, "images"),
                gt_root=os.path.join(tr_root, "labels"),
                img_size=224, mode="train")
val_datastet = GeneralDataset(image_root=os.path.join(val_root, "images"),
                gt_root=os.path.join(val_root, "labels"),
                img_size=224, mode="val")

# 建立DataLoader來加載 training set 和 validation set
batch_size = 12
tr_loader = DataLoader(dataset = tr_datastet, batch_size=batch_size, shuffle=True,
                  num_workers=4, pin_memory=True)
val_loader = DataLoader(dataset = val_datastet, batch_size=batch_size, shuffle=True,
                  num_workers=4, pin_memory=True)

# 可以在這裡確認資料集和加載器是否正確設置
print(f"Training dataset size: {len(tr_datastet)}")
print(f"Validation dataset size: {len(val_datastet)}")

# 測試 DataLoader 加載
for images, gts in tr_loader:
    print(f"Image batch shape: {images.shape}")
    print(f"GT batch shape: {gts.shape}")
    break