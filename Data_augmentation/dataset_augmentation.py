import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class GeneralDataset(Dataset):
    def __init__(self, image_root, gt_root, img_size, mode='train', augment=False):
        assert mode in ['train', 'val']
        self.img_size = img_size
        self.mode = mode
        self.augment = augment
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg')])
        self.gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')])
        self.filter_files()

        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
        
        self.da_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=1.0)  # 水平翻轉
            # transforms.RandomVerticalFlip(p=1.0), # 垂直翻轉
            # transforms.Lambda(lambda img: transforms.functional.adjust_brightness(img, 0.5))  # 固定圖像降低亮度50%，注意此時labels不用跟著降低
        ])
        
        # # =======懶得改程式，對應圖像降低亮度50%時，labels不用跟著降低========
        # self.da_transform2 = transforms.Compose([
        # ])
        # # ================================================================

# =====================================================資料集擴增=============================================================
        # 設定擴增後的保存路徑
        self.aug_image_save_path = "/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Training_dataset_divided_expand_brightness50%/images"
        self.aug_label_save_path = "/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Training_dataset_divided_expand_brightness50%/labels"

        # 創建保存目錄
        os.makedirs(self.aug_image_save_path, exist_ok=True)
        os.makedirs(self.aug_label_save_path, exist_ok=True)

        if self.mode == 'train' and self.augment:
            self.augment_and_save_images()
        else:
            self.load_augmented_images()

    def augment_and_save_images(self):
        base_index = 2010944 # 下一個要創建的名稱
        for i, (image_path, gt_path) in enumerate(zip(self.images, self.gts)):
            image = Image.open(image_path).convert('RGB')
            gt = Image.open(gt_path).convert('L')

            # 進行資料擴增
            aug_image = self.da_transform(image)
            aug_gt = self.da_transform(gt)
            
            # # =======懶得改程式，對應圖像降低亮度50%時，labels不用跟著降低==============
            # aug_gt = self.da_transform2(gt) # 對應圖像降低亮度50%時，labels不用跟著降低
            # # ======================================================================

            # 生成擴增後的名稱
            base_name = os.path.basename(image_path)
            new_index = base_index + i
            if "TRA_RI" in base_name:
                new_name = f"TRA_RI_{new_index:07d}.jpg"
            else:
                new_name = f"TRA_RO_{new_index:07d}.jpg"

            # 保存擴增後的影像和標籤
            aug_img_path = os.path.join(self.aug_image_save_path, new_name)
            aug_gt_path = os.path.join(self.aug_label_save_path, new_name.replace(".jpg", ".png"))
            aug_image.save(aug_img_path)
            aug_gt.save(aug_gt_path)

            print(f"Saved augmented image to {aug_img_path} and label to {aug_gt_path}")
# =============================================================================================================================

    def load_augmented_images(self):
        aug_images = sorted([os.path.join(self.aug_image_save_path, f) for f in os.listdir(self.aug_image_save_path) if f.endswith('.jpg')])
        aug_gts = sorted([os.path.join(self.aug_label_save_path, f) for f in os.listdir(self.aug_label_save_path) if f.endswith('.png')])
        self.images.extend(aug_images)
        self.gts.extend(aug_gts)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        return image, gt
    
    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images, self.gts = images, gts

    def __len__(self):
        return len(self.images)

# 測試資料集
train_images_root = "/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Training_dataset/images"  # 原始訓練集圖片
train_labels_root = "/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Training_dataset/labels"  # 原始訓練集標籤
dataset = GeneralDataset(train_images_root, train_labels_root, img_size=256, mode='train', augment=True)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 測試讀取資料
for images, gts in dataloader:
    print(images.size(), gts.size())
    break
