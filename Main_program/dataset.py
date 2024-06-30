import torch
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class GeneralDataset(Dataset):
    def __init__(self, image_root, gt_root, img_size, mode='train'):
        assert mode in ['train', 'val']
        self.img_size = img_size
        self.mode = mode
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
            # transforms.RandomVerticalFlip(p=0.3), # 垂直翻轉
            # transforms.RandomHorizontalFlip(p=0.3), # 水平翻轉
            # # transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.2), # 旋轉
            # # transforms.ColorJitter(brightness=0.05, contrast=0.05), # 改變圖像的亮度、對比度、飽和度和色調
            # # transforms.GaussianBlur(kernel_size=5, sigma=(0.1,0.2)), # 高斯模糊
            # # transforms.RandomApply([transforms.RandomErasing(p=0.5, scale=(0.02, 0.03), ratio=(0.3, 0.3), value=0)], p=0.2) # 擦除部分圖像，增強模型的魯棒性
        ])

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')

        if self.mode == "train":
            seed = torch.random.initial_seed()
            torch.manual_seed(seed)
            image = self.img_transform(image)  # 先轉換為Tensor
            torch.manual_seed(seed)
            gt = self.gt_transform(gt)         # 先轉換為Tensor
        else:
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

class TestDataset(Dataset): # 用於 testing dataset 的 Dataset
    def __init__(self, image_root, img_size):
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg')])
        self.img_size = img_size
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')  # 打開圖像文件，轉換為RGB
        image = self.img_transform(image)
        name = os.path.basename(self.images[index]).replace('.jpg', '.png')  # 取得圖像的文件名稱，並替換後綴
        return image, name

    def __len__(self):
        return len(self.images)
