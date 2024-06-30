# Data Augmentation
- 原始資料訓練集有4320筆，原本採8:2隨機分割，並且在資料集中做一系列資料增強，包括隨機對圖像做垂直翻轉、水平翻轉、圖像旋轉、圖像裁剪、改變圖像的亮度、對比度、飽和度和色調，高斯模糊以及擦除部分圖像，增強模型的魯棒性。還有使用SAM方法，更好的進行邊緣偵測，去解決例如河道圖上面有橋梁、道路上會有車子的情況。但可能過多的資料增強，導致複雜度提高影響訓練的結果，因此後續做了一些改進。

- **Segment Anything Model (SAM)**，意思是對圖像做分割。由 Meta AI 開發的一種圖像分割模型，使用多層變壓器對這些小塊進行處理，提取高層次的圖像特徵。將提取的特徵進行融合，以生成對整個圖像的全局理解。根據特徵生成分割圖，標識出圖像中不同區域和物體的邊界。
  - Example
    - 對圖像做分割找尋河流、道路邊界
    - River image (SAM) vs Road image (SAM)
<div align="center"><img width="250" height="200" src="https://github.com/RickZhang0730/Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments/blob/main/Images/Data_augmentation1.jpg"><img width="250" height="200" src="https://github.com/RickZhang0730/Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments/blob/main/Images/Data_augmentation2.jpg"></div>



- 後續的改進，採用8:1:1的比例隨機分割，得到訓練集、測試集以及驗證集，並且對訓練集的資料做資料增強與擴增，首先採用隨機垂直翻轉、水平翻轉、以及圖像亮度降低50%，還有擦除部分圖像，並且做資料集擴增，將訓練集的資料擴充為16560筆。

- This folder  focuses on data augmentation and manipulation techniques for image datasets. The repository contains scripts for splitting datasets, augmenting data, merging datasets, checking the status of data loading, and visualizing the data.

### File Structure

- **dataset_splitting.py**: Script to split the dataset into training, validation, and testing datasets.
- **dataset_augmentation.py**: Script to perform data augmentation on the training dataset.
- **merge_datasets.py**: Script to merge the augmented datasets with the original dataset.
- **checking_status.py**: Script to check the status of the DataLoader.
- **visualizing_data.py**: Script to visualize images and their corresponding labels.

### Directory Structure

```plaintext
.
├── dataset_splitting.py
├── dataset_augmentation.py
├── merge_datasets.py
├── checking_status.py
└── visualizing_data.py
