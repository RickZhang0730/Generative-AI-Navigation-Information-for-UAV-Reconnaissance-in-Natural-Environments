# Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments
- A project focused on constructing autonomous UAV navigation datasets for natural environments, leveraging generative AI to enhance and generate training data from images of Taiwan's roads and rivers.
- 建構以台灣道路、河流為目標的無人機自主導航影像資料集，根據天氣、視角、目標大小等因素進行分類。
- 生成式AI用於擴充訓練資料集以及生成所需資料，建構無人機於自然環境偵察時所需之導航資訊。
- https://1drv.ms/i/c/a3d918eaa1794158/EaLfkQssz3xFo-kmhSrycUABAsI-6vDlSeRYwDt0_Vr9Zg
- https://1drv.ms/u/c/a3d918eaa1794158/EaxmD5iAp7NGnjkvpcnmcK0B8FsXiGv27xDRInxmkK8Fxw
- https://1drv.ms/u/c/a3d918eaa1794158/ETKd7amRRRVJoloP3hJVSV4BhLbllZQt5zbwYye4VPB4FA?e=RtFlTW

## Installation
To creating the environment, you can use the provided env.yml file to create a conda environment.
```bash
conda env create -f env.yml
conda activate your_environment_name
```

### File Structure
- **Data_augmentation**
  - **dataset_splitting.py**: Script to split the dataset into training, validation, and testing datasets.
  - **dataset_augmentation.py**: Script to perform data augmentation on the training dataset.
  - **merge_datasets.py**: Script to merge the augmented datasets with the original dataset.
  - **checking_status.py**: Script to check the status of the DataLoader.
  - **visualizing_data.py**: Script to visualize images and their corresponding labels.

- **Image_enhancement**
  - **image_enhancement.py**: Script for enhancing image clarity.

- **Main_program**
  - **dataset.py**: Script for dataset preparation.
  - **evaluation.py**: Script for evaluating the trained model on the test dataset.
  - **loss_functions.py**: Custom loss functions used during model training.
  - **model_definition.py**: Model architecture definitions.
  - **run.sh**: Shell script to run the training, evaluation, testing, and validation processes.
  - **testing.py**: Script for testing the trained model.
  - **training.py**: Script for training the model.
  - **validation.py**: Script for validating the model during training.

- **Visualize_training_process**
  - **visualize_training_process.py**: Script to visualize the training process.

## Usage
### Running the Entire Program
- To run the entire program including training, evaluation, testing, and validation, use the provided run.sh script:
```bash
./run.sh
```
- 在背景執行腳本並輸出紀錄檔
- To run the script in the background and output logs to output.log:
```bash
nohup bash run.sh > output.log &
```
### Training the Model
- To train the model:
```bash
python training.py --data_path '/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Training_dataset_divided_expand_combined' --val_path '/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Validation_dataset' --log_file_path '/home/ttsai/Drone_contest_2/Log_file/training.log' --epochs 50 --batch_size 12 --save_path '/home/ttsai/Drone_contest_2/Save_path_all_test'
```

### Evaluating the Model
- To evaluate the trained model on the validation dataset:
```bash
python evaluation.py --model_path '/home/ttsai/Drone_contest_2/Save_path_all_test/best_model.pth' --test_data_path '/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Validation_dataset' --log_file_path '/home/ttsai/Drone_contest_2/Log_file/evaluation.log' --batch_size 4
```

### Testing the Model
- To test the trained model:
```bash
python testing.py --test_root '/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Testing_dataset' --model_path '/home/ttsai/Drone_contest_2/Save_path_all_test/best_model.pth' --save_path '/home/ttsai/Drone_contest_2/Results_test/Results'
```

### Validating the Model
- To validate the model:
```bash
python validation.py --mask_root '/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Testing_dataset/labels' --pred_root '/home/ttsai/Drone_contest_2/Results_test/Results'
```

### Directory Structure

```plaintext
.
├── Data_augmentation
    ├── README.md
    ├── dataset_splitting.py
    ├── dataset_augmentation.py
    ├── merge_datasets.py
    ├── checking_status.py
    └── visualizing_data.py
├── Image_enhancement
    ├── README.md
    └── image_enhancement.py
├── Images
├── Main_program
    ├── README.md
    ├── dataset.py
    ├── evaluation.py
    ├── loss_functions.py
    ├── model_definition.py
    ├── run.sh
    ├── testing.py
    ├── training.py
    └── validation.py
├── Visualize_training_process
    ├── README.md
    └── visualize_training_process.py
├── README.md
└── env.yml
```
## Introduction
- 建構以台灣道路、河流為目標的無人機自主導航影像資料集，根據天氣、視角、目標大小等因素進行分類。生成式AI用於擴充訓練資料集以及生成所需資料，建構無人機於自然環境偵察時所需之導航資訊。
- 比賽時驗證集以8:2隨機生成，並透過資料增強技術，去提高模型的泛化能力。
使用SAM技術，對圖像的區域做分割，提高河道與道路邊緣範圍的偵測。模型使用UNet架構，並採用EfficientNet-B7作為Encoder。
- 最終在leaderboard為第25名，private分數是0.647805，public分數為0.664916。
- 比賽後做了反思，將錯誤的部份做修改，並且腳本化，以及使用沒有考量到的方法，在後續作了一系列的改進，包括資料集擴增、循序漸進的模型架構、邊緣偵測以及集成學習，得到更好的泛化能力。並且發現，河流與道路兩種資料集會互相影響訓練的結果，因此未來應該把彼此分開做訓練，可能會得到更佳的結果。

## Method
### Image Enhancement
由於無人機空拍會有一些晃動，導致一些空拍圖不太清楚，因此做了一些處理，轉換為灰階影像，定義一個銳化的濾波器，去增強圖像中的邊緣。再來做直方圖均衡化，增強圖像的對比度。接下來降躁處理，減少圖像中的噪點。使得圖像變得更加清晰。

- Due to the vibrations during drone aerial photography, some aerial images may not be very clear. Therefore, we applied several processing steps to enhance the image clarity. First, we converted the images to grayscale. Then, we defined a sharpening filter to enhance the edges in the images. Next, we performed histogram equalization to enhance the image contrast. Finally, we applied noise reduction to minimize noise in the images, making them clearer.
  - Example
    - original image vs enhanced image
      
      <table style="border: none;">
        <tr>
          <td align="center" style="border: none;">
            <p>Original image</p>
                  <img width="350" height="200" src="https://github.com/RickZhang0730/Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments/blob/main/Images/Image_enhancement1.jpg" alt="original image">
          </td>
          <td align="center" style="border: none;">
            <p>Enhanced image</p>
            <img width="350" height="200" src="https://github.com/RickZhang0730/Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments/blob/main/Images/Image_enhancement2.jpg" alt="enhanced image">
          </td>
        </tr>
      </table>
### Data Augmentation
- 原始資料訓練集有4320筆，原本採8:2隨機分割，並且在資料集中做一系列資料增強，包括隨機對圖像做垂直翻轉、水平翻轉、圖像旋轉、圖像裁剪、改變圖像的亮度、對比度、飽和度和色調，高斯模糊以及擦除部分圖像，增強模型的魯棒性。還有使用SAM方法，更好的進行邊緣偵測，去解決例如河道圖上面有橋梁、道路上會有車子的情況。但可能過多的資料增強，導致複雜度提高影響訓練的結果，因此後續做了一些改進。

- **Segment Anything Model (SAM)**，意思是對圖像做分割。由 Meta AI 開發的一種圖像分割模型，使用多層變壓器對這些小塊進行處理，提取高層次的圖像特徵。將提取的特徵進行融合，以生成對整個圖像的全局理解。根據特徵生成分割圖，標識出圖像中不同區域和物體的邊界。
  - Example
    - 對圖像做分割找尋河流、道路邊界
    - River image (SAM) vs Road image (SAM)
<div align="center"><img width="250" height="200" src="https://github.com/RickZhang0730/Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments/blob/main/Images/Data_augmentation1.jpg"><img width="250" height="200" src="https://github.com/RickZhang0730/Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments/blob/main/Images/Data_augmentation2.jpg"></div>

- **Random Erasing**
  - 圖像資料增強的技術，提高模型的泛化能力，隨機擦除圖像中的一部分區域，來模擬部分圖像缺失或遮擋的情況。
- 後續的改進，採用8:1:1的比例隨機分割，得到訓練集、測試集以及驗證集，並且對訓練集的資料做資料增強與擴增，首先採用隨機垂直翻轉、水平翻轉、以及圖像亮度降低50%，還有擦除部分圖像，並且做資料集擴增，將訓練集的資料擴充為16560筆。

## Training Model
- 觀察原始競賽提供的model可以了解到，它在引導我們這是一個U-Net的架構，一種常用於圖像分割任務的深度學習模型架構，可以分為兩個主要部分。
- Eecoder，類似於傳統的卷積神經網絡，由多個卷積層和池化層組成。每個卷積層後面跟著一個激勵函數，接著是一個池化層。負責逐漸縮小圖像的空間尺寸，同時增加特徵的數量，以提取高層次特徵。
- Decoder，由多個反捲積層和UNSampling組成，用於逐漸恢復圖像的空間尺寸，將Encoder提取的高層次特徵轉換回原始圖像的分辨率，並進行逐像素的分類。
- 最終採用U-Net++，一種改進的UNet架構，專門用於圖像分割任務。與傳統的UNet相比，UNet++ 引入了更多的連接層和密集的跳躍連接，從而提升了模型的表現。這些密集塊由多層卷積組成，每一層卷積的輸入來自於前一層的輸出以及同一層次編碼器的輸出。這樣的設計使得模型能夠更好地捕捉多尺度特徵，從而提升分割精度，並且訓練過程採動態更新學習率。
- 使用Efficientnet-b7作為我的Encoder，具有更深的網路結構，這意味著它包含更多的卷積層，能夠學習到更加複雜和高層次的特徵。更寬的網路結構，即每層卷積的通道數更多，這使得它能夠處理更多的特徵圖，從而提高了模型的表現。接受更高解析度的輸入圖像，能夠捕捉到更多的圖像細節，有助於提高模型的準確性。

## Experiment
- 訓練模型，並且每兩輪儲存一次權重，可以把之前訓練最佳的模型權重保存到下一輪接著去做訓練比較，平均較佳則會更新模型，動態調整學習率，防止Overfitting。
- 並且使用視覺化，去觀察訓練過程的軌跡，能夠更好的去做調整。最終把模型拿去生成測試集圖片的labels，透過F-measure與原始labels正解，去測量精準度。

### Visualize Training Process
- 使用視覺化，去觀察訓練過程的軌跡，能夠更好的去做調整。
- Using visualization to observe the training process.
  - Example

    <div align = left><img width="600" height="380" src="https://github.com/RickZhang0730/Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments/blob/main/Images/Visualize_training_process1.png"></div>

## Results
- River image vs Road image
<table style="border: none; width: 100%;">
    <tr>
        <td align="center" style="border: none; width: 50%;">
            <p>Original image</p>
            <img width="350" height="200" src="https://github.com/RickZhang0730/Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments/blob/main/Images/PUB_RI_2000005.jpg" alt="original image">
            <p>Generative label</p>
            <img width="350" height="200" src="https://github.com/RickZhang0730/Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments/blob/main/Images/PUB_RI_2000005.png" alt="generative label">
        </td>
        <td align="center" style="border: none; width: 50%;">
            <p>Original image</p>
            <img width="350" height="200" src="https://github.com/RickZhang0730/Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments/blob/main/Images/PUB_RO_2000432.jpg" alt="original image">
            <p>Generative label</p>
            <img width="350" height="200" src="https://github.com/RickZhang0730/Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments/blob/main/Images/PUB_RO_2000432.png" alt="generative label">
        </td>
    </tr>
</table>

## Conclusion
- 本研究顯示了使用生成式AI技術來構建無人機於自然環境下的地形導航資料的有效方法。結果顯示，使用EfficientNet-B7作為Encoder的U-Net模型在公開測試集和私有測試集上均取得了不錯的成績，然而仍存在提升空間。

- **資料集擴增**：增加資料集的數量和多樣性，通過隨機垂直翻轉、水平翻轉、改變亮度等方法，顯著提高了模型的泛化能力。
- **模型改進**：從原本的U-Net架構升級為U-Net++，引入密集的跳躍連接和動態更新學習率策略，有效地提升了模型的效能。
- **邊緣偵測**：使用Segment Anything Model (SAM)技術，精確地分割河流和道路邊界，解決了橋梁和車輛對圖像邊界的不利影響。

- 透過以上改進，模型的效能有了顯著的提升，表現在泛化能力和精準度上均有所改善。未來的研究應考慮將河流和道路資料集分開訓練，以進一步提升訓練的效果。此外，集成學習方法的引入也將是未來提升模型效能的方向。

## Reference
[1] Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.

[2] Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & Girshick, R. (2023). Segment Anything.

[3] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.

[4] Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2018). UNet++: A Nested U-Net Architecture for Medical Image Segmentation.

[5] Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
