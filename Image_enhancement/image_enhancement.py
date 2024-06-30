import cv2
import numpy as np
from matplotlib import pyplot as plt

# 載入圖像
image_path = '/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Public_Testing_dataset/images/PUB_RI_2000055.jpg'
image = cv2.imread(image_path)

# 將圖像轉為灰階
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 應用銳化濾波器
kernel = np.array([[0, -1, 0],
                   [-1, 6, -1],
                   [0, -1, 0]])
sharpened_image = cv2.filter2D(gray_image, -1, kernel)

# 使用直方圖均衡化增加對比度
equalized_image = cv2.equalizeHist(sharpened_image)

# 去噪
denoised_image = cv2.fastNlMeansDenoising(equalized_image, None, 30, 7, 21)

# 保存處理後的圖像
processed_image_path = '/home/ttsai/Drone_contest_2/Dataset_for_Contest2/process_image/PUB_RI_2000055.jpg'
cv2.imwrite(processed_image_path, denoised_image)

# 顯示原始圖像和處理後的圖像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.title('Processed image')
plt.imshow(denoised_image, cmap='gray')
plt.show()
