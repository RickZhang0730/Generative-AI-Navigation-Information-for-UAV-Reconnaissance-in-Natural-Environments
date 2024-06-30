import numpy as np
from matplotlib import pyplot as plt
from Data_augmentation.checking_status import tr_loader

# 檢查 DataLoader 是否成功載入 image 和 ground truth
data_iter = iter(tr_loader)
images, labels = next(data_iter)


image, label = images[0], labels[0]


image = image.numpy()
label = label.numpy()
image = np.transpose(image, (1, 2, 0))
label = np.transpose(label, (1, 2, 0))

if image.min() < 0 or image.max() > 1:
    image = (image - image.min()) / (image.max() - image.min())
plt.imshow(image)
plt.show()
plt.imshow(label, cmap='gray')
plt.show()