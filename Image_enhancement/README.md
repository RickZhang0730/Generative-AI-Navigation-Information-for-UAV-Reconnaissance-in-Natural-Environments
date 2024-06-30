# Image Enhancement
由於無人機空拍會有一些晃動，導致一些空拍圖不太清楚，因此做了一些處理，轉換為灰階影像，定義一個銳化的濾波器，去增強圖像中的邊緣。再來做直方圖均衡化，增強圖像的對比度。接下來降躁處理，減少圖像中的噪點。使得圖像變得更加清晰。

- Due to the vibrations during drone aerial photography, some aerial images may not be very clear. Therefore, we applied several processing steps to enhance the image clarity. First, we converted the images to grayscale. Then, we defined a sharpening filter to enhance the edges in the images. Next, we performed histogram equalization to enhance the image contrast. Finally, we applied noise reduction to minimize noise in the images, making them clearer.

### File Structure

- **image_enhancement.py**: Script for enhancing image clarity.

### Example
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

### Directory Structure

```plaintext
.
└── image_enhancement.py
