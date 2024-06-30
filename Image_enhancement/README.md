# Image Enhancement

Due to the vibrations during drone aerial photography, some aerial images may not be very clear. Therefore, we applied several processing steps to enhance the image clarity. First, we converted the images to grayscale. Then, we defined a sharpening filter to enhance the edges in the images. Next, we performed histogram equalization to enhance the image contrast. Finally, we applied noise reduction to minimize noise in the images, making them clearer.

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
