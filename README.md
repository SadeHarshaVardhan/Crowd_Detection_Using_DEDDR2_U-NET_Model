# Crowd Detection Using Double Encoder Double Decoder Residual Recurrent U-Net Model

## 📌 Abstract

This project proposes a novel deep learning approach for real-time crowd detection using a **Double Encoder Double Decoder Residual Recurrent U-Net (DEDDR² U-Net)** architecture. Designed for applications like public safety, event monitoring, and urban planning, this system provides accurate crowd density estimation using a Raspberry Pi-powered IoT setup with a stereo camera. Data is processed in real-time using Google Colab to produce density maps and head counts.

---

## 🧠 Methodology

![image](https://github.com/user-attachments/assets/606ab79e-7ed2-467e-a8d4-ee153c050ede)


## 🧠 Methodology

The core of this project is the **DEDDR² U-Net**: *Double Encoder Double Decoder Residual Recurrent U-Net*, a robust image segmentation architecture tailored for precise crowd detection and density estimation.

This model improves upon the classic U-Net by introducing **double paths for encoding and decoding**, **residual connections for better learning flow**, and **recurrent convolutional layers** for temporal consistency — crucial for crowd scenes in video.

---

### 🔄 Architecture Breakdown

#### 🧩 Input Layer
- Takes crowd scene images (RGB format).
- Feeds them into two parallel encoders for multi-perspective feature extraction.

---

### 🧠 Double Encoder Path

The **Double Encoder** consists of two parallel encoder streams:

- **Encoder 1**: Extracts **global contextual features** — overall crowd layout, density gradients.
- **Encoder 2**: Extracts **local fine-grained features** — heads, boundaries, shapes.

Each encoder includes:

- **Convolutional Layers**: 3×3 kernels + ReLU activation, used to detect features like edges, patterns, or textures.
- **Recurrent Convolutional Layers (RCLs)**: Help maintain consistency across frames in video by learning temporal dependencies.
- **Max Pooling Layers**: 2×2 pooling to downsample the feature maps and retain dominant features.

**🔽 Why Max Pooling?**

- Reduces spatial dimensions (e.g., 64×64 → 32×32).
- Highlights the most salient features.
- Increases the field of view while reducing computation.

> **Max Pooling Example (2×2):**
>
> ```
> Input:
> [[1, 3, 2, 4],
>  [5, 6, 7, 8],
>  [9, 4, 3, 2],
>  [3, 1, 5, 6]]
>
> Output:
> [[6, 8],
>  [9, 6]]
> ```

---

### 🔄 Bottleneck Layer

At the center of the network lies the **Bottleneck**, which:

- **Combines outputs** from both encoders.
- Applies **residual blocks** to retain learned patterns and enable deep gradient flow.
- May use **attention mechanisms** to highlight important spatial regions (e.g., crowded zones).
- Produces an abstract, high-dimensional feature representation that bridges the encoder and decoder stages.

---

### 🔼 Double Decoder Path

The **Double Decoder** reconstructs the output (density map):

- **Decoder 1**: Focuses on **global reconstruction**, preserving overall density structure.
- **Decoder 2**: Adds **fine detail refinement**, enabling detection of individuals in dense crowds.

Each decoder contains:

- **Upsampling Layers (Transposed Convolutions)**: Increase spatial resolution back to the input image size.
- **Recurrent Convolutional Layers**: Further refine the upsampled features, leveraging temporal consistency.
- **Residual Connections**: Maintain stability and improve gradient propagation.
- **Skip Connections**: Directly link encoder features to the decoder, preserving spatial accuracy.

---

### 🟢 Final Processing

- **1×1 Convolutional Layer**: Converts the final feature map to a single-channel crowd density map.
- **Thresholding**: Applies a threshold (e.g., > 0.4) to produce binary segmentation masks.
- **Element-wise Multiplication**: Multiplies the binary mask with the original input to highlight detected regions.
- **Final Output**: Includes an annotated image with crowd regions and head count.

---

This architecture is optimized for real-time, low-power environments, making it ideal for IoT deployments using Raspberry Pi and cloud computing platforms.

---

## 🌐 IoT Deployment

![image](https://github.com/user-attachments/assets/037da779-3988-468f-87f9-d115b081123b)    



- **Edge Device**: Raspberry Pi 4B with 4GB RAM
- **Camera**: Stereo Vision Camera
- **Protocols Used**:
  - `HTTP` for live stream
  - `TCP/IP` for data transfer
  - `I2C` for interfacing sensors
- **Data Transfer**: `rclone` to Google Drive
- **Processing**: Jupyter Notebook in Google Colab
- **Drone Integration**: Quardcopter carrying the Raspberry Pi setup for aerial monitoring

---

## 📦 Resources Used

### 💻 Software

- Python
- OpenCV
- TensorFlow / PyTorch
- Google Colab
- Jupyter Notebook
- Rclone
- Raspiconnect

### 🔧 Hardware

- Raspberry Pi 4B
- Stereo Camera
- Quardcopter Drone
- GPS, IMU sensors (optional)

### 🗃️ Datasets

- [ShanghaiTech Part A & B](https://www.kaggle.com/datasets/tthien/shanghaitech)
- [UCF-QNRF](https://crcv.ucf.edu/projects/ucf-qnrf/)
- [WorldExpo’10](https://github.com/sweetyy83/Crowd_Counting/blob/master/Dataset.md)
- u can directly download the dataset from the above "**links for Model & Dataset.text**" file.
---

## 💡 Expected Outputs

- Real-time crowd count
- Density map highlighting crowd concentration
- Annotated images with timestamps and people count

### 📸 Results

| Original Image | Masked Image | Predicted Output |
|----------------|--------------|------------------|
| ![image](https://github.com/user-attachments/assets/664f5e12-5537-4569-94a1-01b02183d76a)| ![image](https://github.com/user-attachments/assets/2137f5f6-714c-472d-abda-47b3105424b1)| ![image](https://github.com/user-attachments/assets/7945d064-8d54-4e79-a11c-cf0013f2cfad)|
| ![image](https://github.com/user-attachments/assets/8c33a241-632f-4c1c-a559-5dc0485c1102)| ![image](https://github.com/user-attachments/assets/7210ce49-e974-41c1-a356-c57f96100488)| ![image](https://github.com/user-attachments/assets/01e3e2d0-545e-4989-90b1-3b12ed611939)|
| ![image](https://github.com/user-attachments/assets/1e62bde1-457b-4ab8-85f7-b5c32095c367)| ![image](https://github.com/user-attachments/assets/e4251a55-5952-4de2-8ddf-2cb862bbb6a0)| ![image](https://github.com/user-attachments/assets/000ec9f3-ca5e-4ef5-b4fb-1b4bb0b8cc40)|
| ![image](https://github.com/user-attachments/assets/4266902e-b0bf-494e-8836-2c5068d3bd7e)|![image](https://github.com/user-attachments/assets/54ea51c0-4e11-4fa5-b00e-df2df1fffa46)| ![image](https://github.com/user-attachments/assets/2be5b5e8-ba75-420c-970b-26e9f706a4ff)|

---
### 📸 Realtime Live Prediction Outputs

![image](https://github.com/user-attachments/assets/c6e600fd-25e4-4883-9a1a-b3961e2648b8)

## 📚 References

1. Huiyuan Fu, Huadong Ma, Hongtian Xiao.  
   *Beijing Key Laboratory of Intelligent Telecommunications Software and Multimedia, Beijing University of Posts and Telecommunications*  
   [Crowd Density Estimation with Convolutional Neural Network](https://dl.acm.org/doi/10.1145/2647868.2655040)

2. *Density-Attentive Head Detector for Crowd Counting*  
   [IEEE Proceedings Article](https://www.computer.org/csdl/proceedings-article/icdsba/2019/464400a097/1pbdQPBR38Y)

3. *People Counting in Dense Crowd Images using Sparse Head Detections*  
   [ResearchGate Publication](https://www.researchgate.net/publication/322998388_People_Counting_in_Dense_Crowd_Images_Using_Sparse_Head_Detections3)

4. *Head Detection Based on Skeleton Graph Method for Counting People in Crowded Environments*  
   [No link provided – please add if available]

5. Deepak Babu Sam, Shiv Surya, R. Venkatesh Babu.  
   *Switching Convolutional Neural Network for Crowd Counting*  
   [arXiv](https://arxiv.org/abs/1708.00199)

6. *Counting People in the Crowd Using a Generic Head Detector*  
   [Semantic Scholar](https://www.semanticscholar.org/paper/Counting-People-in-the-Crowd-Using-a-Generic-Head-Subburaman-Descamps/f34df8090f6a24b37ee1bda15d743502ff03edab)

7. *Density Map Regression Guided Detection Network for RGB-D Crowd Counting and Localization*  
   [CVPR 2019 Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lian_Density_Map_Regression_Guided_Detection_Network_for_RGB-D_Crowd_Counting_CVPR_2019_paper.pdf)

8. *A Real-Time Deep Network for Crowd Counting*  
   [arXiv](https://arxiv.org/pdf/2002.06515.pdf)

9. Kaiming He et al.  
   *Deep Residual Learning for Image Recognition*  
   [CVPR 2016 Paper](https://www.cvfoundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

10. Karen Simonyan, Andrew Zisserman.  
    *Very Deep Convolutional Networks for Large-Scale Image Recognition*  
    [arXiv](https://arxiv.org/pdf/1409.1556.pdf)

11. Olaf Ronneberger et al.  
    *U-Net: Convolutional Networks for Biomedical Image Segmentation*  
    [arXiv](https://arxiv.org/pdf/1505.04597.pdf)

12. *Crowd Counting Using Scale-Aware Attention Networks*  
    [arXiv](https://arxiv.org/pdf/1903.02025.pdf)

13. Li, Y. et al.  
    *CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes*  
    [CVPR 2018 Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_CSRNet_Dilated_Convolutional_CVPR_2018_paper.pdf)

14. *A Survey of Recent Advances in CNN-based Single Image Crowd Counting and Density Estimation*  
    [arXiv](https://arxiv.org/pdf/1707.01202.pdf)

15. Zhang, Y. et al.  
    *Single-Image Crowd Counting via Multi-Column Convolutional Neural Network*  
    [CVPR 2016 Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)

16. Sindagi, V. A. et al.  
    *Generating High-Quality Crowd Density Maps using Contextual Pyramid CNNs*  
    [ICCV 2017 Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Sindagi_Generating_High-Quality_Crowd_ICCV_2017_paper.pdf)

17. Lowe, D.G.  
    *Distinctive Image Features from Scale-Invariant Keypoints*  
    [Paper PDF](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)

18. Khan, M. S., & Muhammad, K.  
    *FGA: Fourier-Guided Attention Network for Crowd Count Estimation*  
    [arXiv](https://arxiv.org/pdf/2407.06110.pdf)

19. *Learning To Count Objects in Images*  
    [NeurIPS 2010 Paper](https://proceedings.neurips.cc/paper_files/paper/2010/file/fe73f687e5bc5280214e0486b273a5f9-Paper.pdf)

20. *Estimating the Crowd Size of a Rally by Crowdsourcing-Geocomputation*  
    [ICA Abstracts](https://ica-abs.copernicus.org/articles/1/46/2019/ica-abs-1-46-2019.pdf)



