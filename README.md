# 🎥 3D CNN for Human Action Recognition

A deep learning project that uses a **3D Convolutional Neural Network (3D CNN)** to classify human actions from video clips. Built and trained on the **UCF101 dataset**, this model captures both spatial and temporal features to recognize activities like jumping, running, and swimming.

---

## 🧠 Project Objective
> Recognize human actions from video by extracting **spatiotemporal features** using 3D convolutions.

---

## 📂 Dataset: UCF101

- 📦 13,320 video clips
- 🏷️ 101 action categories
- 🎞️ Each video contains one person performing a single action
- 🔗 [UCF101 Official Site](https://www.crcv.ucf.edu/data/UCF101.php)

---

## 🏗️ Model Architecture

- **Input**: 16-frame video clips resized to `112x112`
- **Core Layers**:
  - 3D Convolutional Layers + ReLU
  - 3D MaxPooling Layers
  - Dropout Regularization
  - Fully Connected Dense Layers
  - Softmax Output Layer (for classification)

- **Training Config**:
  - Loss Function: `Categorical Crossentropy`
  - Optimizer: `Adam`
  - Metrics: `Accuracy`, `Top-5 Accuracy`

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **OpenCV** (frame extraction)
- **NumPy, Pandas**
- **Matplotlib** (visualizations)

---

## ⚙️ Workflow

1. **Preprocessing**
   - Extract and resize video frames
   - Generate fixed-length clips (16 frames)
   - Normalize pixel values

2. **Model Training**
   - Train 3D CNN on video clips
   - Validate using held-out set

3. **Evaluation**
   - Confusion matrix
   - Class-wise performance
   - Accuracy tracking

---

## 📊 Results (Approximate)

| Metric            | Value     |
|-------------------|-----------|
| Top-1 Accuracy    | ~78–85%   |
| Top-5 Accuracy    | ~93%      |
| Validation Loss   | ~0.6      |

---

