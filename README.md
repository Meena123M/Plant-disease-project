# 🌿 Plant Village Disease Prediction System

A deep learning-powered web application that predicts plant diseases based on leaf images. The system uses image classification techniques and deploys the trained model via Streamlit for real-time user interaction.

---

## 📁 1. Data Collection and Preprocessing

- **Data Collection:** Gathered a labeled dataset of plant leaf images representing 38 different diseases.
- **Image Preprocessing:** Resized, normalized, and augmented images to enhance training efficiency and generalization.

---

## 🔀 2. Data Splitting and Preparation

- **Data Separation:** Divided the dataset into **training**, **validation**, and **testing** sets.
- **Data Generators:** Used `ImageDataGenerator` for training, validation, and testing.
- **Parameter Tuning:** Configured key training parameters including:
  - Batch Size  
  - Learning Rate  
  - Number of Epochs

---

## 🧠 3. Model Training

- **Model Architecture:** Implemented **ResNet50** with a **GlobalAveragePooling2D** layer for robust image classification.
- **Training:** Used TensorFlow/Keras with generators to efficiently train the model on the processed data.

---

## 📊 4. Model Evaluation

- **Performance Metrics:** Achieved  
  - ✅ **Training Accuracy:** 95.54%  
  - ✅ **Test Accuracy:** 94.78%
- **Visualization:** Plotted training/validation accuracy and loss curves to monitor performance.

---

## 🔍 5. Prediction

- **Inference:** Used the trained model to predict disease categories on unseen test images.
- **Evaluation:** Compared predictions with true labels for accuracy assessment.

---

## 🚀 6. Deployment

- **Streamlit Web App:**  
  Deployed the model using **Streamlit** for an intuitive web interface.
- **Features:**  
  - Upload plant leaf images  
  - Get real-time disease predictions  
  - Simple and interactive UI

---

## 🛠️ Tools & Libraries Used

- `cv2`, `numpy`, `pandas`, `matplotlib`  
- `TensorFlow`, `Keras`, `ResNet50`  
- `Streamlit`  
- `ImageDataGenerator` (for image augmentation)

---

## 📌 Project Highlights

- ✅ End-to-end pipeline: data → model → deployment  
- 🌿 Focused on agricultural disease identification  
- 📷 Image-based classification using transfer learning  
- 🌐 Real-time web deployment for user accessibility

---

## 🤝 Contributions

Developed by [Meena M](https://github.com/Meena123M)  
Open to collaboration, feedback, and improvements!


