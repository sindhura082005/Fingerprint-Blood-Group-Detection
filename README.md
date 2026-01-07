# 🧬 Fingerprint-Based Blood Group Detection using CNN

## 📌 Project Overview
This project implements an **end-to-end deep learning system** to predict **human blood groups from fingerprint images** using a **Convolutional Neural Network (CNN)**.  
The system classifies fingerprints into **8 blood group categories**:  
**A+, A−, B+, B−, AB+, AB−, O+, O−**

The project demonstrates a **complete machine learning pipeline** including data preprocessing, model training, evaluation, and deployment using a **Flask web application**.

---

## 🚀 Key Features
- CNN-based multi-class fingerprint image classification  
- Supports **8 blood group classes**  
- Custom dataset collected manually (not from Kaggle)  
- End-to-end ML pipeline: preprocessing → training → evaluation → inference  
- Flask-based web interface for real-time prediction  
- Model performance evaluation using confusion matrix and classification report  

---

## 🧠 Technologies Used
- **Programming Language:** Python  
- **Deep Learning Framework:** TensorFlow, Keras  
- **Model Architecture:** Convolutional Neural Network (CNN)  
- **Web Framework:** Flask  
- **Image Processing:** OpenCV, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score  

---

## 🏗 System Architecture
---
```text
```text
User
 │
 │ uploads fingerprint image
 ▼
Flask Web Application
 │
 │ receives image input
 ▼
Image Preprocessing Module
 │
 │ grayscale conversion
 │ normalization & resizing
 ▼
CNN Model (blood_group_cnn_model.h5)
 │
 │ feature extraction
 │ classification
 ▼
Blood Group Prediction
 │
 │ maps output to label
 ▼
Result Display (Web Interface)
```

## 📂 Project Structure
---
```text
Movie-Recommendation-System/
├── app/
│   ├── app.py                  # Streamlit application logic
│   └── assets/                 # App screenshots & static assets
│       ├── app-screenshot-1.png
│       ├── app-screenshot-2.png
│       └── app-screenshot-3.png
│
├── data/
│   └── movies.csv              # Movie metadata dataset
│
├── .env                        # Environment variables (API keys)
├── .gitignore                  # Git ignored files
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── LICENSE                     # MIT License
```



## ⚙️ Model Training
- Input images resized to a fixed shape
- CNN architecture includes:
  - Convolution layers
  - Max pooling layers
  - Fully connected dense layers
  - Softmax output layer for multi-class classification
- Optimizer: Adam  
- Loss function: Categorical Crossentropy  

---

## 📊 Model Evaluation
The model is evaluated using:
- Confusion Matrix
- Precision, Recall, F1-score
- Classification Report

Evaluation results are saved in the `results/` directory.

---

## 🌐 Web Application
- Built using **Flask**
- Allows users to upload a fingerprint image
- Displays predicted blood group in real time
- Uses the trained CNN model (`.h5`) for inference

---




