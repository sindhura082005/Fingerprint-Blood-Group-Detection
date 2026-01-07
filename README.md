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
Fingerprint Image

↓

Image Preprocessing

↓

CNN Model (Feature Extraction + Classification)

↓

Blood Group Prediction

↓

Flask Web Application

## 📂 Project Structure
---
Fingerprint-Blood-Group-Detection/
│
├── dataset/ # Fingerprint image dataset (8 classes)

├── results/ # Evaluation outputs and plots

├── templates/ # HTML templates for Flask app
│
├── app.py # Flask application

├── train_model.py # CNN model training script

├── data_preprocessing.py # Image preprocessing logic

├── evaluate.py # Model evaluation and metrics

├── blood_group_cnn_model.h5 # Trained CNN model
│
├── requirements.txt # Project dependencies

├── .gitignore

├── LICENSE

└── README.md




