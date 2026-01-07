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
- End-to-end machine learning pipeline: preprocessing → training → evaluation → inference  
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
Fingerprint-Blood-Group-Detection/
├── dataset/                    # Fingerprint image dataset (8 classes)
├── results/                    # Evaluation outputs and plots
├── templates/                  # HTML templates for Flask app
│   └── index.html              # Main UI template (if applicable)
│
├── app.py                      # Flask application logic
├── train_model.py              # CNN model training script
├── data_preprocessing.py       # Image preprocessing logic
├── evaluate.py                 # Model evaluation and metrics
├── blood_group_cnn_model.h5    # Trained CNN model
│
├── .gitignore                  # Git ignored files
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── LICENSE                     # MIT License

```

## 📸Sample Dataset Screenshot
<img width="1919" height="1199" alt="image" src="https://github.com/user-attachments/assets/b7795bc8-abf7-4ca9-95f9-cdfb0adc4e42" />

<img width="1919" height="1199" alt="image" src="https://github.com/user-attachments/assets/7444ce16-907f-4691-aa9c-0172a6e19b58" />

<img width="1919" height="1199" alt="image" src="https://github.com/user-attachments/assets/976d4301-32a9-4ce7-92a5-ea336c14f89c" />

<img width="1919" height="1199" alt="image" src="https://github.com/user-attachments/assets/b2613a3a-6f64-4c68-8faf-f7539d780bdf" />



### ▶️ How to Run the Project

 Create virtual environment (Windows)

    python -m venv venv
    venv\Scripts\activate

 Install dependencies
    
    pip install -r requirements.txt

 Run the Flask application

    python app.py

  Open in browser
  
    http://127.0.0.1:5000/

 ---   

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

## 📁 Dataset Details

- Contains fingerprint images across **8 blood group classes**
- Dataset size is limited and used for **academic and experimental purposes**
- Images are preprocessed using grayscale conversion, resizing, and normalization

> ⚠️ Note: This system is intended for **research and learning purposes only** and should not be used for medical diagnosis.


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

🚧 Limitations & Accuracy Improvement Strategies

The current model achieves approximately 45% accuracy due to the following factors:
- Limited dataset size across 8 classes
- Variability in fingerprint image quality
- Use of a basic CNN architecture

Planned improvements include:
- Data augmentation to increase dataset diversity
- Dataset balancing across blood group classes
- Adoption of transfer learning models (ResNet, EfficientNet)
- Hyperparameter tuning and regularization techniques




