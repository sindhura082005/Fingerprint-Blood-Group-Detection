# 🧬 Fingerprint-Based Blood Group Detection using CNN

This project implements an end-to-end deep learning system to predict human blood groups from fingerprint images using a Convolutional Neural Network (CNN). The system classifies fingerprints into 8 blood group categories:
A+, A−, B+, B−, AB+, AB−, O+, O−.

The objective of this project is to explore the feasibility of non-invasive blood group prediction using biometric patterns and deep learning, contributing to AI-driven healthcare research.

🚀 Key Highlights

CNN-based multi-class image classification

Complete ML lifecycle: preprocessing → training → evaluation → deployment

Flask-based web application for real-time prediction

Transparent evaluation using confusion matrix and classification report

🧠 Tech Stack

Python

TensorFlow / Keras

OpenCV

NumPy, Scikit-learn

Flask

HTML / CSS

📂 Project Structure
Fingerprint-Blood-Group-Detection/
│
├── dataset/                    
│
├── results/
│   ├── classification_report.txt
│   └── confusion_matrix.png
│
├── templates/
│   ├── index.html               
│   └── result.html              
│
├── app.py                       
├── data_preprocessing.py        
├── evaluate.py                  
├── blood_group_cnn_model.h5     
├── requirements.txt

⚙️ Detailed Implementation
🔹 1. Data Preprocessing (data_preprocessing.py)

This module handles all image preprocessing tasks:

Image resizing and normalization

Conversion to grayscale (if required)

Dataset loading using ImageDataGenerator

Train–validation split

Purpose:
Ensures fingerprint images are standardized before feeding into the CNN model.

🔹 2. Model Architecture

The CNN model consists of:

Convolutional layers for feature extraction

Max-pooling layers for dimensionality reduction

Fully connected (Dense) layers for classification

Softmax activation for 8-class output

Loss Function: Categorical Cross-Entropy
Optimizer: Adam

🔹 3. Model Evaluation (evaluate.py)

This script evaluates the trained model using:

Classification Report (precision, recall, F1-score)

Confusion Matrix visualization

Class-wise performance analysis

The results are saved in the /results directory for transparency and reproducibility.

🔹 4. Web Application (app.py)

A Flask-based web interface enables real-time prediction:

User uploads a fingerprint image

Image is preprocessed and passed to the trained CNN model

Predicted blood group is displayed on the result page

This demonstrates end-to-end deployment capability, not just model training.

📊 Model Performance
🔹 Overall Accuracy
Accuracy: 45%

🔹 Class-wise Observations

Good performance for A+, A−, O+, O−

Lower accuracy for AB+ and AB− classes

📈 Results

Confusion Matrix → /results/confusion_matrix.png

Classification Report → /results/classification_report.txt

▶️ How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run the Flask Application
python app.py

3️⃣ Open Browser
http://127.0.0.1:5000/

Upload a fingerprint image to get the predicted blood group.

🔮Future Enhancements

Increase dataset size and balance classes

Apply Transfer Learning (ResNet, MobileNet, EfficientNet)

Use class weighting or focal loss

Extract fingerprint minutiae features

Improve prediction accuracy for rare blood groups
