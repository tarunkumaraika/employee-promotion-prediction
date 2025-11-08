# 🧠 Employee Promotion Prediction

This project focuses on predicting **employee promotions** using various **Machine Learning algorithms** on an **unbalanced dataset**. The goal is to help organizations identify which employees are most likely to get promoted, based on their performance, experience, and other factors.

---

## 🚀 Project Overview

In many companies, the promotion process is manual and biased. This project leverages **supervised machine learning models** to predict employee promotions automatically, especially when dealing with **imbalanced data distributions**.

The dataset used is highly unbalanced, meaning the number of promoted employees is significantly lower than those not promoted.  
To tackle this issue, **sampling techniques** and **algorithmic adjustments** were applied to achieve better prediction performance.

---

## ⚙️ Algorithms Used

- **Random Forest Classifier** 🌲  
- **Support Vector Machine (SVM)** ⚙️  
- **Logistic Regression** 📉  

Each model was trained, evaluated, and compared to determine the best-performing approach.

---

## 🧩 Key Features

- Handles **imbalanced datasets** effectively  
- Implements multiple classification algorithms  
- Evaluates model performance using metrics such as:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
- Includes hyperparameter tuning for better results

---

## 🧮 Tech Stack

- **Programming Language:** Python 🐍  
- **Libraries Used:**
  - pandas  
  - numpy  
  - scikit-learn  
  - matplotlib  
  - seaborn  
  - imbalanced-learn (SMOTE, etc.)

---




---

## 📁 Folder Structure
Employee_Promotion_Prediction/
│
├── data/ # Dataset files (raw and processed)
│ ├── train.csv
│ ├── test.csv
│
├── notebooks/ # Jupyter notebooks for analysis & model training
│ ├── 01_Data_Exploration.ipynb
│ ├── 02_Model_Training.ipynb
│ ├── 03_Model_Evaluation.ipynb
│
├── src/ # Python scripts used for the project
│ ├── data_preprocessing.py # Handles missing values, encoding, scaling, etc.
│ ├── feature_selection.py # Selects important features
│ ├── model_training.py # Trains Random Forest, SVM, Logistic Regression
│ ├── model_evaluation.py # Evaluates models using metrics and visualizations
│
├── models/ # Saved trained models
│ ├── random_forest_model.pkl
│ ├── svm_model.pkl
│ ├── logistic_regression_model.pkl
│
├── results/ # Evaluation results and visualizations
│ ├── confusion_matrix.png
│ ├── classification_report.txt
│ ├── accuracy_comparison.png
│
├── requirements.txt # Dependencies for the project
├── README.md # Project documentation
└── main.py # Main script to run the prediction pipeline



---

## 🔗 Model File

Due to GitHub’s file size limits, the trained model (`best_model.pkl`) is tracked using **Git LFS**.  
Alternatively, you can download it from:  
➡️ [Google Drive Link Here]([https://drive.google.com/](https://drive.google.com/file/d/1Db2mqDXVLzlFSiBQonqZqXOTYl7yT3rl/view?usp=sharing)) 

---

## 👨‍💻 Author

**AIKA Tarun Kumar**  
📍 Eluru, Andhra Pradesh  
📧 [tarunkumartakshye833@gmail.com](mailto:tarunkumartakshye833@gmail.com)  
📱 +91 9381057706  
🔗 [GitHub Profile](https://github.com/tarunkumaraika)

---

⭐ *If you like this project, please give it a star on GitHub!*

