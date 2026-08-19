# 🍷 Wine Classification using Logistic Regression

## 📌 Project Overview

This project uses **Logistic Regression** to classify wines into different wine classes based on their chemical properties.

The project demonstrates a complete **Machine Learning classification workflow**, including data preprocessing, feature scaling, model training, evaluation, and model deployment.

---

## 🎯 Objective

The main objective is to predict the **class of wine** using its chemical characteristics.

The project uses **Logistic Regression**, a supervised machine learning classification algorithm.

---

## 📂 Project Files

```text
Wine-Logistic-Regression/
│
├── wine_dataset.csv
├── wine.pkl
├── scaler.pkl
├── app.py
├── requirements.txt
└── README.md
```

### File Description

| File               | Description                             |
| ------------------ | --------------------------------------- |
| `wine_dataset.csv` | Wine dataset used for training/testing  |
| `wine(3).pkl`      | Trained Logistic Regression model       |
| `scaler(2).pkl`    | StandardScaler used for feature scaling |
| `app.py`           | Streamlit application for prediction    |
| `requirements.txt` | Required Python libraries               |
| `README.md`        | Project documentation                   |

---

## 🧠 Machine Learning Algorithm

### Logistic Regression

Logistic Regression is a **supervised classification algorithm** used to predict categorical outcomes.

For this project, Logistic Regression is used for **multi-class classification** to predict the wine category.

### Why Logistic Regression?

* Simple and easy to understand
* Fast to train
* Works well for classification problems
* Supports multi-class classification
* Provides a good baseline classification model

---

## 📊 Dataset

The dataset contains chemical measurements of different wines.

The features represent properties such as:

* Alcohol
* Malic Acid
* Ash
* Alcalinity of Ash
* Magnesium
* Total Phenols
* Flavanoids
* Nonflavanoid Phenols
* Proanthocyanins
* Color Intensity
* Hue
* OD280/OD315 of Diluted Wines
* Proline

The target variable represents the **wine class**.

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Save Model (.pkl)
   ↓
Streamlit Deployment
```

---

## ⚙️ Data Preprocessing

The dataset is first separated into:

* **X** → Input features
* **y** → Target variable

The data is then divided into training and testing datasets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 📏 Feature Scaling

Since the features have different numerical ranges, **StandardScaler** is used.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is saved so that the same transformation can be applied to new input data during deployment.

---

## 🤖 Model Training

Logistic Regression is trained using the scaled training data.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train_scaled, y_train)
```

The trained model is saved as a `.pkl` file.

---

## 📈 Model Evaluation

The trained model can be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report

Example:

```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## 💾 Model Serialization

The trained Logistic Regression model and scaler are saved using **Pickle**.

```python
import pickle

with open("wine(3).pkl", "wb") as file:
    pickle.dump(model, file)

with open("scaler(2).pkl", "wb") as file:
    pickle.dump(scaler, file)
```

These files can later be loaded during deployment without retraining the model.

---

## 🌐 Deployment using Streamlit

The trained model can be deployed using **Streamlit**.

The application takes wine feature values from the user, scales the input using the saved scaler, and sends it to the trained Logistic Regression model.

```text
User Input
    ↓
Streamlit App
    ↓
Load Scaler
    ↓
Scale Input
    ↓
Load Logistic Regression Model
    ↓
Predict Wine Class
    ↓
Display Result
```

Run the application using:

```bash
streamlit run app.py
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Pickle
* Streamlit
* Matplotlib
* Seaborn

---

## 📦 Installation

Clone the repository and install the required libraries:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

---

## 🔮 Prediction

The application allows the user to enter the chemical properties of a wine.

The trained model predicts the corresponding **wine class**.

Example:

```text
Input Wine Features
        ↓
Feature Scaling
        ↓
Logistic Regression Model
        ↓
Predicted Wine Class
```

---

## 📌 Key Concepts Covered

* Supervised Learning
* Classification
* Logistic Regression
* Multi-Class Classification
* Train-Test Split
* Feature Scaling
* StandardScaler
* Model Training
* Model Evaluation
* Confusion Matrix
* Classification Report
* Pickle Model Serialization
* Streamlit Deployment

---

## 🚀 Future Improvements

* Compare Logistic Regression with other classification algorithms
* Add interactive visualizations
* Improve the Streamlit user interface
* Add probability scores for predictions
* Deploy the application online
* Perform hyperparameter tuning

---

## 👩‍💻 Author

**Aditi Tawade**

Aspiring Data Scientist | Python | Machine Learning | AI

---

## ⭐ Project Purpose

This project is part of my **Machine Learning practice and portfolio projects**, where I am building practical projects to strengthen my understanding of Python, Data Science, Machine Learning, and model deployment.
