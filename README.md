
# AI-Based Heart Disease Early Detection

This project leverages machine learning algorithms to predict whether a person is at risk of heart disease based on health-related attributes like age, cholesterol, blood pressure, etc. It aims to provide a non-invasive, accurate, and accessible diagnosis aid, especially useful in remote or underserved areas.

---

# Objectives

- Create a machine learning-based predictive model for early detection of heart disease.
- Preprocess and analyze health data for model training.
- Compare multiple classification algorithms: KNN, SVM, Decision Tree, Random Forest.
- Identify the most accurate model for practical implementation.
- Visualize data relationships and classifier performance.

---

## 📊 Dataset Overview

- **Source**: UCI Heart Disease Dataset
- **Samples**: 303
- **Features**:
  - `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `target`
  - `target`: 1 = presence of heart disease, 0 = no disease

---

## ⚙️ Technologies Used

- Python 3
- Libraries: `NumPy`, `Pandas`, `Matplotlib`, `Scikit-learn`

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/AI-Heart-Disease-Detection.git
cd AI-Heart-Disease-Detection
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
AI-Heart-Disease-Detection/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── heart_disease_detection.ipynb
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 🧠 Machine Learning Models

We used and evaluated the following classification algorithms:
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)** with different kernels
- **Decision Tree Classifier**
- **Random Forest Classifier**

---

## 📈 Model Evaluation

| Model                  | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| KNN (k=8)              | **91.80%** | 0.92      | 0.90   | 0.91     |
| SVM (Linear Kernel)    | 86.88%   | 0.88      | 0.86   | 0.87     |
| Decision Tree Classifier | 78.69% | 0.79      | 0.78   | 0.78     |
| Random Forest (100 Est.)| 86.88%   | 0.87      | 0.87   | 0.87     |

**Conclusion**: KNN with `k=8` delivered the best performance.

---

## 🔍 Confusion Matrix (KNN)

```
              Predicted
              0    1
Actual   0   25    3
         1    2   31
```

---

## 📊 Visualizations

- ✔️ **Correlation Heatmap** – Identifies relationships between features.
- ✔️ **Target Class Distribution** – Number of people with and without heart disease.
- ✔️ **Model Accuracy Comparison** – Bar chart comparing KNN, SVM, DT, RF.
- ✔️ **KNN Performance vs k** – Line graph to select the best `k` value.
- ✔️ **Decision Tree Accuracy vs Features** – Effect of `max_features` on accuracy.

Images saved in `/results/`

---

## 🔮 Future Scope

- **Real-time Health Monitoring**: Integration with smartwatches and IoT sensors.
- **Web/Mobile App**: Build accessible platforms for regular screenings.
- **Deep Learning Models**: Use CNNs or LSTMs on ECG waveform data.
- **Expanded Dataset**: Include genetic, lifestyle, and demographic diversity.
- **Telemedicine Integration**: Enable remote diagnostics and support.
- **Personalized Risk Prediction**: Tailor forecasts to individual profiles.

---

## 👨‍💻 Team Members

- Ritika Singh – 22BAI70256  
- Dev Kumar Singh – 22BAI70518  
- Aryan Malgotra – 22BAI70258  
- Nikhil Kumawat – 22BAI71370

**Supervisor**: Mrs. Anudeep Kaur  
**Institution**: Chandigarh University  
**Department**: AIT – CSE (Artificial Intelligence & Machine Learning)  
**Submission Date**: April 2025

---

## 📚 References

1. Smith, J. et al. “AI in Healthcare: Advances and Applications.” *Journal of Medical AI*, 2021.  
2. Fatima, M. & Pasha, M. “Survey of Machine Learning Algorithms for Disease Diagnosis”, 2017.  
3. Pahwa, K. & Kumar, R. “Heart Disease Prediction Using Hybrid Techniques”, *IEEE UPCON*, 2017.  
4. UCI Machine Learning Repository: Heart Disease Dataset.

