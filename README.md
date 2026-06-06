# 🎥 YouTube Adview Prediction using Machine Learning

## 📌 Project Overview

YouTube advertisers pay content creators based on ad views and clicks generated through advertisements displayed on videos. Predicting the number of ad views can help advertisers estimate campaign performance and optimize marketing strategies.

This project develops and compares multiple Machine Learning regression models to predict **YouTube Adviews** using video engagement metrics such as views, likes, dislikes, comments, duration, category, and publication information.

---

## 🎯 Objective

To build a Machine Learning regression model capable of accurately predicting the **number of YouTube ad views** based on other available YouTube video metrics.

---

## 📂 Dataset

The dataset contains approximately **13,000 YouTube videos** with various engagement and metadata features.

### Features

| Feature   | Description                          |
| --------- | ------------------------------------ |
| vidid     | Unique Video ID                      |
| views     | Number of video views                |
| likes     | Number of likes                      |
| dislikes  | Number of dislikes                   |
| comment   | Number of comments                   |
| published | Video publication date               |
| duration  | Video duration                       |
| category  | Video category                       |
| adview    | Number of ad views (Target Variable) |

### Dataset Shape

* Rows: ~13,000
* Columns: 9

---

## 🛠 Data Preprocessing

### 1. Missing Value Handling

* Identified null values using:

```python
train_df.isnull()
```

* Removed missing values using:

```python
train_df.dropna()
```

### 2. Duplicate Removal

```python
train_df.drop_duplicates()
```

### 3. Duration Conversion

Converted YouTube duration format:

```text
PT1H2M3S
```

into total seconds for numerical processing.

Example:

```text
PT1H2M3S → 3723 seconds
```

### 4. Feature Transformation

#### Published Year Extraction

Extracted year from the published date column:

```python
train_df['year']
```

#### Numeric Conversion

Converted engagement metrics into numerical format for analysis and modeling.

---

## 📊 Exploratory Data Analysis (EDA)

### Univariate Analysis

Histograms were used to analyze distributions of:

* Adviews
* Views
* Likes
* Dislikes
* Comments
* Publication Year

### Key Findings

* Most features exhibit a **right-skewed distribution**.
* A small number of videos generate extremely high engagement.
* Significant outliers exist in:

  * Views
  * Adviews
  * Likes
  * Comments

### Bivariate Analysis

Correlation heatmaps were generated to study relationships among features.

### Observations

#### Strong Positive Correlations

* Views ↔ Likes
* Views ↔ Comments
* Likes ↔ Comments

#### Moderate Positive Correlations

* Adviews ↔ Views
* Adviews ↔ Likes
* Adviews ↔ Comments

#### Weak Correlations

* Duration
* Category
* Year

---

## 🔄 Data Normalization

Min-Max Scaling was applied to:

```python
[
'adview',
'views',
'likes',
'dislikes',
'comment'
]
```

### Benefits

* Scales data between 0 and 1
* Reduces impact of large-value features
* Improves model convergence
* Enhances comparability among features

---

## 🎯 Feature Selection (Dimensionality Reduction)

The following features were removed:

### vidid

* Unique identifier
* No predictive value

### published

* Already transformed into year

### category

* Weak relationship with target variable

Selected features improved model efficiency and reduced noise.

---

# 🤖 Machine Learning Models

## 1. Linear Regression

### Purpose

Models a linear relationship between input features and ad views.

### Advantages

* Simple
* Fast
* Interpretable

### Evaluation

| Metric | Value        |
| ------ | ------------ |
| MAE    | 1005.56      |
| MSE    | 9,102,470.09 |
| RMSE   | 3017.03      |

---

## 2. Support Vector Regressor (SVR)

### Purpose

Captures linear and non-linear relationships using support vectors.

### Advantages

* Robust to outliers
* Effective for high-dimensional data

### Evaluation

| Metric | Value        |
| ------ | ------------ |
| MAE    | 1005.56      |
| MSE    | 9,102,470.09 |
| RMSE   | 3017.03      |

---

## 3. Decision Tree Regressor

### Purpose

Creates tree-based decision rules to predict ad views.

### Advantages

* Easy interpretation
* Handles non-linearity

### Evaluation

| Metric | Value        |
| ------ | ------------ |
| MAE    | 1011.97      |
| MSE    | 9,437,112.74 |
| RMSE   | 3071.99      |

---

## 4. Random Forest Regressor

### Purpose

Ensemble of multiple decision trees to improve prediction performance.

### Advantages

* Reduces overfitting
* Higher accuracy
* Better generalization

### Evaluation

| Metric | Value        |
| ------ | ------------ |
| MAE    | 1009.04      |
| MSE    | 8,330,953.35 |
| RMSE   | 2886.34      |

### Best Traditional ML Model ✅

Random Forest achieved the lowest prediction error among all traditional machine learning models.

---

## 5. Artificial Neural Network (ANN)

### Architecture

* Dense Neural Network
* Total Parameters: 257
* Trainable Parameters: 85
* Optimizer: Adam

### Evaluation

| Metric | Value         |
| ------ | ------------- |
| MAE    | 2931.34       |
| MSE    | 34,772,052.94 |
| RMSE   | 5896.78       |

### Observation

The ANN performed significantly worse than the traditional machine learning models, suggesting:

* Insufficient data size for deep learning
* Potential need for feature engineering
* Hyperparameter tuning required

---

# 📈 Model Comparison

| Model             | MAE     | MSE           | RMSE        |
| ----------------- | ------- | ------------- | ----------- |
| Linear Regression | 1005.56 | 9,102,470.09  | 3017.03     |
| SVR               | 1005.56 | 9,102,470.09  | 3017.03     |
| Decision Tree     | 1011.97 | 9,437,112.74  | 3071.99     |
| Random Forest     | 1009.04 | 8,330,953.35  | **2886.34** |
| ANN               | 2931.34 | 34,772,052.94 | 5896.78     |

---

# 🏆 Results

### Best Performing Model

✅ **Random Forest Regressor**

Reason:

* Lowest RMSE
* Lowest MSE
* Better generalization capability
* Reduced overfitting through ensemble learning

---

# 🚀 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* TensorFlow / Keras
* Jupyter Notebook

---

# 📁 Project Structure

```text
YouTube-Adview-Prediction/
│
├── Dataset/
│   └── train.csv
│
├── notebooks/
│   └── YouTube_Adview_Prediction.ipynb
│
├── images/
│   ├── before_normalization.png
│   ├── after_normalization.png
│   ├── correlation_heatmap.png
│   └── model_comparison.png
│
├── README.md
└── requirements.txt
```

---

# 🔮 Future Improvements

* Hyperparameter tuning using Grid Search and Random Search
* XGBoost Regressor implementation
* LightGBM implementation
* Feature importance analysis
* Deep learning architecture optimization
* Deployment using Flask/FastAPI
* Real-time prediction dashboard

---

# 📜 Conclusion

This project demonstrates how Machine Learning can be used to predict YouTube ad views using engagement metrics and video metadata. Multiple regression models were evaluated, with the **Random Forest Regressor achieving the best overall performance**. The project highlights the importance of data preprocessing, normalization, feature selection, and model comparison in building effective predictive systems.

---

## 👨‍💻 Author

**Shahzaib Saleem**  

* Data Science Student
* FAST National University of Computer & Emerging Sciences (FAST-NUCES)
* GitHub: https://github.com/Shahzaib-S110
* LinkedIn: https://linkedin.com/in/shahzaib-saleem-110cci

  **Umer Khalid**
  **Moaz Kashif**
