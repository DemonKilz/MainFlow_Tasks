# 🏠 House Price Prediction using Linear Regression

This project is part of my internship at **Main Flow Services and Technologies Pvt. Ltd.**. The task involves building a regression model to predict house prices based on features such as size, location, and number of rooms.

---

## 📌 Objective

To develop a predictive model using **Linear Regression** that estimates house prices using real estate data and to analyze how each feature contributes to pricing.

---

## 📊 Dataset Overview

As part of the internship, a sample dataset (synthetically generated for illustration) includes the following columns:

- `Size` – Size of the house in square feet
- `Location` – Categorical feature with values: Urban, Suburban, Rural
- `Number of Rooms` – Total number of rooms in the house
- `Price` – Target variable (price of the house in USD)

---

## 🛠️ Tools & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🔍 Steps Followed

1. **Data Loading & Inspection**
   - Loaded the dataset into a Pandas DataFrame.
   - Reviewed column types and distributions.

2. **Preprocessing**
   - Applied **StandardScaler** to numerical features.
   - Used **One-Hot Encoding** for the categorical `Location` feature.

3. **Model Building**
   - Created a pipeline using Scikit-learn with preprocessing and `LinearRegression`.

4. **Model Evaluation**
   - Evaluated model performance using:
     - **Root Mean Square Error (RMSE)**
     - **R² Score**

5. **Visualization**
   - Created a scatter plot of actual vs. predicted prices to visualize model performance.

---

## 📈 Results

| Metric         | Value           |
|----------------|------------------|
| RMSE           | ~256,192         |
| R² Score       | ~-0.24 (due to synthetic/random data) |

> ⚠️ Note: The model is trained on a randomly generated dataset. Real-world data would yield more meaningful metrics.

---

## 📂 Files

- `house_price_prediction.py` – Main script for preprocessing, training, and evaluation
- `README.md` – Documentation of the task

---

## 🔚 Conclusion

Through this task, I gained hands-on experience in regression modeling, feature engineering, pipeline construction, and performance evaluation using Scikit-learn. It reinforced the importance of preprocessing and data quality in predictive modeling.

---

