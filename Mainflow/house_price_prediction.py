import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load Dataset (Example synthetic dataset used here)
np.random.seed(42)
data = {
    "Size": np.random.randint(500, 4000, size=100),
    "Location": np.random.choice(["Urban", "Suburban", "Rural"], size=100),
    "Number of Rooms": np.random.randint(1, 10, size=100),
    "Price": np.random.randint(100000, 1000000, size=100)
}
df = pd.DataFrame(data)

# Step 2: Preprocessing
X = df.drop("Price", axis=1)
y = df["Price"]

categorical_features = ["Location"]
numerical_features = ["Size", "Number of Rooms"]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop="first")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

# Step 3: Build Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training
pipeline.fit(X_train, y_train)

# Step 6: Prediction & Evaluation
y_pred = pipeline.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 7: Visualization
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.grid(True)
plt.tight_layout()
plt.show()
