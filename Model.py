# ============================================================
# XGBOOST REGRESSION — MATH-ALIGNED IMPLEMENTATION
# ============================================================

import numpy as np
import xgboost as xgb

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------
X, y = fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to DMatrix (XGBoost internal structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test, label=y_test)

# ------------------------------------------------------------
# 2. Parameters (DIRECTLY MAP TO MATH)
# ------------------------------------------------------------
params = {
    "objective": "reg:squarederror",  # squared loss
    "eta": 0.05,                      # learning rate (η)
    "max_depth": 4,                   # tree depth
    "lambda": 1.0,                    # L2 regularization (λ)
    "gamma": 0.0,                     # leaf penalty (γ)
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "rmse"
}

num_rounds = 200

# ------------------------------------------------------------
# 3. Train XGBoost Model
# ------------------------------------------------------------
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_rounds
)

# ------------------------------------------------------------
# 4. Predict
# ------------------------------------------------------------
y_pred = model.predict(dtest)

# ------------------------------------------------------------
# 5. Evaluation
# ------------------------------------------------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation")
print("----------------")
print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)

# ------------------------------------------------------------
# 6. Predict on a NEW Unseen Data Point
# ------------------------------------------------------------
new_house = np.array([[
    8.3252,   # MedInc
    41,       # HouseAge
    6.984,    # AveRooms
    1.023,    # AveBedrms
    322,      # Population
    2.555,    # AveOccup
    37.88,    # Latitude
    -122.23   # Longitude
]])

dnew = xgb.DMatrix(new_house)
new_pred = model.predict(dnew)

print("\nNew House Prediction")
print("--------------------")
print("Predicted Median House Value:", new_pred[0])