from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from pyswarm import pso
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


################################

##todo: Find the best parameters for the XGBoost model using PSO

################################
# Step 1: Load data
# Data loading and feature-target separation
# Replace with your own file path
file_path = '../data/mydata/data_Li.xlsx'  # Replace 'your_dataset.xlsx' with the actual file name or path
df = pd.read_excel(file_path)
# Split into training and test datasets with an 80:20 ratio
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Read training data
df = train_df

target_train = df.iloc[:, -1]
X_train = df.iloc[:, :-1]

# Initialize the scaler
scaler = MinMaxScaler()
feature_train = scaler.fit_transform(X_train)

# Read and normalize test data
df_test = test_df
target_test = df_test.iloc[:, -1]
X_test = df_test.iloc[:, :-1]
feature_test = scaler.transform(X_test)



# Step 2: Define the objective function (fitness function)
def fitness(params):
    """
    PSO fitness function to evaluate the performance of parameter combinations.
    Input parameters:
    params: Contains XGBoost hyperparameters [learning_rate, n_estimators, max_depth, subsample, colsample_bytree]
    Output:
    Returns the negative R2 score (because PSO defaults to minimizing, we want to maximize R2)
    """
    learning_rate, n_estimators, max_depth, subsample, colsample_bytree = params

    # XGBoost model
    model = xgb.XGBRegressor(
        booster='gbtree',
        objective='reg:squarederror',
        learning_rate=max(0.01, float(learning_rate)),  # Avoid 0
        n_estimators=int(max(10, n_estimators)),  # Avoid less than 10
        max_depth=int(max(1, max_depth)),  # Avoid less than 1
        subsample=max(0.1, min(1.0, subsample)),  # Range [0.1, 1.0]
        colsample_bytree=max(0.1, min(1.0, colsample_bytree)),  # Range [0.1, 1.0]
        random_state=42
    )

    # Model training
    model.fit(feature_train, target_train.ravel())
    predictions = model.predict(feature_test)

    # Evaluate R2 score
    r2 = r2_score(target_test, predictions)
    return -r2  # PSO needs to minimize the objective function


# Step 3: Set the parameter ranges for PSO
# Hyperparameter ranges
lb = [0.01, 50, 3, 0.1, 0.1]  # Lower bounds for [learning_rate, n_estimators, max_depth, subsample, colsample_bytree]
ub = [0.3, 300, 10, 1.0, 1.0]  # Upper bounds

# Step 4: Use PSO to optimize XGBoost hyperparameters
best_params, best_score = pso(fitness, lb, ub, swarmsize=20, maxiter=400)
best_params_rounded = np.round(best_params, 2)
print("Optimal parameters:", best_params_rounded)
print("Best R2 (negative value):", -best_score)

# Step 5: Retrain the XGBoost model with the best parameters
optimized_model = xgb.XGBRegressor(
    booster='gbtree',
    objective='reg:squarederror',
    learning_rate=best_params[0],
    n_estimators=int(best_params[1]),
    max_depth=int(best_params[2]),
    subsample=best_params[3],
    colsample_bytree=best_params[4],
    random_state=42
)

optimized_model.fit(feature_train, target_train.ravel())
final_predictions = optimized_model.predict(feature_test)

# Step 6: Evaluate the performance of the optimized model
print("Final model evaluation:")
print("EVS:", explained_variance_score(target_test, final_predictions))
print("R2:", r2_score(target_test, final_predictions))
print("MSE:", mean_squared_error(target_test, final_predictions))
print("MAE:", mean_absolute_error(target_test, final_predictions))
print("RMSE:", np.sqrt(mean_squared_error(target_test, final_predictions)))


# Visualize the results
#plot_picture(feature_train, target_train.ravel(), target_test, final_predictions, r2_score(target_test, final_predictions))
# plt.figure(figsize=(12, 6))
# plt.plot(target_test, label='True Values')
# plt.plot(final_predictions, label='Predictions')
# plt.legend()
# plt.title("XGBoost with PSO Optimization")
# plt.show()
