import numpy as np
import pandas as pd
from pyswarms.single import GlobalBestPSO
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

################################################################
##todo: Find the optimal feature variable parameters
################################################################

result1 = [],
# Step 1: Load data
data1 = pd.read_excel("data/data_Li.xlsx")  # Replace with your Excel file path
data2 = pd.read_excel("data/data_selectivity.xlsx")  # Replace with your Excel file path
X1 = data1.iloc[:, :-1].values  # The first 12 columns as input features
y1 = data1.iloc[:, -1].values  # The last column as output target values

X2 = data2.iloc[:, :-1].values  # The first 12 columns as input features
y2 = data2.iloc[:, -1].values  # The last column as output target values

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
y1_scaled = scaler1.fit_transform(y1.reshape(-1, 1))
y2_scaled = scaler2.fit_transform(y2.reshape(-1, 1))
# Initialize global variables
best_iteration = -np.inf  # Initialize with a very small value
best_prediction1 = None
best_prediction2 = None
best_prediction1list = []
best_prediction2list = []

i = 0
def fitness(params):
    """
    PSO fitness function used to evaluate the performance of each feature combination.
    params: An n x d matrix where each row represents a feature combination [feature_1, feature_2, ..., feature_12]
    Returns:
    The negative of the objective value for each feature combination (PSO seeks to minimize, but we want to maximize the objective)
    """
    global best_prediction1, best_prediction2, best_iteration, best_prediction1list, best_prediction2list

    fitness_values = []
    for param in params:
        paramList = [
            1.4,
            20.43,
            round(param[0], 2),
            round(param[1], 2),
            round(param[2], 2),
            34,
            round(param[3], 2),
            round(param[4], 2),
        ]

        paramList = np.array(paramList)
        paramList = paramList.reshape(1, -1)  # Ensure the feature combination is a 2D array

        # Use Model 1 and Model 2 for prediction
        prediction1 = model1.predict(paramList)  # Prediction from the first model
        prediction2 = model2.predict(paramList)  # Prediction from the second model

        # Normalize
        # prediction1_scaled = scaler1.transform(prediction1.reshape(-1, 1))
        # prediction2_scaled = scaler2.transform(prediction2.reshape(-1, 1))

        # Combine results with weights
        weight1 = 0.3  # Weight of the first model
        weight2 = 0.7  # Weight of the second model
        final_prediction = weight1 * prediction1.reshape(-1, 1) + weight2 * prediction2.reshape(-1, 1)

        # Record the best prediction value
        if final_prediction[0][0] >= best_iteration:
            best_prediction1list.append(prediction1)
            best_prediction2list.append(prediction2)
            best_iteration = final_prediction[0][0]
            best_prediction1 = prediction1
            best_prediction2 = prediction2

        fitness_values.append(-final_prediction[0][0])  # Return negative for maximization

    return np.array(fitness_values)


# Initialize models
model1 = xgb.XGBRegressor(
    booster='gbtree',
    objective='reg:squarederror',
    learning_rate=0.146,
    max_depth=7,
    subsample=1,
    colsample_bytree=1,
    n_estimators=273,
    random_state=42
)

model2 = xgb.XGBRegressor(
    booster='gbtree',
    objective='reg:squarederror',
    learning_rate=0.15,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    n_estimators=200,
    random_state=42
)

# Train models with full data (models used to predict values for specific feature combinations)
model1.fit(X1, y1)  # Train model 1 with the full data
model2.fit(X2, y2)  # Train model 2 with the full data

# Step 3: Define feature ranges
lower_bounds = [0, 0, 0, 0, 0]  # Lower bounds for features
upper_bounds = [10, 80, 60, 400, 0.8]  # Upper bounds for features

# Step 4: Initialize PSO optimizer
optimizer = GlobalBestPSO(
    n_particles=40,  # Number of particles
    dimensions=4,  # Feature dimensions
    options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},  # PSO parameters
    bounds=(lower_bounds, upper_bounds)  # Feature value bounds
)

# Step 5: Run optimization
best_score, best_features = optimizer.optimize(fitness, iters=50)

# Output the optimal result
print("Maximum predicted value (objective):", -best_score)
print("Optimal feature combination:", best_features)

# Print best prediction results
print("Best prediction results:")
print("Prediction 1 (Model 1):", best_prediction1)
print("Prediction 2 (Model 2):", best_prediction2)

# Get the best fitness values for each iteration
iteration_best_scores = [-cost for cost in optimizer.cost_history]  # Convert negative values to positive (maximize objective)

# Plot the iteration curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(iteration_best_scores) + 1), iteration_best_scores, marker='o', linestyle='-', color='b')
plt.title('PSO Iteration Curve', fontsize=16)
plt.xlabel('Generation', fontsize=14)
# plt.ylabel('Rejection', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# Show the plot
plt.show()
