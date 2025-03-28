import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, \
    BaggingRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, BayesianRidge, SGDRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import ngboost
import joblib
from sklearn.tree import DecisionTreeRegressor
from itertools import combinations
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# Disable warnings when numpy encounters floating-point overflow
np.seterr(over='ignore')


################################################################

# todo: Model training section


################################################################


# Define an average model
class AverageModel:
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return np.mean(predictions, axis=0)


# Read your Excel file, replace with your actual file path
file_path = 'data/data.xlsx'  # Replace 'your_dataset.xlsx' with your actual file name or path
df = pd.read_excel(file_path)
# Split into training and testing datasets in an 80:20 ratio
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Read training data
df = train_df

Y_train = df.iloc[:, -1]
X_train = df.iloc[:, :-1]

# Initialize the scaler
# scaler = MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = X_train
# Read and normalize test data
df_test = test_df
Y1_test = df_test.iloc[:, -1]
X_test = df_test.iloc[:, :-1]
# X_test_scaled = scaler.transform(X_test)
X_test_scaled = X_test
# Define hyperparameter grids
param_grids = {
    'rf': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]},
    'dt': {'max_depth': [None, 10, 20]},
    'lgb': {'n_estimators': [100, 300], 'learning_rate': [0.01, 0.1]},
    'xgb': {'n_estimators': [100, 316, 300], 'learning_rate': [0.01, 0.215, 0.1], 'max_depth': [4, 7, 8], 'subsample': [0.73], 'colsample_bytree': [0.742]},
    'cat': {'iterations': [100, 300], 'learning_rate': [0.01, 0.1]},
    'ngb': {'n_estimators': [100, 300]},
    'lin_reg': {},  # Linear Regression has no hyperparameters
    'svr': {'C': [0.1, 1], 'gamma': ['scale', 'auto']},
    'gbr': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
    'knn': {'n_neighbors': [3, 5, 7]},
    'bayesian': {},  # Bayesian Ridge has no hyperparameters
    'extra_trees': {'n_estimators': [100, 200]},
    'mlp': {'hidden_layer_sizes': [(50, 50), (100, 50)], 'learning_rate_init': [0.001, 0.01]},
    'ada': {'n_estimators': [50, 100]},
    'bagging': {'n_estimators': [10, 50]},
    'kernel_ridge': {'alpha': [0.1, 1]},
    'sgd': {'alpha': [0.0001, 0.001], 'penalty': ['l2', 'l1']},
    'ridge': {'alpha': [0.1, 1, 10]},  # Ridge hyperparameter grid
    'lasso': {'alpha': [0.01, 0.1, 1]},  # Lasso hyperparameter grid
}

# Initialize models
models = {
    'rf': RandomForestRegressor(random_state=42),
    'dt': DecisionTreeRegressor(random_state=42),
    'lgb': lgb.LGBMRegressor(random_state=42),
    'xgb': xgb.XGBRegressor(random_state=42),
    'cat': CatBoostRegressor(random_state=42, verbose=0),
    'ngb': ngboost.NGBRegressor(random_state=42),
    'lin_reg': LinearRegression(),
    'svr': SVR(),
    'gbr': GradientBoostingRegressor(random_state=42),
    'knn': KNeighborsRegressor(),
    'bayesian': BayesianRidge(),
    'extra_trees': ExtraTreesRegressor(random_state=42),
    'mlp': MLPRegressor(random_state=42),
    'ada': AdaBoostRegressor(random_state=42),
    'bagging': BaggingRegressor(base_estimator=DecisionTreeRegressor(), random_state=42),
    'kernel_ridge': KernelRidge(),
    'sgd': SGDRegressor(random_state=42),
    'ridge': Ridge(),
    'lasso': Lasso(),
}

model_names = list(models.keys())

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# Train and save models
def save_model(model, model_name):
    file_name = f"models/{model_name}.joblib"
    joblib.dump(model, file_name)
    print(f"Model {model_name} saved as {file_name}")


# Load model
def load_model(model_name):
    file_name = f"models/{model_name}.joblib"
    return joblib.load(file_name)

def plot_regression_confusion_matrix(y_test, y_pred):
    bins = np.linspace(np.min(y_test), np.max(y_test), 6) # Divide the range into 5 intervals
    bin_labels = [f'{bins[i]:.1f}-{bins[i + 1]:.1f}' for i in range(len(bins) - 1)] # Interval labels

    # Map true and predicted values to intervals
    true_bins = np.digitize(y_test, bins) - 1
    pred_bins = np.digitize(y_pred, bins) - 1

    # Construct confusion matrix
    conf_matrix = np.zeros((len(bin_labels), len(bin_labels)), dtype=int)
    for t, p in zip(true_bins, pred_bins):
        if 0 <= t < len(bin_labels) and 0 <= p < len(bin_labels):
            conf_matrix[t, p] += 1

    # Plot confusion matrix
    plt.figure(figsize=(8, 8))
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title('Regression Confusion Matrix')
    plt.xlabel('Predicted Interval')
    plt.ylabel('True Interval')
    plt.xticks(range(len(bin_labels)), bin_labels, rotation=45)
    plt.yticks(range(len(bin_labels)), bin_labels)

    # Annotate numbers in the matrix
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, str(conf_matrix[i, j]),
            ha='center', va='center',
            color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')
    plt.tight_layout()
    plt.show()


# List to store results
results = []

# Store results for single models, pairwise combinations, and triple combinations
combination_results = []

# Grid search and training for single models
print("\nTraining and testing results for single models:")
for name, model in models.items():
    # Use GridSearchCV to find optimal parameters
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=kf, scoring='r2', n_jobs=1)
    grid_search.fit(X_train_scaled, Y_train)

    # Best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters for {name}: {grid_search.best_params_}")

    # Save the model
    save_model(best_model, name)

    # Make predictions
    y_pred_train = best_model.predict(X_train_scaled)
    y_pred_test = best_model.predict(X_test_scaled)

    # Calculate performance metrics
    r2_train = r2_score(Y_train, y_pred_train)
    r2_test = r2_score(Y1_test, y_pred_test)
    mae_test = mean_absolute_error(Y1_test, y_pred_test)
    mae_train = mean_absolute_error(Y_train, y_pred_train)
    print(f"Single model {name} - Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}, Test MAE: {mae_test:.4f}")

    # Store results
    results.append({
        'type': 'Single model',
        'model_combination': (name,),
        'r2_train': r2_train,
        'r2_test': r2_test,
        'mae_test': mae_test,
        'mae_train': mae_train,
        'y_pred_test': y_pred_test,
        'y_pred_train': y_pred_train
    })

# Save all results to DataFrame
results_df = pd.DataFrame(results)
folder_path = 'results'
file_path_csv = os.path.join(folder_path, 'model_results_sample.csv')
# Save DataFrame to CSV file
# results_df.to_csv(file_path_csv, index=False)

# Sort by R² and select top five models of each category
top_single_models = results_df[results_df['type'] == 'Single model'].sort_values(by='r2_test', ascending=False)


# Plot top models including training and testing sets
def plot_top_models(models_df, title):
    for idx, result in models_df.iterrows():
        plt.figure(figsize=(10, 8))

        # Plot training set
        plt.scatter(Y_train, result['y_pred_train'], s=200, alpha=0.7, c='#4A90E2', label="Train Set", marker='o',
                    edgecolors='white')

        # Plot test set
        plt.scatter(Y1_test, result['y_pred_test'], s=200, alpha=0.7, c='#F55587', label="Test Set", marker='o', edgecolors='white')

        # Plot ideal fit line
        plt.plot([min(min(Y1_test), min(Y_train)), max(max(Y1_test), max(Y_train))],
                 [min(min(Y1_test), min(Y_train)), max(max(Y1_test), max(Y_train))],
                 'r--', lw=2, label="Perfect Fit")

        # Set axis labels and increase font size
        plt.xlabel('Actual Rejection', fontsize=35)
        plt.ylabel('Predicted Rejection', fontsize=35)

        # Set plot title and increase font size
        plt.title(f'{title} ({", ".join(result["model_combination"])})', fontsize=20)

        # Display R² value for the test set in the bottom-right corner
        textstr = f'R² (Test) = {result["r2_test"]:.4f}'
        plt.gcf().text(0.95, 0.20, textstr, fontsize=30, ha='right', va='bottom')

        # Display R² value for the train set in the bottom-right corner
        trainstr = f'R² (Train) = {result["r2_train"]:.4f}'
        plt.gcf().text(0.95, 0.30, trainstr, fontsize=30, ha='right', va='bottom')

        # Adjust legend size and position, upper-left corner, and increase font size
        # plt.legend(fontsize=30, loc='upper left', bbox_to_anchor=(0, 1), frameon=False)
        # Adjust x and y axis limits

        # Set axis tick font size
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        # Set gridlines
        plt.grid(False)

        # Show plot
        plt.show()

# Plot the top five models in each category
plot_top_models(top_single_models, "model")
