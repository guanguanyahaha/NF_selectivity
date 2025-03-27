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
# 设置 numpy 在遇到浮点数溢出时不发出警告
np.seterr(over='ignore')


################################################################

# todo 模型训练部分


################################################################


# 定义平均模型
class AverageModel:
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return np.mean(predictions, axis=0)


# 读取你的Excel文件，替换为你自己的文件路径
file_path = 'data/mydata/data_permeation.xlsx'  # 将'your_dataset.xlsx'替换为实际的文件名或路径
df = pd.read_excel(file_path)
# 按照8:2的比例分割为训练集和测试集 42
train_df, test_df = train_test_split(df, test_size=0.2, random_state=333)

# 读取训练数据
df = train_df

Y_train = df.iloc[:, -1]
X_train = df.iloc[:, :-1]

# 初始化归一化器
# scaler = MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled=X_train
# 测试数据读取和归一化
df_test = test_df
Y1_test = df_test.iloc[:, -1]
X_test = df_test.iloc[:, :-1]
# X_test_scaled = scaler.transform(X_test)
X_test_scaled=X_test
# 定义超参数网格
param_grids = {
    'rf': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]},
    'dt': {'max_depth': [None, 10, 20]},
    'lgb': {'n_estimators': [100, 300], 'learning_rate': [0.01, 0.1]},
    # 'xgb': {'n_estimators': [100, 316, 300], 'learning_rate': [0.01,0.215, 0.1], 'max_depth': [4, 7, 8], 'subsample': [0.73],'colsample_bytree': [0.742]},
    'xgb': {'n_estimators': [289], 'learning_rate': [0.23], 'max_depth': [7], 'subsample': [0.71],
            'colsample_bytree': [0.75]},
    # 'xgb': {
    #     'colsample_bytree': [0.742, 0.75],
    #     'gamma': [0.065, 0.07],
    #     'learning_rate': [0.215, 0.22],
    #     'max_depth': [4, 5],
    #     'min_child_weight': [3.67, 4],
    #     'n_estimators': [316, 320],
    #     'reg_alpha': [0.856, 0.86],
    #     'reg_lambda': [0.13, 0.14],
    #     'subsample': [0.73, 0.75]
    # }
    'cat': {'iterations': [100, 300], 'learning_rate': [0.01, 0.1]},
    'ngb': {'n_estimators': [100, 300]},
    'lin_reg': {},  # 线性回归没有超参数
    'svr': {'C': [0.1, 1], 'gamma': ['scale', 'auto']},
    'gbr': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
    'knn': {'n_neighbors': [3, 5, 7]},
    'bayesian': {},  # Bayesian Ridge 没有超参数
    'extra_trees': {'n_estimators': [100, 200]},
    'mlp': {'hidden_layer_sizes': [(50, 50), (100, 50)], 'learning_rate_init': [0.001, 0.01]},
    'ada': {'n_estimators': [50, 100]},
    'bagging': {'n_estimators': [10, 50]},
    'kernel_ridge': {'alpha': [0.1, 1]},
    'sgd': {'alpha': [0.0001, 0.001], 'penalty': ['l2', 'l1']},
    'ridge': {'alpha': [0.1, 1, 10]},  # Ridge 超参数网格
    'lasso': {'alpha': [0.01, 0.1, 1]},  # Lasso 超参数网格
    # 'random_forest': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]}

}

# 初始化模型
# 初始化模型
models = {
    # 'rf': RandomForestRegressor(random_state=42),
    # 'dt': DecisionTreeRegressor(random_state=42),
    # 'lgb': lgb.LGBMRegressor(random_state=42),
    'xgb': xgb.XGBRegressor(random_state=42),  # xgb
    # 'cat': CatBoostRegressor(random_state=42, verbose=0),
    # 'ngb': ngboost.NGBRegressor(random_state=42),
    # 'lin_reg': LinearRegression(),
    # 'svr': SVR(),  # SVR
    # 'gbr': GradientBoostingRegressor(random_state=42),
    # 'knn': KNeighborsRegressor(),
    # 'bayesian': BayesianRidge(),
    # 'extra_trees': ExtraTreesRegressor(random_state=42),
    # 'mlp': MLPRegressor(random_state=42),
    # 'ada': AdaBoostRegressor(random_state=42),
    # 'bagging': BaggingRegressor(base_estimator=DecisionTreeRegressor(), random_state=42),
    # 'kernel_ridge': KernelRidge(),  # KRR
    # 'sgd': SGDRegressor(random_state=42),
    # 'ridge': Ridge(),  # 新增的 Ridge Regression
    # 'lasso': Lasso(),  # 新增的 Lasso Regression
    # 'random_forest': RandomForestRegressor(random_state=42)  # 新增的 Random Forest
}

model_names = list(models.keys())

# 初始化 KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# 训练并保存模型
def save_model(model, model_name):
    file_name = f"models/{model_name}.joblib"
    joblib.dump(model, file_name)
    print(f"模型 {model_name} 已保存为 {file_name}")


# 加载模型
def load_model(model_name):
    file_name = f"models/{model_name}.joblib"
    return joblib.load(file_name)
def plot_regression_confusion_matrix(y_test, y_pred):
    bins = np.linspace(np.min(y_test), np.max(y_test), 6) # 将值域分成5个区间
    bin_labels = [f'{bins[i]:.1f}-{bins[i + 1]:.1f}' for i in range(len(bins) - 1)] # 区间标签

    # 将真实值和预测值映射到区间
    true_bins = np.digitize(y_test, bins) - 1 # 真实值的区间索引
    pred_bins = np.digitize(y_pred, bins) - 1 # 预测值的区间索引

    # 构造混淆矩阵
    conf_matrix = np.zeros((len(bin_labels), len(bin_labels)), dtype=int)
    for t, p in zip(true_bins, pred_bins):
        if 0 <= t < len(bin_labels) and 0 <= p < len(bin_labels): # 确保索引在合法范围内
            conf_matrix[t, p] += 1

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 8))
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title('Regression Confusion Matrix')
    plt.xlabel('Predicted Interval')
    plt.ylabel('True Interval')
    plt.xticks(range(len(bin_labels)), bin_labels, rotation=45)
    plt.yticks(range(len(bin_labels)), bin_labels)

    # 在矩阵中标注数字
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, str(conf_matrix[i, j]),
            ha='center', va='center',
            color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')
    plt.tight_layout()
    plt.show()


# 存储结果的列表
results = []

# 存储单个模型、两两组合和三三组合的结果
combination_results = []

# 网格搜索并训练单个模型
print("\n单个模型的训练和测试结果：")
for name, model in models.items():
    # 使用 GridSearchCV 寻找最优参数
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=kf, scoring='r2', n_jobs=1)
    grid_search.fit(X_train_scaled, Y_train)

    # 最优模型
    best_model = grid_search.best_estimator_
    print(f"最优参数 {name}: {grid_search.best_params_}")

    # 保存模型
    save_model(best_model, name)

    # 预测
    y_pred_train = best_model.predict(X_train_scaled)
    y_pred_test = best_model.predict(X_test_scaled)

    # 计算性能指标
    r2_train = r2_score(Y_train, y_pred_train)
    r2_test = r2_score(Y1_test, y_pred_test)
    # plot_regression_confusion_matrix(Y_train, y_pred_train)
    # plot_regression_confusion_matrix(Y1_test, y_pred_test)
    mae_test = mean_absolute_error(Y1_test, y_pred_test)
    mae_train = mean_absolute_error(Y_train, y_pred_train)
    print(f"单个模型 {name} - Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}, Test MAE: {mae_test:.4f}")

    # 存储结果
    results.append({
        'type': '单个模型',
        'model_combination': (name,),
        'r2_train': r2_train,
        'r2_test': r2_test,
        'mae_test': mae_test,
        'mae_train': mae_train,
        'y_pred_test': y_pred_test,
        'y_pred_train': y_pred_train
    })

# 保存所有结果到 DataFrame
results_df = pd.DataFrame(results)
folder_path = 'results'
file_path_csv = os.path.join(folder_path, 'model_results_sample.csv')
# 保存 DataFrame 到 CSV 文件
#results_df.to_csv(file_path_csv, index=False)

# 按照 R² 排序并选择每类模型的前五个
top_single_models = results_df[results_df['type'] == '单个模型'].sort_values(by='r2_test', ascending=False)



# 绘制性能最好的模型，包括训练集和测试集
def plot_top_models(models_df, title):
    for idx, result in models_df.iterrows():
        plt.figure(figsize=(10, 8))

        # 绘制训练集
        plt.scatter(Y_train, result['y_pred_train'], s=200, alpha=0.7, c='#4A90E2', label="Train Set", marker='o',
                    edgecolors='white')

        # 绘制测试集
        plt.scatter(Y1_test, result['y_pred_test'], s=200, alpha=0.7, c='#F55587', label="Test Set",marker='o',edgecolors='white')


        # 绘制理想的拟合线
        plt.plot([min(min(Y1_test), min(Y_train)), max(max(Y1_test), max(Y_train))],
                 [min(min(Y1_test), min(Y_train)), max(max(Y1_test), max(Y_train))],
                 'r--', lw=2, label="Perfect Fit")

        # 设置坐标轴标签并增大字体
        plt.xlabel('Actual Rejection', fontsize=35)
        plt.ylabel('Predicted Rejection', fontsize=35)

        # 设置图的标题并增大字体
        plt.title(f'{title} ({", ".join(result["model_combination"])})', fontsize=20)

        # 显示测试集的 R² 值，右下角显示并增大字体
        textstr = f'R² (Test)  = {result["r2_test"]:.4f}'
        plt.gcf().text(0.95, 0.20, textstr, fontsize=30, ha='right', va='bottom')

        # 显示测试集的 R² 值，右下角显示并增大字体
        trainstr = f'R² (Train) = {result["r2_train"]:.4f}'
        plt.gcf().text(0.95, 0.30, trainstr, fontsize=30, ha='right', va='bottom')

        # 调整图例大小和位置，左上角，并增大字体
        # plt.legend(fontsize=30, loc='upper left', bbox_to_anchor=(0, 1), frameon=False)
        # 调整x轴和y轴的范围

        # 设置坐标轴刻度字体大小
        plt.xticks(fontsize=25)  # 调整x轴刻度的字体大小
        plt.yticks(fontsize=25)  # 调整y轴刻度的字体大小
        # 设置网格线
        plt.grid(False)

        # 显示绘图
        plt.show()

# 绘制每类模型排名前五的图表
plot_top_models(top_single_models, "model")
