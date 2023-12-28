from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier , LGBMRegressor
from matplotlib import pyplot as plt


def nb_news():
    # (1)获取数据
    data = pd.read_csv("D:\\a_大二课程\\人工智能\\5G用户预测\\train.csv")
    target = data['target']
    data = data.drop(['id', 'target'], axis=1)
    # 找到num开头的列
    low_cardinality_columns = [col for col in data.columns if col.startswith('num_')]

    # 计算这些列的平均值，创建一个新的特征
    data['low_cardinality_mean'] = data[low_cardinality_columns].mean(axis=1)

    # 选择连续变量和新的特征
    data_continuous = data[low_cardinality_columns + ['low_cardinality_mean']]

    # 连接其他数值型特征
    other_numerical_columns = [col for col in data.columns if col.startswith('cat_') ]
    data_other_numerical = data[other_numerical_columns]

    # 合并数据
    data_processed = pd.concat([data_continuous, data_other_numerical], axis=1)
    # 初始化随机森林分类器
    rf_classifier = LGBMRegressor()
    # 拟合模型
    rf_classifier.fit(data_processed, target)
    # 获取特征重要性
    feature_importances = rf_classifier.feature_importances_
    feature_names = data_processed.columns

    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, feature_importances)
    plt.title('Feature Importance')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    # (2)划分数据集
    x_train, x_test, y_train, y_test = train_test_split(data_processed, target, test_size=0.3)

    
    # (4)LGBMRegressor
    estimator = LGBMRegressor()
    estimator.fit(x_train, y_train)

    # （5）模型评估
    y_predict = estimator.predict(x_test)

    print("预测结果:\n", y_predict)
    print("预测正确率:\n", y_test == y_predict)

    roc_auc = roc_auc_score(y_test, y_predict)
    print("roc_auc:", roc_auc)


if __name__ == "__main__":
    nb_news()
