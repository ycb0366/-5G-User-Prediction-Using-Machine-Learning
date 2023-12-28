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
    #Get the data
    data = pd.read_csv("train.csv")
    target = data['target']
    data = data.drop(['id', 'target'], axis=1)
    # Find columns starting with 'num_'
    low_cardinality_columns = [col for col in data.columns if col.startswith('num_')]

    #Calculate the mean of these columns to create a new feature
    data['low_cardinality_mean'] = data[low_cardinality_columns].mean(axis=1)

    #Select continuous variables and the new feature
    data_continuous = data[low_cardinality_columns + ['low_cardinality_mean']]

    # Combine other numerical features
    other_numerical_columns = [col for col in data.columns if col.startswith('cat_') ]
    data_other_numerical = data[other_numerical_columns]

    # Merge the data
    data_processed = pd.concat([data_continuous, data_other_numerical], axis=1)
    # Initialize the LGBMRegressor
    rf_classifier = LGBMRegressor()
    # Fit the model
    rf_classifier.fit(data_processed, target)
    # Get feature importances
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
    #Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(data_processed, target, test_size=0.3)

    
    # (4)LGBMRegressor
    estimator = LGBMRegressor()
    estimator.fit(x_train, y_train)

    #Model evaluation
    y_predict = estimator.predict(x_test)

    print("Predictions:\n", y_predict)
    print("Accuracy:\n", y_test == y_predict)

    roc_auc = roc_auc_score(y_test, y_predict)
    print("roc_auc:", roc_auc)


if __name__ == "__main__":
    nb_news()
