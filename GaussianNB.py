from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE


def nb_news():
    # (1)Get the data
    data = pd.read_csv("train.csv")
    target = data['target']
    data = data.drop('target', axis=1)

    #Find columns where the number of unique values is less than 100
    low_cardinality_columns = [col for col in data.columns if data[col].nunique() < 100]
    data['low_cardinality_mean'] = data[low_cardinality_columns].mean(axis=1)

    # Select continuous variables and new features
    data_continuous = data[low_cardinality_columns + ['low_cardinality_mean']]
    other_numerical_columns = [col for col in data.columns if col.startswith('num_') or col.startswith('cat_')]
    data_other_numerical = data[other_numerical_columns]

    #  Data transformation
    transformer = PowerTransformer()
    data_other_numerical_transformed = transformer.fit_transform(data_other_numerical)

    # Combine the data
    data_processed = pd.concat(
        [data_continuous, pd.DataFrame(data_other_numerical_transformed, columns=other_numerical_columns)], axis=1)

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(data_processed, target, test_size=0.1, random_state=42)


    # Gaussian Naive Bayes algorithm
    estimator = GaussianNB()
    estimator.fit(x_train, y_train)

    # Model evaluation
    y_predict = estimator.predict(x_test)
    roc_auc = roc_auc_score(y_test, y_predict)

    print("Predictions:\n", y_predict)
    print("roc_auc:", roc_auc)


if __name__ == "__main__":
    nb_news()
