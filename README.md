# 5G User Prediction Using Machine Learning
Overview
This project applies machine learning techniques to predict 5G users based on various user characteristics. It utilizes Python and machine learning libraries to analyze and predict 5G user status from a dataset containing features such as user charges, data usage, active behavior, package types, and regional information.
Objectives
1.To explore and solve real-life problems using AI solutions in Python.
2.To familiarize with handling datasets, employing AI Python libraries, and understanding data characteristics in practical problems.
3.To compare and analyze different models for accurate 5G user prediction.
Data
The dataset includes 60 features, comprising both categorical (cat_0 to cat_19) and numerical (num_0 to num_37) features. The target variable indicates whether a user is a 5G user.
Methodology
（1）Data Preprocessing: Identified low cardinality features and created new features by calculating their mean. Used PowerTransformer for transforming numerical features to approximate a Gaussian distribution.
（2）Model Selection and Analysis: Explored various algorithms like Logistic Regression, Decision Trees, Random Forests, and ultimately selected LightGBM and Gaussian Naive Bayes for their effectiveness.
（3）Feature Importance: Analyzed feature importance using LightGBM to gain insights into the most predictive features.
Models
1.	LightGBM Regressor: An efficient and scalable implementation of gradient boosting framework, particularly effective for large datasets and numerous features.
2.	Gaussian Naive Bayes: A probabilistic classifier that assumes independence among features, effective for its simplicity and efficiency.
Results
1.LightGBM Model achieved an AUC score of 0.91, indicating high accuracy in predicting 5G users.
2.Gaussian Naive Bayes Model achieved an AUC score of 0.72, showing reasonable predictive capability.
Usage
Details on how to use this repository for 5G user prediction are provided in the respective script files.
Contributing
Feel free to fork this repository, and contributions are welcome. Please open an issue first to discuss what you would like to change.
