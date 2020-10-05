import numpy as np


def createParamGrid(X):
    param_grid = {}
    param_grid["Logistic Regression classifier"] = {
        'Logistic Regression classifier__C': [0.001, 0.01, .01, 1, 1.5, 2, 5, 10, 25, 50, 100, 1000],
        # 'Logistic Regression classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

    }

    param_grid["KNN classifier"] = {
        'KNN classifier__n_neighbors': np.linspace(1, 17, 7, dtype=int),
        # 'KNN classifier__algorithm': ['ball_tree', 'kd_tree'],
        # 'KNN classifier__metric': ["euclidean", "manhattan", "chebyshev", "minkowski","wminkowski", "seuclidean", "mahalanobis"]

    }
    param_grid["RandForest classifier"] = {
        "RandForest classifier__criterion": ["gini", "entropy"],
        # "RandForest classifier__max_features": [3, 5, 10, 15],
        # "RandForest classifier__n_estimators": [250, 500, 750]

    }
    param_grid["SVM classifier"] = {
        'SVM classifier__C': [0.001, 0.01, .01, 1, 1.5, 2, 5, 10, 25, 50, 100, 1000],
        # 'SVM classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        # 'SVM classifier__degree': np.arange(0, 5),
        # 'SVM classifier__gamma': ["scale", "auto"],
        # 'SVM classifier__decision_function_shape': ['ovo', 'ovr'],
        # 'SVM classifier__probability': [True, False]

    }
    param_grid["NB classifier"] = {
        'NB Regression classifier__var_smoothing': 10 ** -9

    }
    param_grid["XgBoost classifier"] = {
        'XgBoost classifier__loss': ['exponential', 'deviance'],
        # 'XgBoost classifier__learning_rate': np.linspace(0, 2, 5),
        'XgBoost classifier__criterion': ["friedman_mse", "mse", "mae"],
        # 'XgBoost classifier__max_features': ["auto", "sqrt", "log2"],

    }
    return param_grid
