import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from Preprocess import preprocess, zip_scaling

from SkLearnHelper import fit_classifier_default

df1 = pd.read_csv('train.csv')
# df2 = pd.read_csv('test.csv')

X = df1

y = X.pop('label')
col_names = X.columns


def processClassify(X_train, X_test, y_train, y_test, pca_contribution=[0.20, 0.5, .90]):
    processClassifyDict = {}
    X_dict = preprocess(X_train, X_test, zip_scaling, pca=0.85)

    for k, v in X_dict.items():
        print(k)
        print(pd.DataFrame(v[0]).shape)
        print(pd.DataFrame(v[1]).shape)

        processClassifyDict[k] = fit_classifier_default(v[0], v[1], y_train, y_test)
    return processClassifyDict


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

models = processClassify(X_train, X_test, y_train, y_test)


