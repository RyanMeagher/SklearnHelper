import copy

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from paramGrid import createParamGrid
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

classifier_names = [
    "RandForest",
    "KNN",
    "SVM",
    "NB",
    #"XgBoost",
    "Logistic Regression"
]

classifiers = [
    RandomForestClassifier(n_jobs=-1),
    KNeighborsClassifier(n_jobs=-1),
    SVC(),
    GaussianNB(),
    #GradientBoostingClassifier(),
    LogisticRegression(max_iter=100, n_jobs=-1)
]
zipped_clf = zip(classifier_names, classifiers)


def fit_classifier_default(X_train, X_test, y_train, y_test):
    modelDict = {}
    classGrid = copy.deepcopy(zipped_clf)

    for n, c in classGrid:
        pipe = Pipeline([
            (f'{n}', c)
        ])

        print(f"fitting x_train with {n} ")
        classification_model = pipe.fit(X_train, y_train)
        y_pred = classification_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        modelDict[n] = (classification_model, y_pred, accuracy)
        print(f"{n}  accuracy score on x_test: {accuracy * 100:.2f}%")

    return modelDict


def fit_classifier_gridSearch(X_train, X_test, y_train, y_test, num_cross_val=2):
    modelDict = {}
    classGrid = copy.deepcopy(zipped_clf)

    paramGrid = createParamGrid(X_train)

    for n, c in classGrid:
        print(f'Initializaing grid search to find best parameters of {n} classifier')

        pipe = Pipeline([
            (f'{n}', c)
        ])

        classification_model = GridSearchCV(pipe, paramGrid[f'{n}'], cv=num_cross_val, n_jobs=-1)
        print(f'The best parameters for {n} classifier are are being searched for via GridSearchCV')

        model_fit = classification_model.fit(X_train, y_train)
        print(classification_model)
        y_pred = model_fit.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("accuracy score: {0:.2f}%".format(accuracy * 100))

        # add the best parameters found via GridSearchCV and also the
        modelDict[n] = (classification_model, y_pred, accuracy)

    return modelDict
