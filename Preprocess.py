import copy

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelBinarizer, FunctionTransformer, \
    RobustScaler
import numpy as np
import pandas as pd

scalings = [
    # (x-mean)/std transforms data to represent a normal
    # distribution with mean=0, std=1. 70% values between [-1,1]
    StandardScaler(),

    # This (xi-min(x)/(max(x)-min(x) this preserves
    # distribution while scaling to a range [0,1]
    # Doesnt reduct importance of outliers
    MinMaxScaler(),

    # transforms the feature vector by subtracting the median and
    # then dividing by the interquartile range (75% value — 25% value)
    # outliers have less influence but range is larger than standardization
    RobustScaler()
]
scaling_names = [
    "Standard Scaler",
    "MinMax Scaler",
    "Robust Scaler"
]

zip_scaling = zip(scaling_names, scalings)



def preprocess(X_train, X_test, pca=0, num_imputer='median', cat_imputer='missing',special_indices1=None,
               special_indices2=None):
    '''
    :param X_train: dtype = pd.DataFrame Dataset which is being used this does not include the label being classified

    :param pca: if you want to apply pca to numerical columns then set pca to a number in the range of [0,1]
                the number will represent the the prinicple components needed to  hit the user
                specified explained variance contribution

    :param special_indices1: user defined indexes of columns in df they want to apply a specific pipeline to
                            special_indecies1= (column_list , pipe)

    :param special_indices2: special_indecies2=(column_list , pipe)

    :return: A dictionary is returned of d[scaling_name] = (X_processed , pipe)
    '''
    X_processed_dict = {}
    X_train, X_test = pd.DataFrame(X_train), pd.DataFrame(X_test)
    classGrid=copy.deepcopy(zip_scaling)
    # Find the index of the columns in a df with numerical attributes

    num_indices = [key for key in dict(X_train.dtypes)
                   if dict(X_train.dtypes)[key]
                   in ['float64', 'float32', 'int32', 'int64']]  # Numeric Variable

    # Find the index of the columns in a df with Catagorical attributes
    cat_indices = [key for key in dict(X_train.dtypes)
                   if dict(X_train.dtypes)[key] in ['object']]  # Categorical Varibles

    for n, c in classGrid:
        # create a pipe to scale/transform numerical data and catagorical data in different ways
        # to create various datasets from one dataset

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=num_imputer)),
            ('n', c)
        ])

        # add pca to numerical transformation
        if pca != 0:
            print(f'performing pca for {n}')
            numeric_transformer.steps.append(['PCA', PCA(n_components=pca, svd_solver='full')])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=cat_imputer)),
            ('Label Binarizer', LabelBinarizer())
        ])

        # check to see if the user has defined any of the 2 special columns that need a particular pipe
        if bool(special_indices1):
            # Remove the special_indices from both the numerical indices and catagorical indices
            num_indices = [x for x in num_indices if not x in special_indices1 or special_indices1.remove(x) or
                           special_indices2 or special_indices2.remove(x)]
            cat_indices = [x for x in cat_indices if not x in special_indices1 or special_indices1.remove(x) or
                           special_indices2 or special_indices2.remove(x)]

            if bool(special_indices2):
                preprocess_model = ColumnTransformer(
                    transformers=[
                        ('numerical', numeric_transformer, num_indices),
                        ('catagorical', categorical_transformer, cat_indices),
                        ('special 1', special_indices1[1], special_indices1[0]),
                        ('special 2', special_indices2[1], special_indices2[0])
                    ])
            else:

                # create the column transformer with the pipeline steps created above to be performed
                # on  either catagorical, numerical, and special column indexes with appropriate pipeline
                preprocess_model = ColumnTransformer(
                    transformers=[
                        ('numerical', numeric_transformer, num_indices),
                        ('catagorical', categorical_transformer, cat_indices),
                        ('special 1', special_indices1[1], special_indices1[0])
                    ])

            # fit transfrom
            X_processed = preprocess_model.fit_transform(X_train)
            X_test_processed = preprocess_model.transform(X_test)
            X_processed_dict[n] = [X_processed, X_test_processed, preprocess_model]

        else:
            # create the column transformer with the pipeline steps created above to be performed
            # on  either catagorical, numerical column indexes with appropriate pipeline. No special column transforms
            preprocess_model = ColumnTransformer(
                transformers=[
                    ('numerical', numeric_transformer, num_indices),
                    ('catagorical', categorical_transformer, cat_indices)
                ])

            X_processed = preprocess_model.fit_transform(X_train)
            X_test_processed = preprocess_model.transform(X_test)
            X_processed_dict[n] = [X_processed, X_test_processed, preprocess_model]
            print(f"new df was created using the {n} and has a shape")
            print(pd.DataFrame(X_processed).shape)

            # print(f'Processed X with shape = {pd.DataFrame(X_processed).shape}  and dtype = {type(X_processed)}')
            # print(pd.DataFrame(X_processed).shape)

    return X_processed_dict
