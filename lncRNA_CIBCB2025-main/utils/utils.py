
import numpy as np
import pandas as pd
def filter_labels(y,
                minimum_occurrences = 1):
    total = y.sum(axis=0)
    columns_to_keep = total > minimum_occurrences
    y = y.loc[:, columns_to_keep]
    return y

def iterator_cross_validation(indexes,
                            x,
                            y,
                            n_folds = 10):
    train = []
    test = []
    for fold in range(1, n_folds + 1):
        train_column = "train_fold" + str(fold)
        test_column = "test_fold" + str(fold)

        train_index = indexes[train_column]
        train_index = train_index.loc[~train_index.isna()]
        test_index = indexes[test_column]
        test_index = test_index.loc[~test_index.isna()]

        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        train.append((x_train, y_train))
        test.append((x_test, y_test))        
    return (train,test)
def iterator_cross_validation_evaluation(indexes,
                            y,
                            n_folds = 10):
    train = []
    test = []
    for fold in range(1, n_folds + 1):
        train_column = "train_fold" + str(fold)
        test_column = "test_fold" + str(fold)

        train_index = indexes[train_column]
        train_index = train_index.loc[~train_index.isna()]
        test_index = indexes[test_column]
        test_index = test_index.loc[~test_index.isna()]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        train.append(y_train)
        test.append(y_test)
                
    return (train,test)
def filter_label_space(train_y,
                       test_y,
                       drop_labels_threshold = 30):
    if drop_labels_threshold > 0:
        train_y = train_y.loc[:, train_y.sum(axis=0) > drop_labels_threshold]
        test_y = test_y[train_y.columns]
    return train_y, test_y    

    