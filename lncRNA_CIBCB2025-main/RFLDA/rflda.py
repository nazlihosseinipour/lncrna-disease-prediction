import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from utils.transductive_converter import Converter
class RFLDA:
    n_jobs = -1
#    n_estimators = 1
    n_estimators = 150

    random_state = 0
    
    n_folds = 10
#    n_folds = 2

    shuffle = True

    threshold = 0.5
    def __init__(self,
                feature_importance = None,
                binary_mode = False,
                concatenate_kmers_with_svd = False,
                transductive_mode = False,
                indexes_to_mask = None,
                disease_similarities = None
                ):
        self.feature_importance = feature_importance
        self.converter = Converter(binary_mode = binary_mode, 
                                   concatenate_kmers_with_svd = concatenate_kmers_with_svd,
                                   transductive_mode = transductive_mode, 
                                   indexes_to_mask = indexes_to_mask,
                                   disease_similarities = disease_similarities)
        if binary_mode:
            self.splitter = StratifiedKFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.random_state)
        else:
            self.splitter = MultilabelStratifiedKFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.random_state)
    def fit(self,
            x,
            y):
        x,y = self.converter.process_datasets_fit(x,y)
        y = self.converter.apply_masking(y)
        if self.feature_importance is None:
            # calculate feature importance here
            self.feature_importance = self._calculate_feature_importance(x,y)
        self.indexes_features_sorted_importance = self.feature_importance.index.tolist()
        rearranged_by_importance_x = x.iloc[:,self.indexes_features_sorted_importance]

        self.best_n_features = self._find_best_n_features(rearranged_by_importance_x,
                                                        y)
        top_n_features_x = rearranged_by_importance_x.iloc[:, : self.best_n_features]

        self.rflda = RandomForestRegressor(n_estimators=self.n_estimators, max_features=int(self.best_n_features/3), n_jobs = self.n_jobs, random_state = self.random_state)
        self.rflda.fit(top_n_features_x, y)

    def predict_proba(self,
                x,
                y=None):
        x = self.converter.process_datasets_predict(x,y)
        rearranged_by_importance_x = x.iloc[:,self.indexes_features_sorted_importance]
        top_n_features_x = rearranged_by_importance_x.iloc[:, : self.best_n_features]
        predictions = self.rflda.predict(top_n_features_x)
        if self.converter.binary_mode and not self.converter.transductive_mode:
            predictions = self.converter._unflatten_label_space(pd.DataFrame(predictions),y)
        return predictions
    def _find_best_n_features(self,
                            x,
                            y):
        intervals = self._find_intervals(x)
        self.accuracy_train_results = []
        self.accuracy_test_results = []

        for cols in intervals:
            top_n_features_x = x.iloc[:, :cols ]
#            top_n_features_x = top_n_features_x.to_numpy()
            accuracies_train = []
            accuracies_test = []
            for train_index, test_index in self.splitter.split(top_n_features_x, y):
                x_train, x_test = top_n_features_x.iloc[train_index], top_n_features_x.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                rf = RandomForestRegressor(n_estimators=self.n_estimators, max_features=int(cols/3), n_jobs = self.n_jobs, random_state = self.random_state)
                rf.fit(x_train, y_train)
                y_train_pred = rf.predict(x_train)
                y_test_pred = rf.predict(x_test)

                y_train_pred[y_train_pred > self.threshold] = 1    
                y_train_pred[y_train_pred <= self.threshold] = 0

                y_test_pred[y_test_pred > self.threshold] = 1    
                y_test_pred[y_test_pred <= self.threshold] = 0

                accuracies_train.append(accuracy_score(y_train, y_train_pred))
                accuracies_test.append(accuracy_score(y_test, y_test_pred))
                
            mean_train_accuracy = np.mean(accuracies_train)
            mean_test_accuracy = np.mean(accuracies_test)

            self.accuracy_train_results.append(mean_train_accuracy)
            self.accuracy_test_results.append(mean_test_accuracy)

        self.accuracy_find_best_n = pd.DataFrame([self.accuracy_train_results, self.accuracy_test_results])
        self.accuracy_find_best_n.columns = intervals
        
        best_performance_train = self.accuracy_find_best_n.iloc[0].max()
        best_performance_n_features_train = self.accuracy_find_best_n.iloc[0].argmax()

        best_performance_test = self.accuracy_find_best_n.iloc[1].max()
        best_performance_n_features_test = self.accuracy_find_best_n.iloc[1].argmax()

        performance_test = self.accuracy_find_best_n.iloc[1].iloc[best_performance_n_features_test]

        best_n_features = self.accuracy_find_best_n.columns[best_performance_n_features_test]

        self.accuracy_find_best_n.insert(0, "subset", ["train_accuracy", "test_accuracy"])

        
        return best_n_features
    def _find_intervals(self, 
                        x):
        return range(50, x.shape[1] + 1, 50)

    def _calculate_feature_importance(self,
                                     x,
                                     y):
        rf = RandomForestRegressor(n_estimators=self.n_estimators, max_features=int(x.shape[1]/3), n_jobs = self.n_jobs, random_state = self.random_state)
        rf.fit(x,y)
        return pd.Series(rf.feature_importances_, index = range(x.shape[1]))   
    def get_output_name(self):
        return self.converter.get_output_name()
