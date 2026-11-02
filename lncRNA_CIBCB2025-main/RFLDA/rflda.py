import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import accuracy_score, average_precision_score
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
                 disease_similarities = None,
                 n_folds = None,
                 selection_metric = "micro_auprc",
                 ):
        self.feature_importance = feature_importance
        self.n_folds = self.__class__.n_folds if n_folds is None else n_folds
        if self.n_folds < 2:
            raise ValueError("RFLDA inner CV requires at least two folds.")
        if selection_metric not in {"micro_auprc", "subset_accuracy"}:
            raise ValueError(
                "selection_metric must be 'micro_auprc' or 'subset_accuracy'."
            )
        self.selection_metric = selection_metric
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

        self.rflda = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self._rf_max_features(self.best_n_features),
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
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
            scores_train = []
            scores_test = []
            for train_index, test_index in self.splitter.split(top_n_features_x, y):
                x_train, x_test = top_n_features_x.iloc[train_index], top_n_features_x.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_features=self._rf_max_features(cols),
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                )
                rf.fit(x_train, y_train)
                y_train_pred = rf.predict(x_train)
                y_test_pred = rf.predict(x_test)

                scores_train.append(self._selection_score(y_train, y_train_pred))
                scores_test.append(self._selection_score(y_test, y_test_pred))
                
            mean_train_accuracy = np.mean(scores_train)
            mean_test_accuracy = np.mean(scores_test)

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

        metric_label = (
            "micro_auprc"
            if self.selection_metric == "micro_auprc"
            else "subset_accuracy"
        )
        self.accuracy_find_best_n.insert(
            0, "subset", [f"train_{metric_label}", f"validation_{metric_label}"]
        )

        
        return best_n_features

    def _selection_score(self, y_true, y_score):
        if self.selection_metric == "micro_auprc":
            return average_precision_score(y_true, y_score, average="micro")
        y_binary = np.asarray(y_score) > self.threshold
        return accuracy_score(y_true, y_binary)
    def _find_intervals(self, 
                        x):
        if x.shape[1] < 50:
            return [x.shape[1]]
        intervals = list(range(50, x.shape[1] + 1, 50))
        if intervals[-1] != x.shape[1]:
            intervals.append(x.shape[1])
        return intervals

    def _rf_max_features(self,
                         n_features):
        return max(1, int(n_features/3))

    def _calculate_feature_importance(self,
                                     x,
                                     y):
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self._rf_max_features(x.shape[1]),
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        rf.fit(x,y)
        return pd.Series(rf.feature_importances_, index = range(x.shape[1]))   
    def get_output_name(self):
        return self.converter.get_output_name()
