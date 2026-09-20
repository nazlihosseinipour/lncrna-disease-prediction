from sklearn.metrics import hamming_loss
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd
from utils.transductive_converter import Converter

class Evaluator:
    evaluation_measures_transductive = ["precision", "recall", "fscore", "accuracy",  "auprc"]
    evaluation_measures_inductive = ["hamming", "label_ranking", "micro_roc", "micro_auprc", "precision", "recall", "fscore", "accuracy"]

    columns_to_add_df = [
                        "fold1",
                        "fold2",
                        "fold3",
                        "fold4",
                        "fold5",
                        "fold6",
                        "fold7",
                        "fold8",
                        "fold9",
                        "fold10",
                        "mean(std)"
                        
    ]

    def __init__(self,
                binary_mode = False,
                concatenate_kmers_with_svd = False,
                transductive_mode = False,
                indexes_to_mask = None,
                threshold = 0.5):
#        self.threshold = threshold
#        self.threshold = 0.1

        self.converter = Converter(binary_mode = binary_mode, 
                                   concatenate_kmers_with_svd = concatenate_kmers_with_svd,
                                   transductive_mode = transductive_mode, 
                                   indexes_to_mask = indexes_to_mask)
    def evaluate(self, 
                    y_true,
                    y_pred):
        if self.converter.transductive_mode:
            y_true = pd.DataFrame(self.converter._flatten_label_space(y_true))

            if not self.converter.binary_mode:
                ### change multi label output to binary
                y_pred = pd.DataFrame(self.converter._flatten_label_space(y_pred))
            return self._evaluate_transductive(y_true,
                                                y_pred)
        else:
            return self._evaluate_inductive(y_true,
                                                y_pred)
    def evaluate_flat_mesh(self,
                            y_true,
                            y_pred_flat,
                            y_pred_mesh):
        y_pred_mesh_filtered = y_pred_mesh[y_pred_flat.columns]
#        print(y_pred_mesh.columns)
#        print(y_pred_flat.columns)
        return self._evaluate_inductive(y_true,
                                        y_pred_mesh_filtered)
    def _evaluate_inductive(self,
                            y_true,
                            y_pred):
#        print(y_true.shape)
#        print(y_pred.shape)
        thresholds = self.sensivity_specifity_cutoff(y_true, y_pred)
        y_pred_thresholded = self._apply_threshold(y_pred, thresholds)                   
        hamming = hamming_loss(y_true, y_pred_thresholded)
        label_ranking = label_ranking_loss(y_true, y_pred_thresholded)
        roc = roc_auc_score(y_true, y_pred, average = "micro")
        average_precision = average_precision_score(y_true, y_pred, average="micro")
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred_thresholded, average = "micro")
        accuracy = accuracy_score(y_true, y_pred_thresholded)

        return hamming, label_ranking, roc, average_precision, precision, recall, fscore, accuracy
    def _evaluate_transductive(self,
                            y_true,
                            y_pred):
        
        y_pred = self._select_indexes_to_evaluate(y_pred)
        y_true = self._select_indexes_to_evaluate(y_true)
        
        y_pred_thresholded = self._apply_threshold(y_pred)
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred_thresholded, average = "binary")
        accuracy = accuracy_score(y_true, y_pred_thresholded)
        auprc = average_precision_score(y_true, y_pred)
        return precision, recall, fscore, accuracy, auprc
#        return precision, recall, fscore, accuracy, auprc, auroc

    def _select_indexes_to_evaluate(self,
                                    y):
        return y.iloc[self.converter.indexes_to_mask.values.flatten()]                      
    def _apply_threshold(self,
                    y,
                    thresholds):
        y = y.copy()
        # y[y < self.threshold] = 0  
        # y[y >= self.threshold] = 1
        for i, c in enumerate(y):
            y.loc[y[c] < thresholds[i], c] = 0 
            y.loc[y[c] >= thresholds[i], c] = 1

        return y
    def create_df_transductive(self,
                list_performance):
        list_performance.append([str(a) + "(" + str(b) + ")" for a,b in zip(np.mean(list_performance, axis=0), np.std(list_performance, axis=0))])
        df = pd.DataFrame(list_performance , columns = self.evaluation_measures_transductive)
        df.insert(0, "folds", self.columns_to_add_df)
        return df
    def create_df_inductive(self,
                list_performance):
        list_performance.append([str(a) + "(" + str(b) + ")" for a,b in zip(np.mean(list_performance, axis=0), np.std(list_performance, axis=0))])
        df = pd.DataFrame(list_performance , columns = self.evaluation_measures_inductive)
        df.insert(0, "folds", self.columns_to_add_df)
        return df

    def sensivity_specifity_cutoff(self, y_true, y_score):
        '''Find data-driven cut-off for classification
        
        Cut-off is determied using Youden's index defined as sensitivity + specificity - 1.
        
        Parameters
        ----------
        
        y_true : array, shape = [n_samples]
            True binary labels.
            
        y_score : array, shape = [n_samples]
            Target scores, can either be probability estimates of the positive class,
            confidence values, or non-thresholded measure of decisions (as returned by
            “decision_function” on some classifiers).
            
        References
        ----------
        
        Ewald, B. (2006). Post hoc choice of cut points introduced bias to diagnostic research.
        Journal of clinical epidemiology, 59(8), 798-801.
        
        Steyerberg, E.W., Van Calster, B., & Pencina, M.J. (2011). Performance measures for
        prediction models and markers: evaluation of predictions and classifications.
        Revista Espanola de Cardiologia (English Edition), 64(9), 788-794.
        
        Jiménez-Valverde, A., & Lobo, J.M. (2007). Threshold criteria for conversion of probability
        of species presence to either–or presence–absence. Acta oecologica, 31(3), 361-369.
        '''
        threshold_per_label = []
        for column in y_true:
            fpr, tpr, thresholds = roc_curve(y_true[column], y_score[column])
            
            scores = tpr - fpr
            if scores.sum() > 0:
                idx = np.nanargmax(scores)
                threshold = thresholds[idx]
            else:
                threshold = 0.5
            threshold_per_label.append(threshold)
        return threshold_per_label