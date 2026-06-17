import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from utils.transductive_converter import Converter


class RF:
    """Plain multilabel Random Forest baseline.

    Mirrors the RFLDA / IPCARF interface (``fit`` / ``predict_proba`` /
    ``get_output_name``) so it flows through the identical inductive Evaluator
    (Youden-index thresholding, micro metrics). Unlike RFLDA it performs no
    feature-importance ranking or nested feature-count search: it is a single
    ``RandomForestRegressor`` fit on the full multilabel label matrix, which is
    exactly RFLDA minus the feature-selection step.
    """

    n_jobs = -1
    n_estimators = 150
    random_state = 0

    def __init__(self,
                feature_importance=None,
                binary_mode=False,
                concatenate_kmers_with_svd=False,
                transductive_mode=False,
                indexes_to_mask=None,
                disease_similarities=None):
        # feature_importance kept for signature parity with RFLDA; unused here.
        self.feature_importance = feature_importance
        self.converter = Converter(binary_mode=binary_mode,
                                   concatenate_kmers_with_svd=concatenate_kmers_with_svd,
                                   transductive_mode=transductive_mode,
                                   indexes_to_mask=indexes_to_mask,
                                   disease_similarities=disease_similarities)

    def fit(self,
            x,
            y):
        x, y = self.converter.process_datasets_fit(x, y)
        y = self.converter.apply_masking(y)
        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.rf.fit(x, y)

    def predict_proba(self,
                x,
                y=None):
        x = self.converter.process_datasets_predict(x, y)
        predictions = self.rf.predict(x)
        if self.converter.binary_mode and not self.converter.transductive_mode:
            predictions = self.converter._unflatten_label_space(pd.DataFrame(predictions), y)
        return predictions

    def get_output_name(self):
        return self.converter.get_output_name()
