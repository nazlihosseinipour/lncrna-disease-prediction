from utils.transductive_converter import Converter
import Code.test2_CNN as test2_CNN
from Code.multi_adaboost_CNN import AdaBoostClassifier as Ada_CNN
import lightgbm as lgb
import numpy as np
import pandas as pd
class LDA_SABC:
    lgbn_estimators = 150
#    lgbn_estimators = 1

    CNNn_estimators = 150
#    CNNn_estimators = 1

    lgblearning_rate = 0.1
    CNNlearning_rate = 0.1
    CNNepochs = 10
 #   CNNepochs = 1

    batch_size = 10

    def __init__(self,
                alpha = 0.4,
                binary_mode = False,
                concatenate_kmers_with_svd = False,
                transductive_mode = False,
                indexes_to_mask = None,
                disease_similarities = None

                ):
        self.alpha = alpha
        self.converter = Converter(binary_mode = binary_mode, 
                                   concatenate_kmers_with_svd = concatenate_kmers_with_svd,
                                   transductive_mode = transductive_mode, 
                                   indexes_to_mask = indexes_to_mask,
                                   disease_similarities = disease_similarities)
       
    def fit(self,
            x,
            y):
        x, y = self.converter.process_datasets_fit(x,y)
        y = self.converter.apply_masking(y)

        self._fit_Ada_CNN(x,y)
        self._fit_lightgbm(x,y)
    def _fit_Ada_CNN(self, 
                        x,
                        y):
        x = self._reshape_for_CNN(x)
        self.ada_cnn = Ada_CNN(base_estimator=test2_CNN.baseline_model(n_features=x.shape[1]),
        n_estimators=self.CNNn_estimators,
        learning_rate=self.CNNlearning_rate, epochs=self.CNNepochs)

        self.ada_cnn.fit(x, y.values.flatten(), batch_size = self.batch_size)
    def _reshape_for_CNN(self,
                        x):
        x = np.array([x])     
        x = x.reshape((x.shape[1], x.shape[2], x.shape[0]))
        return x
    def _fit_lightgbm(self,
                       x,
                       y):
        self.lightgbm = lgb.LGBMClassifier(learning_rate=self.lgblearning_rate, n_estimators=self.lgbn_estimators)
        self.lightgbm.fit(x, y)
        
    def predict_proba(self,
                x,
                y=None):            
        x = self.converter.process_datasets_predict(x,y)

        predictions_ada_cnn = self.ada_cnn.predict_proba(self._reshape_for_CNN(x))[:,1:2]
        predictions_lightgbm = self.lightgbm.predict_proba(x)[:,1:2]
        
        predictions = (self.alpha * predictions_ada_cnn + (1-self.alpha) * predictions_lightgbm)/2
        print(predictions.shape)
        if self.converter.binary_mode and not self.converter.transductive_mode:
            predictions = self.converter._unflatten_label_space(pd.DataFrame(predictions),y)
        return predictions
    def get_output_name(self):
        return self.converter.get_output_name()