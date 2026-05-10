from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from utils.transductive_converter import Converter
import pandas as pd
class IPCARF:
    n_jobs = -1
    n_estimators = 10
#    n_estimators = 150
    random_state = 0
    n_folds = 2
#    n_folds = 10

    n_components_optimize = [2]
#    n_components_optimize = [2, 4, 8, 16, 32, 64, 128]

    def __init__(self,
                n_components = None,
                binary_mode = False,
                concatenate_kmers_with_svd = False,
                transductive_mode = False,
                indexes_to_mask = None,
                disease_similarities = None
                ):
        self.n_components = n_components
        self.ipcarf = None
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
#        pd.DataFrame(x).to_csv("binary_x.csv")
#        pd.DataFrame(y).to_csv("binary_y.csv")

        if self.n_components is None:
            self.n_components = self._optimize_n_components(x,
                                                            y)
        pca = IncrementalPCA(n_components = self.n_components)
        pca.fit(x,
                y)
        rf = RandomForestRegressor(n_estimators=self.n_estimators, n_jobs = self.n_jobs, random_state = self.random_state)
        pipe = Pipeline(steps=[("pca", pca), ("rf", rf)])
        pipe.fit(x,
                y)
        self.ipcarf = pipe
    def predict_proba(self,
                x,
                y=None):
                
        x = self.converter.process_datasets_predict(x,y)

        predictions = self.ipcarf.predict(x) 
        
        if self.converter.binary_mode and not self.converter.transductive_mode:
            predictions = self.converter._unflatten_label_space(pd.DataFrame(predictions),y)
        return predictions
    def get_output_name(self):
        return self.converter.get_output_name()
    def _optimize_n_components(self,
                                x,
                                y):
        pca = IncrementalPCA()
        rf = RandomForestRegressor(n_estimators=self.n_estimators, n_jobs = self.n_jobs, random_state = self.random_state)
        pipe = Pipeline(steps=[("pca", pca), ("rf", rf)])
        param_grid = {
        "pca__n_components": self.n_components_optimize,
        }
        opt = GridSearchCV(
            pipe,
            param_grid,
            )
        opt.fit(x,
                y)
        return opt.best_params_["pca__n_components"]

