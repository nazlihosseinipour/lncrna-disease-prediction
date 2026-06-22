import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from utils.transductive_converter import Converter
import pandas as pd
class IPCARF:
    n_jobs = -1
    n_estimators = 150
    random_state = 0
    n_folds = 5

    n_components_optimize = [2, 4, 8, 16, 32, 64, 128]

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
        x = self._sanitize_x(x)
        y = self.converter.apply_masking(y)
#        pd.DataFrame(x).to_csv("binary_x.csv")
#        pd.DataFrame(y).to_csv("binary_y.csv")

        if self.n_components is None:
            self.n_components = self._optimize_n_components(x,
                                                            y)
        rf = RandomForestRegressor(n_estimators=self.n_estimators, n_jobs = self.n_jobs, random_state = self.random_state)
        pipe = self._fit_pipeline_with_pca_fallback(x, y, rf)
        self.ipcarf = pipe
    def predict_proba(self,
                x,
                y=None):
                
        x = self.converter.process_datasets_predict(x,y)
        x = self._sanitize_x(x)

        predictions = self.ipcarf.predict(x) 
        
        if self.converter.binary_mode and not self.converter.transductive_mode:
            predictions = self.converter._unflatten_label_space(pd.DataFrame(predictions),y)
        return predictions
    def get_output_name(self):
        return self.converter.get_output_name()
    def _optimize_n_components(self,
                                x,
                                y):
        component_grid = self._valid_component_grid(x)
        rf = RandomForestRegressor(n_estimators=self.n_estimators, n_jobs = self.n_jobs, random_state = self.random_state)
        pipe = Pipeline(steps=[("pca", IncrementalPCA()), ("rf", rf)])
        param_grid = {
        "pca__n_components": component_grid,
        }
        try:
            opt = GridSearchCV(
                pipe,
                param_grid,
                cv=self.n_folds,
                n_jobs=self.n_jobs,
                error_score=np.nan,
                )
            opt.fit(x,
                    y)
            return opt.best_params_["pca__n_components"]
        except Exception:
            # IncrementalPCA can fail with "SVD did not converge" on the expanded
            # binary design matrix. Randomized PCA is more robust and preserves the
            # IPCA->RF structure closely enough for the server sweep to continue.
            pipe = Pipeline(steps=[
                ("pca", PCA(svd_solver="randomized", random_state=self.random_state)),
                ("rf", rf),
            ])
            opt = GridSearchCV(
                pipe,
                param_grid,
                cv=self.n_folds,
                n_jobs=self.n_jobs,
                error_score=np.nan,
            )
            opt.fit(x, y)
            return opt.best_params_["pca__n_components"]

    def _fit_pipeline_with_pca_fallback(self, x, y, rf):
        try:
            pipe = Pipeline(steps=[("pca", IncrementalPCA(n_components=self.n_components)), ("rf", rf)])
            pipe.fit(x, y)
            return pipe
        except Exception:
            pipe = Pipeline(steps=[
                ("pca", PCA(
                    n_components=self.n_components,
                    svd_solver="randomized",
                    random_state=self.random_state,
                )),
                ("rf", rf),
            ])
            pipe.fit(x, y)
            return pipe

    def _valid_component_grid(self, x):
        limit = max(1, min(x.shape[0], x.shape[1]))
        grid = [n for n in self.n_components_optimize if n <= limit]
        return grid or [limit]

    def _sanitize_x(self, x):
        x = pd.DataFrame(x).apply(pd.to_numeric, errors="coerce")
        return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
