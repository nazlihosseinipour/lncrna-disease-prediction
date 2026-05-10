from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
from scipy import linalg
class Converter:
    random_state = 0
    def __init__(self,
                binary_mode = False,
                transductive_mode = False,
                concatenate_kmers_with_svd = False,
                indexes_to_mask = None,
                disease_similarities = None):
        self.binary_mode = binary_mode
        self.transductive_mode = transductive_mode
        self.concatenate_kmers_with_svd = concatenate_kmers_with_svd
        self.indexes_to_mask = indexes_to_mask
        self.disease_similarities = disease_similarities
    ## 3 boolean parameters
    
    # if concatenate_kmers_with_svd then concatenate_kmers_with_svd

    # true true true binary, kmers + rna_features + disease_features
    # true false true # binary, rna_features + disease_features


    # false false true multi_label, kmers # 
    # false true true multi_label, rna_features

    ## invalid
    # true true false ## does not exist 
    # true false false # does not exist # binary inductive 

    def process_datasets_predict(self,
                                 x,
                                 y = None):
        x = self._convert_input_multilabel_to_binary(x,
                                                        y)
        return x
    def process_datasets_fit(self,
                        x,
                        y):
        
        if self.binary_mode:
            x = self._convert_input_multilabel_to_binary(x,
                                                        y)
            y = self._convert_output_multilabel_to_binary(y)            
        elif self.concatenate_kmers_with_svd:
            x = self._convert_input_multilabel_to_binary(x,
                                                            y)
            
        return x,y
    def apply_masking(self,
                      y):
        if self.indexes_to_mask is not None:
            if self.binary_mode:
                y = self._apply_masking(y)
                y = y.values.ravel()
            else:
                multilabel_y = y.copy()
                y = self._convert_output_multilabel_to_binary(y)            
                y = self._apply_masking(y)
                y = self._unflatten_label_space(y, multilabel_y)
            
        return pd.DataFrame(y)    
    def _convert_output_multilabel_to_binary(self,
                                multilabel_y,
                                ): 
        binary_y = self._flatten_label_space(multilabel_y)   
        return pd.DataFrame(binary_y)

    def _convert_input_multilabel_to_binary(self,
                                multilabel_x,
                                multilabel_y,
                                feature_type_to_generate = "svd"): 
        # if self.binary_mode
        # if self.transductive_mode 

        if self.binary_mode:
            if self.transductive_mode:
                if feature_type_to_generate == "svd":
                    #self.fit_svd(multilabel_y)
                    new_rna_features, disease_features = self._transform_svd(multilabel_y)
                    # run SVD and return the other side
                if self.concatenate_kmers_with_svd:
                    x = np.hstack((multilabel_x, new_rna_features))
                else:
                    x = new_rna_features
                x = self._concatenate_rna_disease_features(x, disease_features)
            else:
                x = self._concatenate_rna_disease_features(multilabel_x.values, self.disease_similarities)
            return pd.DataFrame(x)
        else:
            return multilabel_x    
    def _concatenate_rna_disease_features(self,
                                        rna_features,
                                        disease_features):
        ## for each row in rna_features, duplicate it disease_features times and append to a new np array
        new_x = []
        for rna in rna_features:
            
            new_rows = np.tile(rna, disease_features.shape[0]).reshape((-1, rna_features.shape[1]))
#            new_rows = np.repeat(rna, disease_features.shape[0]).reshape((-1, rna_features.shape[1]))
            new_rows = np.hstack((new_rows, disease_features))
            new_x.append(new_rows)
#            print(new_rows)
#            break
        new_x = np.vstack(new_x)
        return np.array(new_x)

    def _flatten_label_space(self, y):
        return y.values.flatten(order = "C")
    def _unflatten_label_space(self, binary_y, multilabel_y):
        unflatted_y = pd.DataFrame(binary_y.values.reshape(multilabel_y.shape), columns = multilabel_y.columns)
        return unflatted_y
    def _transform_svd(self,
                    input): ## returns U and Vt
        U, _, Vh = linalg.svd(input)
        return U, Vh.T
    def _apply_masking(self, 
                      binary_y,
                      ):
        binary_y.iloc[self.indexes_to_mask.values] = 0
        return binary_y
    def get_output_name(self):
        mode = ""
        if self.transductive_mode:
            mode = "transductive"
        else:
            mode = "inductive"  
        if self.binary_mode:
            mode+= "_binary"
        else:
            mode+= "_multilabel"
        if self.concatenate_kmers_with_svd:
            mode+= "_KmersConcatenatedWithSVD"
        if self.disease_similarities is not None:
            mode+= "_diseaseSimilarities"
        return mode