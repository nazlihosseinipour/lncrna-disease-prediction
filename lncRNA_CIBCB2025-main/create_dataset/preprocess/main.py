import sys
import pandas as pd
sys.path.append("../../")
from utils.utils import filter_labels
import numpy as np
from disease_converter import Converter

path_features = sys.argv[1] # raw_data KMERS
path_labels = sys.argv[2] # raw_data sequences
minimum_occurrences = 5
x = pd.read_csv(path_features, index_col=0)
y = pd.read_csv(path_labels, index_col=0)

c = Converter()
# DROP seqs
ids = y["ID"]

y = y.drop(["ID","seqs"], axis = 1)
y = filter_labels(y, minimum_occurrences = minimum_occurrences)

y["lymphoma"] = (y.lymphoma.astype(bool) + y.cancer.astype(bool)).astype(int) ### lymphoma and cancer merged as single column because of MeSH terms
y = y.drop("cancer", axis = 1)
y.columns = c.get_new_columns_flat(list(y.columns))
new_features = x.loc[y.index]
new_ids = ids.loc[y.index]

#print(y.sum(axis=1))
#print(y.sum() > 5)

new_features = pd.concat([new_ids, new_features], axis = 1)
new_y = pd.concat([new_ids, y], axis = 1)

new_features.to_csv("kmers_preprocessed.csv", index=False)
new_y.to_csv("labels_flat_preprocessed.csv", index=False)

