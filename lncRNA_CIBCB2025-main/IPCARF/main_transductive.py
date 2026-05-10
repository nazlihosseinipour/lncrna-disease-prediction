import sys
import pandas as pd
sys.path.append("../")
from ipcarf import IPCARF
from utils.utils import filter_labels
import numpy as np

path_features = sys.argv[1]
path_labels = sys.argv[2]
path_indexes = sys.argv[3]
binary_mode = sys.argv[4] == "true"
concatenate_kmers_with_svd = sys.argv[5] == "true"
transductive_mode = sys.argv[6] == "true"

x = pd.read_csv(path_features, index_col=0)
y = pd.read_csv(path_labels, index_col=0)

n_folds = 10    
best_n_components = []

dataset = sys.argv[2].split("/")[-1].replace(".csv","_")

for i in range(n_folds):
    indexes_to_mask = pd.read_csv(path_indexes.replace("fold0", "fold" + str(i)), index_col=0)
    model = IPCARF(binary_mode = binary_mode,
                    concatenate_kmers_with_svd = concatenate_kmers_with_svd,
                    transductive_mode = transductive_mode,
                    indexes_to_mask = indexes_to_mask)
    model.fit(x, y)
    y_test_pred = model.predict_proba(x,y) # y is ignored if not needed
    mode = model.get_output_name()
    print(mode)
    pd.DataFrame(y_test_pred).to_csv(dataset + "IPCARF_" + mode + "_predictions_test_fold" + str(i + 1) + ".csv", index=False) 
    best_n_components.append(model.n_components)
pd.DataFrame(best_n_components).to_csv(dataset + mode + "_best_n_components.csv", index = False)
 