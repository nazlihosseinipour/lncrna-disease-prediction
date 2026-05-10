import sys
import pandas as pd
sys.path.append("../")
from rflda import RFLDA
from utils.utils import filter_labels
import numpy as np

path_features = sys.argv[1]
path_labels = sys.argv[2]
path_indexes = sys.argv[3]

x = pd.read_csv(path_features, index_col=0)
y = pd.read_csv(path_labels,index_col=0)

binary_mode = sys.argv[4] == "true"
concatenate_kmers_with_svd = sys.argv[5] == "true"
transductive_mode = sys.argv[6] == "true"

n_folds = 10
dataset = sys.argv[2].split("/")[-1].replace(".csv","_")

for i in range(n_folds):
    print(i)
    indexes_to_mask = pd.read_csv(path_indexes.replace("fold0", "fold" + str(i)), index_col=0)
    model = RFLDA(binary_mode = binary_mode,
                    concatenate_kmers_with_svd = concatenate_kmers_with_svd,
                    transductive_mode = transductive_mode,
                    indexes_to_mask = indexes_to_mask)
    mode = model.get_output_name()
    print(mode)
    model.fit(x, y)
    y_test_pred = model.predict_proba(x,y)
    pd.DataFrame(y_test_pred).to_csv(dataset + "RFLDA_" + mode + "_predictions_test_fold" + str(i + 1) + ".csv", index=False) 
    model.accuracy_find_best_n.to_csv(dataset + "accuracy_" + mode + "_find_best_n_fold" + str(i + 1) +  ".csv", index=False)
