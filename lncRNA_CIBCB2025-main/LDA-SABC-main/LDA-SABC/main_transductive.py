import sys
import pandas as pd
sys.path.append("../../")
from LDA_SABC import LDA_SABC
from utils.utils import filter_labels

path_features = sys.argv[1]
path_labels = sys.argv[2]
path_indexes = sys.argv[3]
concatenate_kmers_with_svd = sys.argv[4] == "true"

x = pd.read_csv(path_features, index_col=0)
y = pd.read_csv(path_labels, index_col=0)

binary_mode = True
transductive_mode = True

n_folds = 10
dataset = sys.argv[2].split("/")[-1].replace(".csv","_")
alpha = 0.4
for i in range(n_folds):
    print(i)
    indexes_to_mask = pd.read_csv(path_indexes.replace("fold0", "fold" + str(i)), index_col=0)
    model = LDA_SABC(alpha = alpha,
                    binary_mode = binary_mode,
                    concatenate_kmers_with_svd = concatenate_kmers_with_svd,
                    transductive_mode = transductive_mode,
                    indexes_to_mask = indexes_to_mask)
    mode = model.get_output_name()
    print(mode)
    model.fit(x, y)
    y_test_pred = model.predict_proba(x,y)[:,1:2]
    pd.DataFrame(y_test_pred).to_csv(dataset + "LDA_SABC_" + mode + "_predictions_test_fold" + str(i) + ".csv", index=False) 
