import sys
import pandas as pd
sys.path.append("../../")
from LDA_SABC import LDA_SABC
from utils.utils import filter_labels
from utils.utils import iterator_cross_validation
import numpy as np

path_features = sys.argv[1]
path_labels = sys.argv[2]
path_indexes = sys.argv[3]
binary_mode = sys.argv[4] == "true"

x = pd.read_csv(path_features, index_col=0)
y = pd.read_csv(path_labels,index_col=0)

indexes = pd.read_csv(path_indexes)

n_folds = 10
train, test = iterator_cross_validation(indexes,
                                                    x,
                                                    y)
dataset = sys.argv[2].split("/")[-1].replace(".csv","_")

concatenate_kmers_with_svd = False
transductive_mode = False
indexes_to_mask = None

if binary_mode:
    disease_similarities = pd.read_csv(sys.argv[5])
else:
    disease_similarities = None

for fold in range(4,n_folds):
    print(fold)
    x_train = train[fold][0]
    y_train = train[fold][1]

    x_test = test[fold][0]
    y_test = test[fold][1]
    model = LDA_SABC(binary_mode = binary_mode,
                    concatenate_kmers_with_svd = concatenate_kmers_with_svd,
                    transductive_mode = transductive_mode,
                    indexes_to_mask = indexes_to_mask,
                    disease_similarities = disease_similarities)
    model.fit(x_train, y_train)
    mode = model.get_output_name()
    print(mode)
    y_test_pred = np.array(model.predict_proba(x_test, y_test)) ### n_labels, n_instances, 2    
    pd.DataFrame(y_test_pred, columns = y.columns ).to_csv(dataset + "LDA_SABC_" + mode +"_predictions_test_fold" + str(fold + 1) + ".csv", index=False) 
    fold +=1
