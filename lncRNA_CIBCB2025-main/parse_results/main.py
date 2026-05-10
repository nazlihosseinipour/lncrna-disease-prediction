import sys
import pandas as pd
sys.path.append("../")
import numpy as np
from evaluate import Evaluator
from utils.utils import iterator_cross_validation


path_flat_labels = sys.argv[1]
path_flat_predictions = sys.argv[2]

binary_mode = sys.argv[3] == "true"
path_indexes = sys.argv[4]


flat_y = pd.read_csv(path_flat_labels, index_col=0)
indexes = pd.read_csv(path_indexes)


performance_flat = []
dummy_df = pd.DataFrame(flat_y.copy())
_, test_flat = iterator_cross_validation(indexes,
                                                    dummy_df,
                                                    flat_y)


if not binary_mode:
    path_hierarchical_labels = sys.argv[5]
    path_hierarchical_predictions = sys.argv[6]
    hierarchical_y = pd.read_csv(path_hierarchical_labels, index_col=0)
    _, test_hierarchical = iterator_cross_validation(indexes,
                                                    dummy_df,
                                                    hierarchical_y)
    performance_hierarchical = []
    performance_hierarchical_to_flat = []

n_folds = 10

for fold in range(1, n_folds + 1):
    predictions_flat = pd.read_csv(path_flat_predictions.replace("fold1", "fold" + str(fold)))
    test_flat_y = test_flat[fold - 1][1]    
    predictions_flat.columns = test_flat_y.columns
 
    evaluator = Evaluator()
    performance_flat.append(evaluator.evaluate(test_flat_y,predictions_flat))
    if not binary_mode:
        predictions_hierarchical = pd.read_csv(path_hierarchical_predictions.replace("fold1", "fold" + str(fold)))    
        test_hierarchical_y = test_hierarchical[fold - 1][1]        
        predictions_hierarchical.columns = test_hierarchical_y.columns    
    
        performance_hierarchical.append(evaluator.evaluate(test_hierarchical_y,predictions_hierarchical))
        performance_hierarchical_to_flat.append(evaluator.evaluate_flat_mesh(test_flat_y,
                                predictions_flat,
                                predictions_hierarchical))
df_flat = evaluator.create_df_inductive(performance_flat)
df_flat.to_csv(path_flat_predictions.split("/")[-1].replace("predictions_test_fold1.csv","")  + "performance.csv", index = False)

if not binary_mode:
    df_hierarchical = evaluator.create_df_inductive(performance_hierarchical)
    df_hierarchical.to_csv(path_hierarchical_predictions.split("/")[-1].replace("predictions_test_fold1.csv","")  + "performance.csv", index = False)

    df_hierarchical_to_flat = evaluator.create_df_inductive(performance_hierarchical_to_flat)
    df_hierarchical_to_flat.to_csv(path_hierarchical_predictions.split("/")[-1].replace("predictions_test_fold1.csv","")  + "hierarchical_to_flat_performance.csv", index = False)

