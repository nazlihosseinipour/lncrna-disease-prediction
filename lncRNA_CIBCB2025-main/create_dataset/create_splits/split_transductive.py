import pandas as pd
import sys
sys.path.append("../../")
from utils.transductive_converter import Converter
from utils.utils import filter_labels
import numpy as np
path_labels = sys.argv[1]

converter = Converter()
y = pd.read_csv(path_labels,index_col=0)
minimum_occurrences = 5
percentage_masking = 0.2
y = filter_labels(y, minimum_occurrences = minimum_occurrences)
transductive_y = y.values.flatten(order = "F")   
positive_associations = np.where(transductive_y == 1)[0]
nb_masked_interactions = int(percentage_masking * positive_associations.shape[0])
for i in range(10):
    np.random.seed(i)
    indexes_to_mask = np.random.choice(positive_associations, nb_masked_interactions)
    indexes_to_mask = pd.DataFrame(indexes_to_mask)
    indexes_to_mask.to_csv("indexes_to_mask_fold" + str(i) + ".csv")