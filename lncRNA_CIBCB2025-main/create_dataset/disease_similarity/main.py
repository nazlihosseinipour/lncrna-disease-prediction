import sys
import pandas as pd
from similarity_measurer import SimilarityMeasurer


tree_c = pd.read_csv("Tree_C.csv", index_col = 0)
flat_labels = pd.read_csv(sys.argv[1], index_col = 0)

sm = SimilarityMeasurer(tree_c)

similarities = sm.get_similarities(flat_labels)

similarities.to_csv("disease_similarities.csv", index = False)