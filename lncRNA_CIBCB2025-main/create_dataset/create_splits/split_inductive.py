from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd
import sys

mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# Run on the hierarchical label space
path_features = sys.argv[1]
path_labels = sys.argv[2]

x = pd.read_csv(path_features, index_col=0)
y = pd.read_csv(path_labels,index_col=0)

columns = [("train_fold" + str(i), "test_fold" + str(i)) for i in range(1,11)]
columns = [a for c in columns for a in c]
splits = pd.DataFrame([index.astype(int) for indexes in mskf.split(x, y) for index in indexes]).T
splits.columns = columns
splits.to_csv(path_features.replace(".csv", "_splits.csv"), index=False)