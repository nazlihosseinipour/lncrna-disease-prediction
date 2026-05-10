import numpy as np
import pandas as pd



f_r = np.load('./data1/r10_feature.npy')
f_d = np.load('./data1/d10_feature.npy')

all_associations = pd.read_csv('./data1' + '/pair.txt', sep=' ', names=['r', 'd', 'label'])


dataset = []

for i in range(int(all_associations.shape[0])):
    r = all_associations.iloc[i, 0]
    c = all_associations.iloc[i, 1]
    label = all_associations.iloc[i, 2]
    dataset.append(np.hstack((f_r[r], f_d[c], label)))

all_dataset = pd.DataFrame(dataset)

all_dataset.to_csv("./data1/data10linear.csv",header=None,index=None)

print("Finished!")