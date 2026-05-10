import pandas as pd
import functools
import sys
import numpy as np
from operator import itemgetter
sys.path.append("../../../")

def retrieveMeshTreeCode(x):
    terms = [t.replace("_ ",", ").lstrip() for t in x.split(",")]
    mesh_terms = [mesh_tree.loc[mesh_tree["DESCRIPTOR"] == t]["TREE_NUMBER"] for t in terms]
    mesh_terms = pd.concat(mesh_terms, ignore_index=True)
    if len(mesh_terms) > 0:
        mesh_terms = functools.reduce(lambda x, y: str(x) + "@" + str(y) , mesh_terms) 
        mesh_terms = createSubClasses(mesh_terms)
#        print(mesh_terms)
    else:
        mesh_terms = ""        
    return mesh_terms


# def buildFullLabelSet(x):
# #    print(functools.reduce(lambda x, y: str(x) + str(y).replace(" ",""), x).split(","))
# #    print(len(set(functools.reduce(lambda x, y: str(x) + str(y).replace(" ",""), x).split(","))))
    
#     allLabels = list(set(functools.reduce(lambda x, y: str(x.replace(" ", "")) + "," + str(y.replace(" ", "")), x).split(",")))
#     allLabels.sort()
#     labelSet = dict(zip(allLabels,range(len(allLabels))))
#     return labelSet



def createSubClasses(x):
    classes = [x]
    subClasses = list(set([".".join(c.split(".")[:i+1])  for c in classes for i,k in enumerate(c.split("."))]))
    subClasses.sort()
    #subClasses = str(subClasses).replace("[","").replace("]","").replace("'","")
    return subClasses

def build_dict_flat_mesh(flat_labels,
                    tree_c): ## tree_c file
    flat_label_to_mesh = {}
    for label in flat_labels:
        mesh_terms = tree_c.loc[mesh_tree["DESCRIPTOR"] == label]["TREE_NUMBER"]
        paths_to_root = []
        for term in mesh_terms:
            paths_to_root.extend(createSubClasses(term))
        paths_to_root.append(label)
        paths_to_root = np.unique(paths_to_root)            
        flat_label_to_mesh[label] = paths_to_root
    return flat_label_to_mesh

def replace_flat_by_mesh(flat_labels, 
                         dict_flat_to_mesh):
    new_labels = []
    for _, row in flat_labels.iterrows():
        annotated_diseases = list(row.loc[row==1].index)
        new_labels_row = [dict_flat_to_mesh[disease] for disease in annotated_diseases]
        new_labels_row = np.unique([b for a in new_labels_row for b in a ]) ## flatten
        new_labels.append(new_labels_row)
    return new_labels

def create_new_label_space(new_labels,
                            complete_label_set):
    binarized_labels = []
    for row_label in new_labels:
        binarized_labels_row = np.zeros(len(complete_label_set))
        if len(row_label) > 1:
            indexes = itemgetter(*row_label)(complete_label_set)
            binarized_labels_row[indexes,] = 1
        binarized_labels.append(binarized_labels_row)
    columns = list(complete_label_set.keys())
    binarized_labels = pd.DataFrame(binarized_labels, columns = columns)
    return binarized_labels

def binarizeLabels(x, labelSet):
    newLabels = np.zeros(len(labelSet))
    labels = [l.replace(" ","") for l in x.split(",")]
    indexes = itemgetter(*labels)(labelSet)
    newLabels[indexes,] = 1
    newLabels = pd.Series(newLabels, index = labelSet.keys())
    return newLabels 

def build_full_label_set(new_labels):
    complete_label_set = []
    complete_label_set.extend([labels for labels in new_labels])
    complete_label_set = np.unique([b for a in complete_label_set for b in a ]) ## flatten
    complete_label_set.sort()
    label_set = dict(zip(complete_label_set,range(len(complete_label_set))))
    return label_set
if __name__ == "__main__":
    path_flat_labels = sys.argv[1]
    mesh_tree = pd.read_csv("Tree_C.csv", index_col = 0 )
    flat_labels = pd.read_csv(path_flat_labels)
    ids = flat_labels["ID"]
    flat_labels = flat_labels.drop("ID", axis = 1)
    dict_flat_to_mesh = build_dict_flat_mesh(list(flat_labels.columns),
                        mesh_tree)
    new_labels = replace_flat_by_mesh(flat_labels, dict_flat_to_mesh)
    full_label_set = build_full_label_set(new_labels)
    binarized_labels = create_new_label_space(new_labels, full_label_set)
    binarized_labels.insert(0,"ID", ids)
    binarized_labels.to_csv("mesh_labels_final.csv", index=False)
