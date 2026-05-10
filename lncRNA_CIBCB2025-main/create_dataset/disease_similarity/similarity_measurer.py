import networkx as nx
import numpy as np
import itertools
import pandas as pd
class SimilarityMeasurer():

    root = "root"
    initial_semantic_contribution = 1
    def __init__(self,
                tree_c,
                weight = 0.5):
        self.weight = weight
        self.tree_c = tree_c
        self.DAG = nx.DiGraph()
        self._populate()
    def _populate(self):
#        self.DAG.add_node(self.root)
        for i, row in self.tree_c.iterrows(): 
            code = row["TREE_NUMBER"]
            disease = row["DESCRIPTOR"]
            path_to_root = self._get_path_to_root(code)
            self.DAG.add_node(code)
            self.DAG.add_edges_from(path_to_root)
            nx.set_node_attributes(self.DAG,{code:{"disease": disease}})
    def _get_path_to_root(self,
                        code):
        
        path_to_root = [[".".join(code.split(".")[:i]) , ".".join(code.split(".")[:i+1])] for i,v in enumerate(code.split("."))]
        path_to_root[0][0] = self.root
        path_to_root = tuple(path_to_root)
        return path_to_root
    
    def get_similarities(self,
                    flat_y):
        labels = list(flat_y.columns)
        disease_interactions = itertools.product(*[labels,labels])

        disease_interactions = list(itertools.product(*[labels,labels]))
        #print(len(disease_interactions))
        similarities = []
        for i, interaction in enumerate(disease_interactions):
            print(i)
            semantic_disease1 = self._get_semantic_contributions(interaction[0])
            semantic_disease2 = self._get_semantic_contributions(interaction[1])
            
            sum_semantic_disease1 = np.sum(list(semantic_disease1.values()))
            sum_semantic_disease2 = np.sum(list(semantic_disease2.values()))
#            print(sum_semantic_disease1)
            node_intersection = set.intersection(set(semantic_disease1.keys()), set(semantic_disease2.keys()))

            summed_intersection = np.sum(semantic_disease1[node_in] + semantic_disease2[node_in] for node_in in node_intersection)
            similarity = summed_intersection/(sum_semantic_disease1 + sum_semantic_disease2)    
#            print(similarity)
            similarities.append(similarity)
        similarities = pd.DataFrame(np.array(similarities).reshape((len(labels),len(labels))))
        return similarities
    def _get_semantic_contributions(self, 
                                disease):
        disease_nodes = self._get_nodes_by_disease(disease)
        all_nodes_to_root = self._get_all_nodes_to_root(disease_nodes)
        semantic_contributions = dict(zip(disease_nodes, np.full(len(disease_nodes), self.initial_semantic_contribution)))
        for node in all_nodes_to_root:
            children = nx.descendants_at_distance(self.DAG, node, 1)
            max_semantic_children = np.max(list(semantic_contributions[child] for child in children if child in semantic_contributions))
            semantic_contributions[node] = max_semantic_children * self.weight         
        return semantic_contributions
    def _get_all_nodes_to_root(self, 
                                disease_nodes): 
        all_paths_to_root = [list(nx.all_simple_paths(self.DAG, self.root, disease_node))[0] for disease_node in disease_nodes]
#        print(all_paths_to_root)
        all_paths_to_root = np.unique([node for path in all_paths_to_root for node in path[1:-1]]) ## [2:] drops the root and the initial node, since they are already calculated
#        print(all_paths_to_root)
        depth_nodes = np.array([len(node) for node in all_paths_to_root])
        sorted_nodes_index = np.argsort(depth_nodes)[::-1]
        sorted_nodes = all_paths_to_root[sorted_nodes_index]
#        print(sorted_nodes)
        
        return sorted_nodes
 #       semantic_contribution[disease] = 
    def _get_nodes_by_disease(self, disease):
        nodes = []
        for node, attrs in self.DAG.nodes.data():
            if disease == attrs.get("disease"):
                nodes.append(node)
        return nodes
    def _is_root(self,
            node):
        return node == self.DAG.nodes[self.root]
