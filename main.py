import numpy as np 
import pandas as pd
import toy_matrix as Toy
import RnaFeatures 
import DiseaseFeatures



def main (): 
    createdMatrix = Toy.ToyMatrix()
    matrix = createdMatrix.getData()



    print("\nGIP kernel similarity between diseases & incRNA :")
    rna = RnaFeatures.RnaFeatures()
    result =  Rna.calculate_GIP_lncRNA_and_DIS(matrix)
    print(result) 
    print("SVD features are shown here : ")
    print(Rna.extract_svd_features(matrix, k=64)) 

    Dis = DiseaseFeatures.DiseaseFeatures()
    print(Dis.calculate_GIP_DIS(matrix))




if __name__ == "__main__":
    main()
