import numpy as np 
import pandas as pd
import toy_matrix as Toy
import extract_features as feature



def main (): 
    createdMatrix = Toy.ToyMatrix()
    matrix = createdMatrix.getData()
    print("the toy matrix : incrRNA_dis (rows = incrRNA , col = Diseases );")
    print(matrix)



    print("\nGIP kernel similarity between diseases & incRNA :")
    print( feature.calculate_GIP_lncRNA_and_DIS(matrix)) 
    print("SVD features are shown here : ")
    print(feature.extract_svd_features(matrix, k = 64)) 

    #print("lables version of the table")
    #print(df_A)
    #print(A.shape)
    #print(A.shape[0])
    #print("\nA**2 (square each element):\n", A**2)

    #print("\nnp.sum(A**2, axis=1)  # one number per row")
    #print(np.sum(A**2, axis=1))

    #print("\nnp.sum(A**2, axis=0)  # one number per column")
    #print(np.sum(A**2, axis=0))


main()



if __name__ == "__main__":
    main()
