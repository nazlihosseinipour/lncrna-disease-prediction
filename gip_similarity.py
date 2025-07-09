import numpy as np 
import pandas as pd

def calculat_GIP_lncRNA_and_DIS(matrix ): 
    #create lables for rows (incRNAs) and columns (diseases)    #A.shape→(rows, columns)
    incRNA_lables = [f"L{i+1}" for i in range(matrix.shape[0])]      #A.shape[0] = number of rows
    dis_lables= [f"D{j+1}" for j in range(matrix.shape[1])]          #A.shape[1] = number of cloumn


    df_A = pd.DataFrame(matrix, index=incRNA_lables, columns=dis_lables)
    print("lables version of the table")
    print(df_A)


    # Step 2: compute squared Euclidean distances between lncRNAs (rows)
    profiles = matrix        # Each row is a lncRNA profile


    # Compute squared norms (||x||^2 for each lncRNA)
    row_norms_squared = np.sum(profiles**2, axis=1)

    # Compute distance matrix using formula:
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * xᵀy
    dot_product = profiles @ profiles.T
    dists_squared_row = (
        row_norms_squared.reshape(-1, 1) +
        row_norms_squared.reshape(1, -1) -
        2 * dot_product
    )

    # turn into DataFrame for readability
    df_dists = pd.DataFrame(dists_squared_row, index=incRNA_lables, columns=incRNA_lables)

    print("\nSquared distances between lncRNAs:")
    print(df_dists)

    # calculate 𝛾 = 1 / average row norm square

    #calculate gip kernel : K(i,j) = exp (−γ⋅ ∥x i −x j∥ ^  2)
    #  ∥x i −x j∥ ^  2 = square_dif 
    # --- GIP for lncRNAs ---
    gamma_lnc = 1 / np.mean(row_norms_squared)
    gip_lnc = np.exp(-gamma_lnc * dists_squared_row)

    df_gip_lnc = pd.DataFrame(gip_lnc, index=incRNA_lables, columns=incRNA_lables)

    #the same pattern but now just for DIS so we use the transpose version of the thing 
    
    # Step 3: compute squared Euclidean distances between diseases (columns)
    disease_profiles = matrix.T    # transpose: now rows = diseases
    m = disease_profiles.shape[0]

    # Compute squared norms
    norms_squared_cols = np.sum(disease_profiles**2, axis=1)

    # Compute dot product and squared distance matrix
    dot_product_cols = disease_profiles @ disease_profiles.T
    dists_squared_cols = (
        norms_squared_cols.reshape(-1, 1) +
        norms_squared_cols.reshape(1, -1) -
        2 * dot_product_cols
    )

    # turn into DataFrame for readability
    df_dists_cols = pd.DataFrame(dists_squared_cols, index=dis_lables, columns=dis_lables)

    print("\nSquared distances between diseases:")
    print(df_dists_cols)


    # calculate 𝛾 = 1 / average row norm square

    #calculate gip kernel : K(i,j) = exp (−γ⋅ ∥x i −x j∥ ^  2)
    #  ∥x i −x j∥ ^  2 = square_dif 
    # --- GIP for diseases ---
    gamma_dis = 1 / np.mean(norms_squared_cols)
    gip_dis = np.exp(-gamma_dis * dists_squared_cols)

    df_gip_dis = pd.DataFrame(gip_dis, index=dis_lables, columns=dis_lables)
    return df_gip_dis , df_gip_lnc




def main (): 
    np.random.seed(2)           #so we get each time the same random repeatable matrix 
    num_incRNA = 5              #number of fake different incRNA's 
    num_dis = 4                 #number of different fake Disseas's

    A = np.random.randint(0, 2 , size=(num_incRNA, num_dis))#so here we create the matrix and we give :  0 or 1 (random tho)
    print("the toy matrix : incrRNA_dis (rows = incrRNA , col = Diseases );")
    print(A)



    #create lables for rows (incRNAs) and columns (diseases)    #A.shape→(rows, columns)
    incRNA_lables = [f"L{i+1}" for i in range(A.shape[0])]      #A.shape[0] = number of rows
    dis_lables= [f"D{j+1}" for j in range(A.shape[1])]          #A.shape[1] = number of cloumn


    df_A = pd.DataFrame(A, index=incRNA_lables, columns=dis_lables)
    print("lables version of the table")
    print(df_A)
    print(A.shape)
    print(A.shape[0])



    print("\nA**2 (square each element):\n", A**2)

    print("\nnp.sum(A**2, axis=1)  # one number per row")
    print(np.sum(A**2, axis=1))

    print("\nnp.sum(A**2, axis=0)  # one number per column")
    print(np.sum(A**2, axis=0))

    print("\nGIP kernel similarity between diseases & incRNA :")
    print( calculat_GIP_lncRNA_and_DIS(A)) 



main()



if __name__ == "__main__":
    main()
