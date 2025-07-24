import numpy as np 
import pandas as pd
import Levenshtein



def calculate_GIP_lncRNA_and_DIS(matrix ): 
    #create lables for rows (incRNAs) and columns (diseases)    #A.shape→(rows, columns)
    incRNA_lables = [f"L{i+1}" for i in range(matrix.shape[0])]      #A.shape[0] = number of rows
    dis_lables= [f"D{j+1}" for j in range(matrix.shape[1])]          #A.shape[1] = number of cloumn


    # Step 2: compute squared Euclidean distances between lncRNAs (rows)
    profiles = matrix        # Each row is a lncRNA profile


    # Compute squared norms (||x||^2 for each lncRNA)
    row_norms_squared = np.sum(profiles**2, axis=1).values

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
    norms_squared_cols = np.sum(disease_profiles**2, axis=1).values

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




    #Y=UΣV^T
    #Where:
    #Y: your original matrix (e.g. lncRNA–disease matrix)
    #U: matrix of left singular vectors (represents lncRNA features) 
    #Σ: diagonal matrix with singular values (importance of each feature)
    #V^T: transpose of right singular vectors (represents disease features)

def extract_svd_features(matrix, k) : 

    # Perform SVD decomposition
    #SVD tells us :"Hey, I found these patterns that explain the most important ways lncRNAs and diseases are related."

    matrix = np.array(matrix, dtype=float)
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)

    #i needed to do this to be able to do the multiplication 
    if k is not None: 
     U_k = U[:, :k]       # keep first k columns of U
     S_k = np.diag(S[:k]) # keep first k singular values and make them diagonal
     VT_k = VT[:k, :]     # keep first k rows of VT

    # Convert Σ Create diagonal matrix from singular values
    s = np.diag(np.sqrt(S_k))  # use sqrt(s) to match paper logic (bcs usually it's used for both in the original formula so now it's one item we also do want half of it)


    # Feature matrices
    lncRNA_features = U_k @ s       # left side = lncRNA 
    disease_features = (s @ VT_k).T     # right side = disease

    return lncRNA_features, disease_features


def editDistance (seqA , seqB ): 
    # matrix = np.zeros(size=(size.length(), seqB.legnth()))
    # #create lables for rows (incRNAs) and columns (diseases)    #A.shape→(rows, columns)
    # seqALables = [f"L{i+1}" for i in range(matrix.shape[0])]      #A.shape[0] = number of rows
    # seqBLables= [f"D{j+1}" for j in range(matrix.shape[1])]          #A.shape[1] = number of cloumn
    # matrixReadable = pd.DataFrame(matrix, index=seqALables, columns=seqBLables)
    # i = 0 
    # j = 0 
    # for i in range(seqA):
    #     for j in range(seqB):
    #        if seqA[i] == seqB[j]: 
    #           matrixReadable[i] = matrixReadable[j] = 1 
    
    return Levenshtein.distance(seqA, seqB)
    # still working on it wanna learn the logic and write the algo myself 
            