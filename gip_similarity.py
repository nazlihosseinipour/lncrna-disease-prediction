import numpy as np 
import pandas as pd


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


# Step 2: compute squared Euclidean distances between lncRNAs (rows)
profiles = A        # Each row is a lncRNA profile
n = profiles.shape[0]

# Compute squared norms (||x||^2 for each lncRNA)
norms_squared = np.sum(profiles**2, axis=1)

# Compute distance matrix using formula:
# ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * xᵀy
dot_product = profiles @ profiles.T
dists_squared = (
    norms_squared.reshape(-1, 1) +
    norms_squared.reshape(1, -1) -
    2 * dot_product
)

#  turn into DataFrame for readability
df_dists = pd.DataFrame(dists_squared, index=incRNA_lables, columns=incRNA_lables)

print("\nSquared distances between lncRNAs:")
print(df_dists)
