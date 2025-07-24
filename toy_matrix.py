import numpy as np 
import pandas as pd 

class ToyMatrix:

    def __init__ (self):
       self.data = self.createData()



    def createData(self): 
        np.random.seed(2)           #so we get each time the same random repeatable matrix 
        num_incRNA = 5              #number of fake different incRNA's (rows)
        num_dis = 4                 #number of different fake Disseas's (cloumn )
                    
        A = np.random.randint(0, 2 , size=(num_incRNA, num_dis))#so here we create the matrix and we give :  0 or 1 (random tho)

            #create lables for rows (incRNAs) and columns (diseases)    #A.shape→(rows, columns)
        incRNA_lables = [f"L{i+1}" for i in range(A.shape[0])]      #A.shape[0] = number of rows
        dis_lables= [f"D{j+1}" for j in range(A.shape[1])]          #A.shape[1] = number of cloumn
        matrixToy = pd.DataFrame(A, index=incRNA_lables, columns=dis_lables)
        return matrixToy

  


    def getData(self): 
        return self.data