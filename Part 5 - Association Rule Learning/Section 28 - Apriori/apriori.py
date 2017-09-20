# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#dataset : Market_Basket_Optimisation    ,products bought in market over a week

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
    
#We need to make a list of list of dataset to pass Apriori function 
#transactions=np.array(dataset.iloc[:,:].values).tolist() #not working : not supported between instances of 'float' and 'str'



# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

#try combinantions until we find optimal rules value
"""in_length -> min no of products in basket
   min_support -> is asssume products purchased 3 times a day:for week 3*7 =>minsupport = 3*7/7500=0.003ish
   min_confidence-> some dont have logical bought relationship between products(violates if buy this it will lead to buying that relationship )
   hight confidence means accuracy of buying products ;here we choose 20%
   min_lift= put 3 hoping we'll get more relavant rules (min_lift  ==== relavance )
  """
# Visualising the results
results = list(rules)