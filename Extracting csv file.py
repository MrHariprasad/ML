import pandas as pd
dataset=pd.read_csv("27.csv",delimiter=',')
print(dataset) 
X =dataset[['Names']].values
Y = dataset[['Roll Number']].values
print(Y) 
print(X)
X1 = dataset[['Names','Roll Number']].values
print(X1) 
print(X[0:5])

'''

import pandas as pd

#Load the entire CSV file
dataset = pd.read_csv("27.csv", delimiter=',')

#Print the entire dataset
print("Full Dataset:")
print(dataset)

#If you want the values as a NumPy array:
data_values = dataset.values
print("\nAll Values (as NumPy array):")
print(data_values)

#Optionally, print first 5 rows
print("\nFirst 5 Rows:")
print(dataset.head())

'''
