# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Splitting the dataset into the Training set and Test set -- no need to devide to training and test set cuz 
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling -- done by library
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting linear Regression to the dataset

from sklearn.linear_model import LinearRegression
lin_Reg =LinearRegression()
lin_Reg.fit(X,y)

#Fitting Polynomial Regreesion to the dataset
from sklearn.preprocessing import PolynomialFeatures#-gives tools to import polynomial features to linear regression model
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)#fit poly

lin_Reg2 =LinearRegression()
lin_Reg2.fit(X_poly,y)

#Visualizing the Linear Regression results
plt.scatter(X,y,color='red')
plt.plot(X,lin_Reg.predict(X),color='blue')
plt.title('Truth or bluff (Linear Regression model)')
plt.xlabel('position')
plt.ylabel('Salary')
plt.show()


#Visualizing the Polynomial Regression results
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_Reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or bluff (Polynomial Regression model)')
plt.xlabel('position')
plt.ylabel('Salary')
plt.show()

#Predictiong the salary for given input -Linear regression
lin_Reg.predict(6.5)

#Predictiong the salary for given input -Polynomial regression
lin_Reg2.predict(poly_reg.fit_transform(6.5))