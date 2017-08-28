# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values 
# Splitting the dataset into the Training set and Test set -- no need to devide to training and test set cuz 
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling - no need library is doing this job for us
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting the Regreesion to the dataset
#from sklearn.preprocessing import PolynomialFeatures#-gives tools to import polynomial features to linear regression model
#poly_reg=PolynomialFeatures(degree=4)
#X_poly=poly_reg.fit_transform(X)#fit poly
#
#lin_Reg2 =LinearRegression()
#lin_Reg2.fit(X_poly,y)

#Predictiong the salary for given input -Polynomial regression
regressor=
y_pred=regressor.predict(6.5)




#Visualizing the Polynomial Regression results
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_Reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or bluff (Polynomial Regression model)')
plt.xlabel('position')
plt.ylabel('Salary')
plt.show()
