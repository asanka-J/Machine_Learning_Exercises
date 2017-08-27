# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('50_Startups.csv')


# Encoding categorical data //because linear regression needs numbers
dataset$State = factor(dataset$State,
                         levels = c('New York', 'California', 'Florida'),
                         labels = c(1, 2, 3))


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Feature Scaling //- no need replaces by a function we use

#Fitting multiple Linear Regression to the training set
regressor=lm(formula = Profit~. ,data=training_set)#lm(relationship between dependent and independent Variables)


#Since R & D spent is the highly strongly predictor other variables are not much useful 
#regressor=lm(formula = Profit~R.D.Spend ,data=training_set)

#predictiing the test set results
y_pred=predict(regressor,newdata=test_set)
 
#making optimal model using ~ BACKWORD ELIMINATION ~
regressor=lm(formula = Profit~R.D.Spend+Administration+Marketing.Spend+State ,data=training_set)
y_pred=predict(regressor,newdata=dataset)
summary(regressor)


#making optimal model using ~ BACKWORD ELIMINATION ~
regressor=lm(formula = Profit~R.D.Spend+Administration+Marketing.Spend ,data=training_set)#Remove state since p=0.9xx
y_pred=predict(regressor,newdata=dataset)
summary(regressor)

#making optimal model using ~ BACKWORD ELIMINATION ~
regressor=lm(formula = Profit~R.D.Spend+Marketing.Spend,data=training_set)#Remove Administration since p>SL
y_pred=predict(regressor,newdata=dataset)
summary(regressor)

#making optimal model using ~ BACKWORD ELIMINATION ~
regressor=lm(formula = Profit~R.D.Spend,data=training_set)#Remove Marketing.Spend since p>SL
y_pred=predict(regressor,newdata=dataset)
summary(regressor)#making optimal model using ~ BACKWORD ELIMINATION ~

regressor=lm(formula = Profit~R.D.Spend+Marketing.Spend,data=training_set)#Since marketing.Spend have no big difference,keep it
y_pred=predict(regressor,newdata=dataset)
summary(regressor)