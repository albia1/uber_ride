import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df= pd.read_csv("taxi.csv")
print(df.head)

X = df.iloc[:,0:-1].values
y= df.iloc[:,-1].values
print(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
regressor=LinearRegression()
regressor.fit(X_train,y_train)
print("train score",regressor.score(X_train,y_train))
print("test score",regressor.score(X_test,y_test))
pickle.dump(regressor,open("taxi.pkl","wb"))
model=pickle.load(open("taxi.pkl","rb"))
print(model.predict([[80,1778859,6000,85]]))
