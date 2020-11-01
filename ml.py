import pandas as pd 
from sklearn import model_selection
import sklearn
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics
df=pd.read_csv("C:\\Users\\ersd7\\OneDrive\\Documents\\diabetes.csv")
print(df.head())
#cleaning data 
df["Glucose"]=df["Glucose"].fillna(0)
x=df.iloc[:,:-1]
y=df.iloc[:,[-1]]
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2,random_state=0)
clf=RandomForestClassifier(n_estimators=6,criterion='entropy',random_state=0)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print(y_pred)
print("this is the accuracy ",clf.score(x_test,y_test))



  