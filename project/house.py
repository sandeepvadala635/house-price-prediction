import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from math import sqrt
import pickle

df=pd.read_csv("A:/project/house price data.csv")
print(df)

print(df.shape)

print(df.describe())

print(df.info())

print(sns.pairplot(data = df))
print(plt.show())

print(sns.heatmap(df.corr(),annot=True))
print(plt.title("correlation heatmap"))
print(plt.show())

print(plt.hist(df['Price']))
print(plt.title("Histogram of price"))
print(plt.xlabel("house prices"))
print(plt.ylabel("frequency"))
print(plt.show())

print(sns.countplot(df['No.of Bedrooms']))
print(plt.title("plot of no.of bedrooms"))
print(plt.xlabel("no.of bedrooms"))
print(plt.ylabel("frequency"))
print(plt.show())

print(plt.hist(df['Area']))
print(plt.title("histogram of Area"))
print(plt.xlabel("parking"))
print(plt.ylabel("frequency"))
print(plt.show())

print(sns.countplot(df['Parking']))
print(plt.title("parking plot"))
print(plt.xlabel("parking"))
print(plt.ylabel("frequency"))
print(plt.show())

print(sns.countplot(df['Power Backup']))
print(plt.title("count plot of Power Backup"))
print(plt.xlabel("Power Backup"))
print(plt.ylabel("frequency"))
print(plt.show())

print(sns.countplot(df['Resale']))
print(plt.title("count plot of resale"))
print(plt.xlabel("Resale"))
print(plt.ylabel("frequency"))
print(plt.show())

y=df.iloc[:,0]
X=df.iloc[:,1:6]
print(X)
print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=3)
print(X_train.shape)
print(X_test.shape)

lr=LinearRegression()
lr.fit(X_train,y_train)

rfr=RandomForestRegressor()
rfr.fit(X_train,y_train)

y_pred1=lr.predict(X_test)



y_pred2=rfr.predict(X_test)




df1=pd.DataFrame({"Actual":y_test,"lr":y_pred1,"rfr":y_pred2,})
print(df1)

print(plt.scatter(y_test,y_pred1,color="green"))
print(plt.title("scatter plt of actual values and prediction of linear regression"))
print(plt.xlabel("predicted values "))
print(plt.ylabel("actual values"))
print(plt.show())

print(plt.scatter(y_test,y_pred2,color="red"))
print(plt.title("scatter plt of actual values and prediction of random forest regression"))
print(plt.xlabel("predicted values "))
print(plt.ylabel("actual values"))
print(plt.show())

lr_accuracy=r2_score(y_test,y_pred1)
print("lr_accuracy:",lr_accuracy)

rfr_accuracy=r2_score(y_test,y_pred2)
print("rfr_accuracy:",rfr_accuracy)

print(plt.bar(["LR","RFR"],[lr_accuracy,rfr_accuracy]))
print(plt.title("lr_accuracy  V/S rfr_accuracy"))
print(plt.ylabel("ACCURACY"))
print(plt.show())

lr_rmse=sqrt(mean_squared_error(y_test,y_pred1))
print("RMSE of lr:",lr_rmse)

rfr_rmse=sqrt(mean_squared_error(y_test,y_pred2))
print("RMSE of rfr:",rfr_rmse)

print(plt.bar(["LR","RFR"],[lr_rmse,rfr_rmse]))
print(plt.title("RMSE of LR  V/S  RMSE of RFR"))
print(plt.ylabel("RMSE"))
print(plt.show())

rfr=RandomForestRegressor()
rfr.fit(X,y)
pickle.dump(rfr,open("model.pkl",'wb'))
model=pickle.load(open("model.pkl",'rb'))
