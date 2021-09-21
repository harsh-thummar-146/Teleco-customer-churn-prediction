import pandas as pd

df = pd.read_csv('./WA_Fn-UseC_-Telco-Customer-Churn.csv')


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df.drop('customerID',axis='columns',inplace=True)

df[pd.to_numeric(df.TotalCharges,errors='coerce').isna()]

df1=df[df.TotalCharges!=' ']
df1.shape

df1.TotalCharges=pd.to_numeric(df1.TotalCharges)


def unique_col_values(df):
  for col in df:
    if df[col].dtype=='object':
      print(f'{col}: {df[col].unique()}')
 
df1.replace('No internet service','No',inplace=True)
df1.replace('No phone service','No',inplace=True)

yes_no_columns= ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']

for col in yes_no_columns:
  df1[col].replace({'Yes':1,'No':0},inplace=True)

df1['gender'].replace({'Male':1,'Female':0},inplace=True)


df2=pd.get_dummies(data=df1,columns=['InternetService','Contract','PaymentMethod'])
df2.columns


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

to_be_scaled=['tenure','MonthlyCharges','TotalCharges']

df2[to_be_scaled]=scaler.fit_transform(df2[to_be_scaled])
df2.head()

X=df2.drop('Churn',axis='columns')
y=df2.Churn

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)

import tensorflow as tf
from tensorflow import keras

model=keras.Sequential([
                        keras.layers.Dense(26,input_shape=(26,),activation='relu'),
                        keras.layers.Dense(15, activation='relu'),
                        keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    
              optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']
)

model.fit(X_train,y_train,epochs=100)


model.evaluate(X_test,y_test)

yp=model.predict(X_test)
yp[0:5]

y=[]

for i in yp:
  if i>=0.5:
    y.append(1)
  else:
    y.append(0)
    
from sklearn.metrics import confusion_matrix,classification_report

print(classification_report(y,y_test))

 
