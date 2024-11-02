import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
%matplotlib inline
a=pd.a=pd.read_csv("/Users/shankusasmal/Downloads/diabetes.csv")
a.head()
x=a.drop(columns="Outcome",axis=1)
y=a["Outcome"]

from sklearn.model_selection import train_test_split
xt,xte,yt,yte=train_test_split(x,y,test_size=.1,stratify=y,random_state=3)
from sklearn.preprocessing import StandardScaler
z=StandardScaler()
xt=z.fit_transform(xt)
xte=z.fit_transform(xte)
##model on which you want to work
model=keras.Sequential([
#     keras.layers.Flatten(input_shape(8,)) ##for it no i/p shape declear into layer Dense
    keras.layers.Dense(4,input_shape=(8,),activation="relu"),
    keras.layers.Dense(2,activation="sigmoid")
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(xt,yt,epochs=5)
n=np.argmax(model.predict(xte))
print(model.evaluate(xte,yte))
data=tf.math.confusion_matrix(yte,n)
plt.figure(figsize=(10,10))
sns.heatmap(data,annot=True,fmt="d", annot_kws={'size': 124, 'weight': 'bold', 'color': 'red'})
plt.xlabel("orginal")
plt.ylabel('prediction')
plt.show()