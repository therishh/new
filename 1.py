#data preprocessiong
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y =  dataset.iloc[:,3].values

#taking care of missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#encoding data 
#from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#labelencoder_x = LabelEncoder()
#X[:,0] = labelencoder_x.fit_transform(X[:,0])
#onehotencoder = OneHotEncoder(handle_unknown ='ignore')
#X = onehotencoder.fit_transform(X).toarray()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0])],remainder='passthrough')
X = np.array(transformer.fit_transform(X), dtype=np.float)
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

#splitting the dataset into test and training set
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)



























