# Importing required libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)
import datetime
#import pandas_profiling as pp
from sklearn import preprocessing
#%matplotlib inline
import seaborn as sns

# Loading data
excel= pd.ExcelFile("Bikewale_Data_Result_Final v2.0 (1).xlsx")
df=excel.parse('Bikewale_Data_Result_Final')
df.drop(columns=['Unnamed: 15','Unnamed: 16','Unnamed: 17','Url','Date Updated'],inplace = True)
df.rename(index=str, columns={"Km/s": "Distance"},inplace=True)
print(df.head())

df.shape
df.info()
df.describe()
df.isnull().sum()

# Checking for model year and registration year
test = df[['Model',' Registration year ']].isna()
test['check'] = df['Model']== df[' Registration year ']
test[test['check'] == True]

# Preprocessing columns (removing white spaces, unwanted characters)
df['   Registration no. '] = df['   Registration no. '].str.split('\t').str[0]
df['   Registration no. '].fillna(0,inplace=True)
df['Insurance'] = df['Insurance'].str.split('\t').str[0]
df['Insurance'].fillna(0,inplace=True)
df['Insurance'].replace(to_replace='NA', value=0, regex=True, inplace=True)

for i in range(0,df.shape[0]):
  if ((df.iloc[i]['Insurance']) != 0):
    df.iloc[i]['Insurance'] = df.iloc[i]['Insurance'].replace(" ", "")

df['Insurance'].unique()
df['Insurance'].replace({'ThirdParty':3 ,'NoInsurance':1 ,'Comprehensive':2}, inplace=True)
df.dropna(inplace=True)
print(df.head())

import re
df['Model'] = df['Model'].apply(lambda x: re.sub('[^0-9]+', ' ',x))
df['Distance'] = df['Distance'].apply(lambda x: re.sub('[^0-9]+', ' ',x))
df['Distance'] = df['Distance'].apply(lambda x: x.strip())

def remove_space(string): 
    return string.replace(" ", "") 
  
df['Distance'] = df['Distance'].apply(lambda x: remove_space(x))
df['Model'] = df['Model'].apply(lambda x: int(x))
df['Distance'] = df['Distance'].apply(lambda x: int(x))

# Creating Age feature of bikes
def Age(x):
  if ((x) != 2019):
    return (2019 - x)
  elif ((x) == 2019):
    return (0.1)

import multiprocessing
import tqdm
import concurrent.futures
num_processes = multiprocessing.cpu_count()
with concurrent.futures.ProcessPoolExecutor(num_processes) as pool:
  df['Age']=list(tqdm.tqdm(pool.map(Age, df['Model'],
                                           chunksize=10), total=df.shape[0]))

# Creating feature of distance covered w.r.t age of bike
def feature(x,y):
  if ((x) != 2019):
    return (y/(2019 - x))
  elif ((x) == 2019):
    return (y/(0.1))

num_processes = multiprocessing.cpu_count()
with concurrent.futures.ProcessPoolExecutor(num_processes) as pool:
  df['Dist_year']=list(tqdm.tqdm(pool.map(feature, df['Model'], df['Distance'],
                                           chunksize=10), total=df.shape[0]))

df['Price'] = df['Price'].apply(lambda x: str(x))
df['Price'] = df['Price'].apply(lambda x: re.sub('[^0-9]+', ' ',x))
df['Price'] = df['Price'].apply(lambda x: remove_space(x))
df['Price'] = df['Price'].apply(lambda x: int(x))
df['Owner'] = df['Owner'].apply(lambda x: x.replace(" ", ""))
df['Owner'] = df['Owner'].apply(lambda x: re.sub('[^0-9]+', ' ',x))
df['Owner'] = df['Owner'].apply(lambda x: remove_space(x))
df['Owner'].unique()
df.drop(columns=[' Registration year ','Bike registered at ','   Registration no. '],inplace=True)
df['Owner'].replace({'1':1 ,'2':2 ,'3':3,'4':4 ,'5':5}, inplace=True)
df['Seller'] = df['Seller'].apply(lambda x: x.replace(" ", ""))
df['Seller'].replace(to_replace='NA', value=0, regex=True, inplace=True)
df['Seller'].replace({'Individual': 1, 'Dealer': 2},inplace=True)
print(df.head())

# printing out Data profile
# profile = pp.ProfileReport(df)
# profile.to_file(outputfile="data profiling.html")


df1 = df[df['Owner']==1]
df1.drop(columns=['Colour'],inplace=True)
df1.drop(columns=['Profile Id'],inplace=True)

# Removing the outliers from data
indexNames = df1[df1['Distance'] < 20].index
df1.drop(indexNames,inplace=True)
indexNames1 = df1[df1['Price'] > 200000].index
df1.drop(indexNames1,inplace=True)

df1.drop(columns=['Location'],inplace = True)

# plotting some features
plt.figure(figsize = (10,5))
sns.distplot(df1["Distance"])
plt.figure(figsize = (10,5))
sns.distplot(df1["Price"])

print(df1.head())
df1.shape

# Creating a validation set
val = df1.iloc[5001:]
val.head()

df1_1 = df1.iloc[:5000]
df1_1.head()

t1=df1_1.drop(columns=['Bike Name','Price'], axis=1, inplace=False)
X1=t1
y1=df1_1['Price']

import math
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Splitting the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.33, random_state=42)

# Training the Linear Regression Model
reg = LinearRegression()
reg.fit(X_train, y_train)
pred = reg.predict(X_test)

print('MAE',mean_absolute_error(y_test,pred))
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print('MAPE',mean_absolute_percentage_error(y_test,pred))
math.sqrt(mean_absolute_error(y_test,pred))

# Hyper-Parameter Tunning for Random Forest Model using GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

rf =RandomForestRegressor()
parameters = {'max_depth': [5, 6, 7, 8, 9, 10],
              'min_samples_split': [4,10,15,20,25],
              'max_leaf_nodes' : [10,20,25,30],
              'min_weight_fraction_leaf' : [0,0.01,0.1,0.3,0.5],
              'n_estimators': [500,120,200]}
rf_grid = GridSearchCV(rf,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        scoring='neg_mean_absolute_error',
                        verbose=True)
rf_grid.fit(X_train,y_train)
print(rf_grid.best_score_)
print(rf_grid.best_params_)

# Now Training the model with best Parameters
from sklearn.ensemble import RandomForestRegressor
rf1 =RandomForestRegressor(n_estimators=120, random_state=3, max_depth=5,
                                        min_weight_fraction_leaf=0.1,max_leaf_nodes = 10,
                                       min_samples_split = 25)
rf1.fit(X_train,y_train)
pred_rf1 = rf1.predict(X_test)
math.sqrt(mean_absolute_error(y_test,pred_rf1))
print('MAE ',mean_absolute_error(y_test,pred_rf1))
print('MAPE ',mean_absolute_percentage_error(y_test,pred_rf1))

from sklearn.ensemble import RandomForestRegressor
rf2 =RandomForestRegressor(n_estimators=150, random_state=3, max_depth=10,
                                        min_weight_fraction_leaf=0,max_leaf_nodes = 20,
                                       min_samples_split = 30)
rf2.fit(X_train,y_train)
pred_rf2 = rf2.predict(X_test)
math.sqrt(mean_absolute_error(y_test,pred_rf2))
('MAE ',mean_absolute_error(y_test,pred_rf2))
print('MAPE ',mean_absolute_percentage_error(y_test,pred_rf2))


# Hyper-Parameter Tunning for XGBoost Model using GridSearchCV
from sklearn.model_selection import GridSearchCV
import xgboost

xgb1 = xgboost.XGBRegressor()
parameters = {'objective':['reg:linear'],
              'learning_rate':np.arange(0.01, 1.0, 0.05)  ,                             
              'max_depth': [ 6, 7, 8, 9, 10,12],
              'min_child_weight': [4,10,15,20],
              'subsample': [0.01,0.2,0.7],
              'colsample_bytree': [0.1,0.5,0.7],
              'n_estimators': [500,120,200]}
xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        scoring='neg_mean_absolute_error',
                        verbose=True)
xgb_grid.fit(X_train,y_train)
print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

# training the model with best parameters
import xgboost                                     
xgb1 = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.01,gamma=0, subsample=0.2,
                           colsample_bytree=0.5, max_depth=6, min_child_weight=20,objective= 'reg:linear')
xgb1.fit(X_train,y_train)                         
pred_xgb1 = xgb1.predict(X_test)
math.sqrt(mean_absolute_error(y_test,pred_xgb1))
print('MAE ',mean_absolute_error(y_test,pred_xgb1))
print('MAPE ',mean_absolute_percentage_error(y_test,pred_xgb1))

import xgboost                                     
xgb2 = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.01,gamma=0, subsample=0.7,
                           colsample_bytree=0.7, max_depth=12, min_child_weight=4,objective= 'reg:linear')
xgb2.fit(X_train,y_train)                         
pred_xgb2 = xgb2.predict(X_test)
math.sqrt(mean_absolute_error(y_test,pred_xgb2))
print('MAE ',mean_absolute_error(y_test,pred_xgb2))
print('MAPE ',mean_absolute_percentage_error(y_test,pred_xgb2))


# Validation with the best model
val.reset_index(inplace = True)
val.drop(columns=['index'],inplace = True)
print(val.head(4))
val1 = val.drop(columns=['Price','Bike Name'],inplace = False)
val1.head(2)
pred_xgb2 = xgb2.predict(val1)
t = pd.DataFrame(pred_xgb2)
val=pd.merge(val, t, left_index=True, right_index=True)
val.head()
val.rename(index=str, columns={0: 'Predicted Price'},inplace=True)
val.to_csv('prediction_regression_XGBoost.csv')

# Neural Network
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
from keras.layers import Dropout

# Function for multilayer neural network model
def create_model():

    model = Sequential()
    model.add(Dense(30, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer ='adam', loss = 'mean_squared_error', 
              metrics = [metrics.mae])
    return model

# Training the model on data
model = create_model()
model.summary()
history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=10)
pred_nn = model.predict(X_test)
print('MAE ',mean_absolute_error(y_test,pred_nn))
print('MAPE ',mean_absolute_percentage_error(y_test,pred_nn))

# for accuracy
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# # for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# neural network Hyper-Parameter tunning
def create_model(optimizer='rmsprop', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(20, input_dim=X_train.shape[1]))  # 10, input_dim=X_train.shape[1]
    model.add(Activation('relu')) 
    model.add(Dropout(0.2))  
    model.add(Dense(10,init=init))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,init=init))
    #model.add(Activation('softmax')) 
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=["accuracy"]) 
    return model

from keras.layers import Activation, Dense  
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
model = KerasClassifier(build_fn=create_model)
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = np.array([100, 150, 200, 250, 300])
batches = np.array([5, 10, 20, 30])
param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init)
grid = GridSearchCV(model, param_grid, cv = 3)
grid_result = grid.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

def create_model1(optimizer='adam', init='uniform'):
    model1 = Sequential()
    model1.add(Dense(10, input_dim=X_train.shape[1]))  # 10, input_dim=X_train.shape[1]
    model1.add(Activation('relu')) 
    model1.add(Dropout(0.2))  
    model1.add(Dense(10,init=init))
    model1.add(Activation('relu'))
    model1.add(Dropout(0.2))
    model1.add(Dense(1,init=init))
    #model.add(Activation('softmax')) 
    model1.compile(loss='mean_squared_error', optimizer=optimizer, metrics=["accuracy"]) 
    return model1

model1 = create_model1()
model1.summary()
history = model1.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=50, batch_size=5)
pred_nn1 = model1.predict(X_test)
math.sqrt(mean_absolute_error(y_test,pred_nn1))
print('MAE ',mean_absolute_error(y_test,pred_nn1))
print('MAPE ',mean_absolute_percentage_error(y_test,pred_nn1))