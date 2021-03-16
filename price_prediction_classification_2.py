# Importing required Libraries
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
# %matplotlib inline
import seaborn as sns

# Loading data
excel= pd.ExcelFile("Bikewale_Data_Result_Final v2.0 (1).xlsx")
df=excel.parse('Bikewale_Data_Result_Final')
df.drop(columns=['Unnamed: 15','Unnamed: 16','Unnamed: 17','Url','Date Updated'],inplace = True)
df.rename(index=str, columns={"Km/s": "Distance"},inplace=True)
print(df.head())

df.columns
df.shape
df.info()
df.describe()
df.isnull().sum()

test = df[['Model',' Registration year ']].isna()
test['check'] = df['Model']== df[' Registration year ']
test[test['check'] == True]

# Preprocessing
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
df.head()
df.shape

import re
df['Model'] = df['Model'].apply(lambda x: re.sub('[^0-9]+', ' ',x))
df['Distance'] = df['Distance'].apply(lambda x: re.sub('[^0-9]+', ' ',x))
df['Distance'] = df['Distance'].apply(lambda x: x.strip())
def remove_space(string): 
    return string.replace(" ", "") 
  
df['Distance'] = df['Distance'].apply(lambda x: remove_space(x))

# for i in range(0,df.shape[0]):
#   if (type(df.iloc[i][' Registration year ']) == datetime.datetime):
#     df.iloc[i][' Registration year '] = df.iloc[i][' Registration year '].year 
    
# for i in range(0,df.shape[0]):
#   if (type(df.iloc[i][' Registration year ']) != int ):
#     df.iloc[i][' Registration year '] = df.iloc[i][' Registration year '].split('\t')[0]
#     df.iloc[i][' Registration year '] = re.sub('[^0-9]+', ' ',df.iloc[i][' Registration year '])
#     df.iloc[i][' Registration year '] = df.iloc[i][' Registration year '].strip()
    
# df[' Registration year '].replace(to_replace='', value='0', regex=True, inplace=True)

df['Model'] = df['Model'].apply(lambda x: int(x))
df['Distance'] = df['Distance'].apply(lambda x: int(x))
# creating feature 'Age'
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

# profile = pp.ProfileReport(df)
# profile.to_file(outputfile="data profiling.html")

# Analysing for only second hand pricing
df1 = df[df['Owner']==1]
df1.drop(columns=['Colour'],inplace=True)
df1['Price'].value_counts()

df1.drop(columns=['Profile Id'],inplace=True)
df1.drop(columns=['Location'],inplace = True)
print(df1.head())

# Function to divide 'price' feature into various ranges
import math
def ranges(df,n,min_v,max_v):
  #df_new = df[df['Price'] >= min_v & df['Price'] < max_v]
  x =math.ceil((max_v - min_v)/n)
  temp_min = min_v
  for i in range(1,x):
    temp_max = min_v + i*n
    for j in range(0,df.shape[0]):
      if (df['Price'].iloc[j] >= temp_min and df['Price'].iloc[j] < temp_max):
        df['Price Range'].iloc[j] = str(temp_min) + ' - ' + str(temp_max)
    temp_min = temp_max
    print(i)
  return df

# Price range
df3 = df1.loc[(df1['Price'] >= 5000) & (df1['Price'] < 77000)]
df3['Price Range'] = ''
ranges(df3,2500,df3['Price'].min(),df3['Price'].max())

df4 = df1.loc[(df1['Price'] >= 77000) & (df1['Price'] < 150000)]
df4['Price Range'] = ''
ranges(df4,8000,df4['Price'].min(),df4['Price'].max())

df_list = [df3, df4]
df_f = pd.concat(df_list)
df_f.drop_duplicates(inplace=True)
indexNames = df_f[(df_f['Price Range'] == '')].index
df_f.drop(indexNames,inplace=True)
print(df_f.head())

df_f['Price Range'].value_counts()
df_f.to_csv('final_data.csv')

# removing outliers
indexNames = df_f[df_f['Distance'] < 20].index
df_f.drop(indexNames,inplace=True)

# Plotting
plt.figure(figsize = (10,5))
sns.distplot(df_f["Distance"])
plt.figure(figsize = (10,5))
sns.distplot(df_f["Price"])

df_f.head(3)
df_f.drop(columns=['Price'],inplace = True)
df_f.shape

# validation set as 'Test'
test = df_f.iloc[3163:]
test.head()
df1_1 = df_f.iloc[:3162]
df1_1.head()

import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

t1 = df1_1.drop(columns=['Bike Name','Price Range'], inplace=False)
X1 = t1
y1 = df1_1['Price Range']

# using Somte
smt = SMOTE(random_state=42, k_neighbors=1,kind='borderline1')
X_SMOTE, y_SMOTE = smt.fit_sample(X1, y1)
X_train, X_test, y_train, y_test = train_test_split(X_SMOTE, y_SMOTE, test_size=0.33, random_state=42)

# Hyper-Parameter tunning on Random Forest using GridSearchcv
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
rf_c =RandomForestClassifier()
parameters = {'max_depth': [5, 6, 7, 8, 9, 10],
              'min_samples_split': [4,10,15,20,25],
              'max_leaf_nodes' : [10,20,25,30],
              'min_weight_fraction_leaf' : [0,0.01,0.1,0.3,0.5],
              'n_estimators': [500,120,200]}
rf_grid = GridSearchCV(rf_c,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        scoring='f1_weighted',
                        verbose=True)
rf_grid.fit(X_train,y_train)
print(rf_grid.best_score_)
print(rf_grid.best_params_)

# Training with best parameters
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

clf1 = RandomForestClassifier(random_state=0, n_jobs=-1, class_weight='balanced',max_depth=8, max_leaf_nodes=30, min_samples_split= 15, min_weight_fraction_leaf=0)  
model = clf1.fit(X_train, y_train)
predict = model.predict(X_test)
proba = model.predict_proba(X_test)
score = metrics.accuracy_score(y_test, predict)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, predict)
print(metrics.classification_report(y_test, predict))

# Hyper-Parameter tunning on XGBoost using GridSearchcv
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import xgboost

param_test = {
 'max_depth':[4,5,6,10,12],
 'min_child_weight':[4,6,8,10],
 'learning_rate' : [0.001,0.01,0.1],
 'subsample' : [0.6,0.7,0.8]
}    
gsearch = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
                       min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                       nthread=4, scale_pos_weight=1,seed=27), 
                       param_grid = param_test,n_jobs=4,iid=False, cv=3,verbose = True)

train_model4 = gsearch.fit(X_train, y_train)
pred4 = train_model4.predict(X_test)
print(gsearch.best_score_)
print(gsearch.best_params_)


import xgboost                                     
xgb1 = xgb.XGBClassifier( learning_rate=0.1, n_estimators=200,
                        gamma=0, subsample=0.7, colsample_bytree=0.9,
                       nthread=4, scale_pos_weight=1,seed=27,max_depth=10,min_child_weight=3)
xgb1.fit(X_train,y_train)                         
pred_xgb1 = xgb1.predict(X_test)
accuracy_score(y_test, pred_xgb1)
cm = metrics.confusion_matrix(y_test, pred_xgb1)
print(metrics.classification_report(y_test, pred_xgb1))

# Test
x_test = pd.DataFrame(X_test)
x_test.rename(index=str, columns={0: 'Model', 1: 'Distance',2: 'Owner',
                                  3: 'Seller',4:'Insurance', 5:'Age',6:'Dist_year'},inplace=True)
x_test.head()
x_test.to_csv('x_test.csv')
y_test = pd.DataFrame(y_test)
y_test.head()
y_test.to_csv('y_test.csv')
#test_xy = pd.merge(x_test, y_test, left_index=True, right_index=True)

pred = pd.DataFrame(pred_xgb1)
pred.to_csv('pred.csv')

# saving classification report
classification_report = metrics.classification_report(y_test, pred_xgb1,output_dict=True)
class_r = pd.DataFrame(classification_report).transpose()
class_r.to_csv('classification_report.csv')

print(metrics.classification_report(y_test, pred_xgb1))
test.reset_index(inplace=True)
test.head()
test1 = test.drop(columns=['index','Price Range','Bike Name'],inplace=False)
np.array(test1)
test.head()

# test['Price Range'].value_counts()
# l = pd.DataFrame(xgb1.predict(np.array(test1)))
# test_11 = pd.merge(test, l, left_index=True, right_index=True)
# test_11
# test1_t = test['Price Range']
# smt = SMOTE(random_state=42, k_neighbors=1,kind='borderline1')
# X_SMOTE, y_SMOTE = smt.fit_sample(test1, test1_t)
# len(X_SMOTE)
# X_train, X_test, y_train, y_test = train_test_split(X_SMOTE, y_SMOTE, test_size=0.33, random_state=42)
# y_SMOTE[:5]
# xgb1.predict(X_SMOTE)
# test_df = pd.DataFrame(xgb1.predict(X_test))
# test_df.to_csv('predictions_classification.csv')

# performing classification on new data set

# excel= pd.ExcelFile("Bike Model Data.xlsx")
# df=excel.parse('Sheet 1')
# df.drop(columns=['Registration.no.','Url','Registration.year','Bike.registered.at','Colour','Date.Updated','Profile.Id'],inplace = True)
# df.head()
# df.isnull().sum()
# df['Seller'].fillna(0,inplace=True)
# df['Insurance'].fillna(0,inplace=True)
# df.dropna(inplace = True)
# df.shape
# df.info()
# df['Insurance'].unique()
# df['Seller'].replace({'Individual':1 ,'Dealer':2 }, inplace=True)
# df['Insurance'].replace({'ThirdParty':1 ,'Comprehensive':2, 'NoInsurance': 0 }, inplace=True)
# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()
# df['Seller'] = le.fit_transform(df['Seller'])
# df['Insurance'] = le.fit_transform(df['Insurance'])
# df['Make'] = le.fit_transform(df['Make'])
# df['Origin'] = le.fit_transform(df['Origin'])
# df.drop(columns=['Location'],inplace= True)
# df.head()
# plt.figure(figsize = (10,5))
# sns.distplot(df["Price"])
# df['Price'].value_counts()
# indexNames1 = df[(df['Price'] < 1000)].index
# indexNames1
# df.drop(indexNames1,inplace=True)
# df_1 = df.loc[(df['Price'] >= 5000) & (df['Price'] < 80000)]
# df_1['Price Range'] = ''
# print(df_1['Price'].min())
# print(df_1['Price'].max())
# ranges(df_1,4000,5000.0,79000.0)
# df_2 = df.loc[(df['Price'] >= 5000) & (df['Price'] < 80000)]
# df_2['Price Range'] = ''
# print(df_2['Price'].min())
# print(df_2['Price'].max())
# ranges(df_2,4000,5000.0,79000.0)
# indexNames = df_2[(df_2['Price Range'] == '')].index
# indexNames
# df_2.drop(indexNames,inplace=True)
# df_2.head()
# df_3 = df.loc[(df['Price'] >= 80000) & (df['Price'] < 150000)]
# df_3['Price Range'] = ''
# print(df_3['Price'].min())
# print(df_3['Price'].max())
# ranges(df_3,8000,80000.0,149000.0)
# df_list = [df_2, df_3]
# df_f1 = pd.concat(df_list)
# #df_f.drop_duplicates(inplace=True)
# df_f1.head()
# indexNames = df_f1[(df_f1['Price Range'] == '')].index
# indexNames
# df_f1.drop(indexNames,inplace=True)
# df_f1['Price Range'].value_counts()
# df_f1.shape
# test = df_f1.iloc[2001:]
# test.head()
# df_11 = df_f1.iloc[:2000]
# df_11.drop(columns=['Price'],inplace = True)
# df_11.head()
# t11=df_11.drop(columns=['Bike.Name','Price Range'], axis=1, inplace=False)
# X11=t11
# y11=df_11['Price Range']
# X_train, X_test, y_train, y_test = train_test_split(X11, y11, test_size=0.33, random_state=42)
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import metrics
# clf1 = RandomForestClassifier(random_state=0, n_jobs=-1, class_weight='balanced',max_depth=8, max_leaf_nodes=30, min_samples_split= 15, min_weight_fraction_leaf=0)
# model = clf1.fit(X_train, y_train)
# predict = model.predict(X_test)
# proba = model.predict_proba(X_test)
# score = metrics.accuracy_score(y_test, predict)
# print("accuracy:   %0.3f" % score)
# import xgboost
# xgb1 = xgb.XGBClassifier( learning_rate=0.01, n_estimators=200,
#                         gamma=0, subsample=0.7, colsample_bytree=0.9,
#                        nthread=4, scale_pos_weight=1,seed=27,max_depth=12,min_child_weight=4)
# xgb1.fit(X_train,y_train)
# pred_xgb1 = xgb1.predict(X_test)
# accuracy_score(y_test, pred_xgb1)
# print(metrics.classification_report(y_test, pred_xgb1))
# test.reset_index(inplace=True)
# test.head()
# test1 = test.drop(columns=['Bike.Name','Price Range','index'],inplace=False)
# test1.head()
# xgb1.predict(test1)