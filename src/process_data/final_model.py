import pandas as pd
import numpy as np
import sys
import os
from importlib import reload
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from sklearn.decomposition import PCA

os.chdir(r'./Cricket Clairvoyant/src')

#Import Data
data = pd.read_csv(r'../Processed/model_data_v4.csv')

data=data.replace([np.inf, -np.inf], 1)
data= data.applymap(lambda x: 0 if pd.isnull(x) else x)

data= data[data.Winner != 'no result']
data =data[data.Winner !='tied']
data['dep_var']=data[['Team 1','Team 2','Winner']].apply(lambda x: 1 if x[2]==x[0] else 0, axis=1)
data['Margin'] = data['Margin'].astype('str')

data=data[data.odi_id>1800]

rej_col = ['Margin','T1_WR','T2_WR']

# t1p1 * t2p1
f1 = [col for col in data.columns.tolist() if col.lower().endswith('bowl_avg')]
f2 = [col for col in data.columns.tolist() if col.lower().endswith('bowl_sr')]


f3 = [col for col in data.columns.tolist() if col.lower().endswith('bat_avg')]
f4 = [col for col in data.columns.tolist() if col.lower().endswith('bat_sr')]


f = f1+f2 +f3+f4

for idx in range(len(f1)):
    data['%s_%s'%(f1[idx],f2[idx])] = data[f1[idx]] * data[f2[idx]]

data_oov = data[data['odi_id'] >= 4143]
rej_col=[]
rej_col = rej_col  + f 

data= data.drop(columns= rej_col,axis=1)

data= data.drop(columns= ['pitch_impact','T1_ranking_bat','T2_ranking_bat','T2_ranking_bowl',\
                          'T1_ranking_bowl'],axis=1)


#training and testing sets
data_train= data[data.odi_id<3900]
data_test= data[(data.odi_id>=3900) & (data.odi_id<4143)]
data_oov= data[data.odi_id>=4143]

corr_matrix=data_train.corr()
corr_matrix.to_csv(r'../Processed/corr.csv')

df_train=data_train.iloc[:,9::]
df_test= data_test.iloc[:,9::]
df_oov = data_oov.iloc[:,9::]

X_train= df_train.iloc[:,df_train.columns !='dep_var']
y_train= df_train.iloc[:,df_train.columns =='dep_var']
X_test= df_test.loc[:,df_train.columns !='dep_var']
y_test= df_test.loc[:,df_train.columns =='dep_var']
X_oov= df_oov.loc[:,df_train.columns !='dep_var']

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_oov.shape)

###########################Random Forest##################################
import random
random.seed(42)
y_proba_train_rf,feature_importance, rf= rf_pred(X_train,y_train,X_train)
y_proba_test_rf,feature_importance, rf= rf_pred(X_train,y_train,X_test)

y_pred_train_rf = [ 1 if y1>0.55 else 0 for y1 in y_proba_train_rf]
y_pred_test_rf = [ 1 if y1>0.55 else 0 for y1 in y_proba_test_rf]

rf_model_pkl = open('rf_model.pkl', 'rb')
rf_model_pkl = pickle.load(rf_model_pkl)
pred_y = rf.predict_proba(X_test)[:,1]

#World cup prediction
pred_wc = rf.predict_proba(X_oov)[:,1]

# training accuracy = 0.76
# test accuracy  = 0.70
# threshold = 0.55

#training
from sklearn import metrics
cm_train = metrics.confusion_matrix(y_train, y_pred_train_rf)
acc_train = (cm_train[0,0]+cm_train[1,1])/(cm_train[0,0]+cm_train[1,1]+cm_train[1,0]+cm_train[0,1])
print("acc_train",acc_train)

#test
from sklearn import metrics
cm_test = metrics.confusion_matrix(y_test, y_pred_test_rf)
acc_test = (cm_test[0,0]+cm_test[1,1])/(cm_test[0,0]+cm_test[1,1]+cm_test[1,0]+cm_test[0,1])
print("acc_test",acc_test)
data.to_csv('../Processed/Model/data_iter_%d_%d.csv'%(s,acc_test*100),index= False)

##########################################################################

data_oov['pred'] = pred_wc
data_oov['pred_class'] =  [ 1 if y1>0.55 else 0 for y1 in pred_wc]

data_oov['Winner_pred'] = data_oov[['Team 2','Team 1','pred_class']].apply(lambda x : x[x[2]],axis=1)

data_oov['win_rank'] = ((data_oov['T1_Rel_bat_strength']+data_oov['T1_Rel_bowl_strength'])-
(data_oov['T2_Rel_bat_strength']+data_oov['T2_Rel_bowl_strength']))
data_oov['win_rank_class'] = data_oov['win_rank'].apply(lambda x: 1 if x > 0 else 0)
data_oov['win_rank_class_right'] = data_oov[['win_rank_class','dep_var']].apply(lambda x : int(x[0]==x[1]),axis=1)


data_oov.to_csv(r'../Processed/Model/predicted_wc.csv',index= False)

