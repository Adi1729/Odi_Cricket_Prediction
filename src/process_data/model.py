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

s= 5
s=s+1
data = pd.read_csv(r'../Processed/model_data_v4.csv')
#form_index_features = ['T1_Batsman1_form_index','T1_Batsman2_form_index',
#         'T1_Batsman3_form_index','T1_Batsman4_form_index',
#         'T1_Batsman5_form_index' , 
#         'T2_Batsman1_form_index','T2_Batsman2_form_index',
#         'T2_Batsman3_form_index','T2_Batsman4_form_index' ,
#         'T2_Batsman5_form_index'  
#         ]
#
#data= data.drop(columns = form_index_features)
#form = pd.read_csv(r'../Processed/form_index_batsman_10.csv')
#data  = data.merge(form, on = 'match_id',how = 'left')



data=data.replace([np.inf, -np.inf], 1)
data= data.applymap(lambda x: 0 if pd.isnull(x) else x)

data= data[data.Winner != 'no result']
data =data[data.Winner !='tied']
data['dep_var']=data[['Team 1','Team 2','Winner']].apply(lambda x: 1 if x[2]==x[0] else 0, axis=1)
data['Margin'] = data['Margin'].astype('str')

data['Run_Margin'] = data['Margin'].apply(lambda x:re.findall('\d+',x)[0] if 'run' in x.lower()  else 100).astype(int)
data['Wic_Margin'] = data['Margin'].apply(lambda x:re.findall('\d+',x)[0] if 'wicket' in x.lower()  else 10).astype(int)

data[['Wic_Margin','Run_Margin','Margin']]


data=data[data.odi_id>1900]
multiplier = 'pitch_impact'

data.columns.tolist()
#data=data[:-1]
bat =['T1_Batsman1_form_index','T1_Batsman2_form_index','T1_Batsman3_form_index','T1_Batsman4_form_index','T1_Batsman5_form_index',
                         'T2_Batsman1_form_index','T2_Batsman2_form_index','T2_Batsman3_form_index','T2_Batsman4_form_index','T2_Batsman5_form_index']
rej_col = ['Margin','T1_WR','T2_WR']
cols = [col for col in data.columns.tolist() if col.lower().endswith('bat_avg') or col.lower().endswith('bat_sr') ]

# t1p1 * t2p1
f1 = [col for col in data.columns.tolist() if col.lower().endswith('bowl_avg')]
f2 = [col for col in data.columns.tolist() if col.lower().endswith('bowl_sr')]


f3 = [col for col in data.columns.tolist() if col.lower().endswith('bat_avg')]
f4 = [col for col in data.columns.tolist() if col.lower().endswith('bat_sr')]


f = f1+f2 +f3+f4

for idx in range(len(f1)):
    data['%s_%s'%(f1[idx],f2[idx])] = data[f1[idx]] * data[f2[idx]]

#t1 = [col for col in data.columns.tolist() if 't1' in col.lower() and 'wr' not in col.lower()]
#t2 = [col for col in data.columns.tolist() if 't2' in col.lower() and 'wr' not in col.lower()]     
##     
##
#for idx in range(len(t1)):
#    data['rel_%s_%s'%(t1[idx],t2[idx])] = data[[t1[idx],t2[idx]]].apply(lambda x : 0 if (x[1]==0) else (x[0]/x[1]),axis=1)
#
    
#
#bat = [col for col in data.columns.tolist() if 'bat' in col.lower() and 'strength' not in col.lower() 
#and 'impact' not in col.lower() and 'form' not in col.lower()  and col not in f3  + f4]
#
#
#for col in bat:
#    data['pitch_%s'%col]  = data[col] * data[multiplier] 


#form_bat_t1 = [col for col in data.columns.tolist() if 'form' in col.lower() and 't1' in col.lower() and 'bat' in col.lower()]
#form_bat_t2 = [col for col in data.columns.tolist() if 'form' in col.lower() and 't2' in col.lower() and 'bat' in col.lower()]
#
#
#form_bowl_t1 = [col for col in data.columns.tolist() if 'form' in col.lower() and 't1' in col.lower() and 'bowl' in col.lower()]
#form_bowl_t2 = [col for col in data.columns.tolist() if 'form' in col.lower() and 't2' in col.lower() and 'bowl' in col.lower()]
#
#data['form_avg_bat_t1_t2'] = data[form_bat_t1].sum(axis=1) / data[form_bat_t2].sum(axis=1)
#
#data['form_avg_bowl_t1_t2'] = data[form_bowl_t2].sum(axis=1) / data[form_bowl_t2].sum(axis=1)


#
#for col in form:
#    data['sq_%s'%col]  = np.square(data[col])  
data_oov = data[data['odi_id'] >= 4143]
rej_col=[]
rej_col = rej_col  + f 
#rej_col = rej_col + f3 + f4 
#rej_col = rej_col + f3 + f4 + t1 + t2


#rej_col = rej_col + cols + f1 + f2 + t1 + t2 +
#rej_col =  f1 + f2

#rej_col=[]
#rej_col = rej_col + bat + f1 + f2 + [col for col in data.columns.tolist() if 'ranking' in col.lower()] + ['T1_WR','T2_WR','pitch_impact'] 

#rej_col = rej_col + [col for col in data.columns.tolist() if 'ranking_bat' in col.lower()]


data= data.drop(columns= rej_col,axis=1)

data= data.drop(columns= ['T1_WR','T2_WR','pitch_impact','T1_ranking_bat','T2_ranking_bat','T2_ranking_bowl',\
                          'T1_ranking_bowl'],axis=1)


data= data.drop(columns= ['T1_P1_Bat_Avg_SR'],axis=1)
data= data.drop(columns= ['T1_P4_Bat_Avg_SR'],axis=1)


 
#data= data.drop(columns= ['form_avg_bat_t1_t2','form_avg_bowl_t1_t2'],axis=1)

#null_df= pd.DataFrame(data.isnull().sum())


#training and testing sets
data_train= data[data.odi_id<3900]
data_test= data[(data.odi_id>=3900) & (data.odi_id<4143)].drop(columns = ['Run_Margin','Wic_Margin'], axis=1)
data_oov= data[data.odi_id>=4143].drop(columns = ['Run_Margin','Wic_Margin'], axis=1)

#data_train = data_train[(data_train.Run_Margin >= 50)]
#data_train = data_train[(data_train.Wic_Margin >=3)].drop(columns = ['Run_Margin','Wic_Margin'], axis=1)
#
data_train = data_train.drop(columns = ['Run_Margin','Wic_Margin'], axis=1)

corr_matrix=data_train.corr()
corr_matrix.to_csv(r'../Processed/corr.csv')
#max_df= pd.DataFrame(df.max())


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
pca = PCA(n_components= 30)
pca.explained_variance_ratio_
pca.explained_variance_ratio_
pca.fit(X_train)

from sklearn.preprocessing import scale

X_train = scale(X_train)

X_test = scale(X_test)

X_train = pca.fit_transform(X_train)

X_test = pca.fit_transform(X_test)

###########################Random Forest##################################
def rf_pred(xtrain,ytrain,xtest):
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=2, min_samples_leaf=50) 
    rf.fit(xtrain,ytrain)
    FI= pd.DataFrame(data={'features' :xtrain.columns.tolist(),'score':rf.feature_importances_})
    pred_y = rf.predict_proba(xtest)
    
    return pred_y[:,1],FI

####Cross Validation #####
def cross_validation(X_train,y_train):
    X=X_train.reset_index().drop('index',axis=1)
    y=y_train.reset_index().drop('index',axis=1)
    scores=[]
    cv = KFold(n_splits=8, random_state=42, shuffle=False)

    for train_index, test_index in cv.split(X):
        print("Train Index: ", train_index, "\n")
        print("Test Index: ", test_index)
    
        X_train, X_test, y_train, y_test = X[X.index.isin(train_index)], X[X.index.isin(test_index)], y[y.index.isin(train_index)], y[y.index.isin(test_index)]
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=0, min_samples_leaf=50)
        model=rf.fit(X_train,y_train)
        scores.append(model.score(X_test, y_test))
    acc= sum(scores)/len(scores)
    return acc,scores
acc_cv,scores= cross_validation(X_train,y_train)
print(acc_cv)


y_proba_train_rf,feature_importance= rf_pred(X_train,y_train,X_train)
y_proba_test_rf,feature_importance= rf_pred(X_train,y_train,X_test)

#rfe = RFE(RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=2, min_samples_leaf=50) , 20)
#rfe = rfe.fit(X_train,y_train)
#a= rfe.predict(X_test)
#len(rfe.support_)
#rfe.ranking_



#feature = pd.DataFrame(data={'fetaues':X_train.columns.tolist(),
#                   'rank':rfe.ranking_.tolist()}).sort_values('rank')
#sel= feature['fetaues'][:30].tolist()
#feature_importance = feature_importance.reset_index()
y_pred_train_rf = [ 1 if y1>0.50 else 0 for y1 in y_proba_train_rf]
y_pred_test_rf = [ 1 if y1>0.50 else 0 for y1 in y_proba_test_rf]

#confusion matrix
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

data_test['pred'] = y_pred_test_rf
data_test['Winner_pred'] = data_test[['Team 2','Team 1','pred']].apply(lambda x : x[x[2]],axis=1)
data_test['wrong'] = data_test[['pred','dep_var']].apply(lambda x : int(x[0]==x[1]),axis=1)
data_test['wrong'] = data_test[['pred','dep_var']].apply(lambda x : int(x[0]==x[1]),axis=1)

data_test['win_rank'] = ((data_test['T1_Rel_bat_strength']+data_test['T1_Rel_bowl_strength'])-
(data_test['T2_Rel_bat_strength']+data_test['T2_Rel_bowl_strength']))

data_test['win_rank_class'] = data_test['win_rank'].apply(lambda x: 1 if x > 0 else 0)

data_test['win_rank_class_right'] = data_test[['win_rank_class','dep_var']].apply(lambda x : int(x[0]==x[1]),axis=1)


data_test.to_csv(r'../Processed/predicted_all.csv',index= False)


##### Logistic Regression #####
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression() 
model1=logisticRegr.fit(X_train, y_train)

predictions = model1.predict(pca.fit_transform(X_test))

proba_lr = model1.predict_proba(X_test)

# Use score method to get accuracy of model
score = model1.score(X_test, y_test)
print(score)

from sklearn import metrics
cm = metrics.confusion_matrix(y_test, predictions)

precision = cm[1,1]/sum(cm[:,1]) * 100
recall = cm[1,1]/sum(cm[1,])  * 100
#####

ensem = [rf + lr  for rf,lr in zip(y_proba_test_rf,proba_lr[:,1])]
ensem_class = [1 if val > 1 else 0 for val in ensem ]

pr= pd.DataFrame({'pred':ensem_class,  'act':y_test['dep_var'].tolist(),'rf' : y_proba_test_rf,
                  'lr': proba_lr[:,1]})
corr_matrix=pr.corr()

sum(pr['pred'] == pr['act'])/356


sum([ print(x==y)  for x,y in zip(ensem_class,y_test['dep_var'].astype(int).tolist())])/len(y_test)


###### Xg Boost #######

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
param_test1 = {
 'max_depth':[1,2,3,4,5,6,7],
 'min_child_weight':[1,3,5]
}

from sklearn.grid_search import GridSearchCV   #Perforing grid search
gsearch1 = GridSearchCV(estimator = xgboost.XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X_train,y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_



predictors = [x for x in train.columns if x not in [target, IDcol]]
xgb1 = xgboost.XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, X_train, X_train.columns)


train_X, train_y = X_train.values, y_train.values
test_X, test_y = X_test.values, y_test.values


FM = FinalModel(model_params=xgb_params)
FM.fit_model(train_X, train_y)
train_y_proba = FM.transform_model(train_X)
train_y_pred = [ 1 if y1>0.5 else 0 for y1 in train_y_proba]
from sklearn import metrics
cm = metrics.confusion_matrix(train_y, train_y_pred)
acc = (cm[0,0]+cm[1,1])/2808

test_y_proba = FM.transform_model(test_X)
test_y_pred = [ 1 if y1>0.5 else 0 for y1 in test_y_proba]
from sklearn import metrics
cm = metrics.confusion_matrix(test_y, test_y_pred)
acc = (cm[0,0]+cm[1,1])/703



print('precision = %f,  recall = %f'%(precision,recall))

model_summary = pd.DataFrame(data={'Features': x_vars, 
                   'Coeff' : model.coef_.T.tolist()})

model_summary.to_csv('HCR_result.csv')
# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Feature Scaling

# Any results you write to the current directory are saved as output.

#For plotting
from IPython.display import display, HTML

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);