import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xg
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
#%%
for direc,_,filenames in os.walk('pubg-finish-placement-prediction1'):
    for filename in filenames:
        print(os.path.join(direc,filename))
#%%
df_train=pd.read_csv(r"pubg-finish-placement-prediction1\train_V2.csv")
df_test=pd.read_csv(r"pubg-finish-placement-prediction1\test_V2.csv")
df_sub=pd.read_csv(r"pubg-finish-placement-prediction1\sample_submission_V2.csv")
#%%
df_train.isna().sum()
df_train.winPlacePerc.value_counts()

df_train.dropna(inplace=True)

df_train['matchType'].value_counts()

df_train['matchType']=df_train['matchType'].map({'squad-fpp':0,'duo-fpp':1,'squad':2,'solo-fpp':3,'duo':4,'solo':5,'normal-squad-fpp':6,'crashfpp':7,'normal-duo-fpp':8,'flaretpp':9,'normal-solo-fpp':10,'flarefpp':11,'normal-squad':12,'crashtpp':13,'normal-solo':14,'normal-duo':15})

df_train.info()
#%%
X=df_train.drop(['Id','groupId','matchId'],axis=1)
y=df_train['winPlacePerc']

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

feat_cols = ['feature'+str(i) for i in range(X_scaled.shape[1])]
normalised_breast = pd.DataFrame(X_scaled,columns=feat_cols)


pca=PCA().fit(X)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.show()


pca1=PCA(.9)
pca1.fit(X)
pca1.n_components_

pca2=PCA(n_components=3)
pca2.fit(X_scaled)
print(pca2.explained_variance_)
X_pca = pca2.transform(X_scaled)

pca_train = pd.DataFrame(data = X_pca, columns = ['principal component 1', 'principal component 2','principal component 3'])
print('Explained variation per principal component: {}'.format(pca2.explained_variance_ratio_))
#%%
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=.2,random_state=42)
#%%
xg_without_tuning=xg.XGBRFRegressor(objective='reg:linear',n_estimators=10,seed=123)
xg_without_tuning.fit(X_train,y_train)
pred=xg_without_tuning.predict(X_test)
MAE=(mean_absolute_error(y_test,pred))
print("MAE: %f" %(MAE))

DM_train=xg.DMatrix(X_train,y_train)
DM_test=xg.DMatrix(X_test,y_test)
params={'booster':'gblinear',"objective":"reg:linear"}
xg_reg=xg.train(params=params,dtrain=DM_train,num_boost_round=5)
pred1=xg_reg.predict(DM_test)
MAE1=(mean_absolute_error(y_test,pred1))
print("RMSE1: %f" %MAE1)