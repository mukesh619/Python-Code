import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
#%%
for direc,_,filenames in os.walk('bigquery-geotab-intersection-congestion'):
    for filename in filenames:
        print(os.path.join(direc,filename))
#%%

df_train=pd.read_csv(r"bigquery-geotab-intersection-congestion\train.csv")

df_train.isna().sum()

df_train.info()

df_train['EntryStreetName'].fillna(df_train['EntryStreetName'].mode()[0],inplace=True)
df_train['ExitStreetName'].fillna(df_train['ExitStreetName'].mode()[0],inplace=True)
df_train.isna().sum()
