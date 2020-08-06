import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#matplotlib qt   plt.show() ke jagah and plot ka separtate tab khulta hai
#%%
df=pd.read_csv(r'C:\Users\mukes\Downloads\fatal-police-shootings-data.csv')
df['date']=pd.to_datetime(df['date'])
df['month']=df['date'].dt.month
df['year']=df['date'].dt.year
df['no. of records']=df['manner_of_death'].apply(len)

df.info()

df.drop(columns=['Unnamed: 14'],inplace=True)

df.isna().sum()

#%%                 Treating missing value
df['armed'].fillna(df['armed'].mode()[0],inplace=True)
df['age'].fillna(df['age'].mode()[0],inplace=True)
df['gender'].fillna(df['gender'].mode()[0],inplace=True)
df['race'].fillna(df['race'].mode()[0],inplace=True)
df['flee'].fillna(df['flee'].mode()[0],inplace=True)

#%%                 What about rate of shootings
df['manner_of_death'].value_counts()
gun_shot=df[df['manner_of_death']=='shot']

rate_of_shooting=len(gun_shot['manner_of_death'])/(len(df['manner_of_death']))*100
print(rate_of_shooting)

#rate_of_shooting=df['manner_of_death'].value_counts(normalize=True)*100
#print(rate_of_shooting)
#%%                               Yearswise Shooting
yearwise_shooting=df.groupby(['year'])['manner_of_death'].count()
yearwise_shooting_arr=np.array(yearwise_shooting)
print(yearwise_shooting)

yearwise_shooting.plot(kind='bar')
plt.xlabel("Years")
plt.ylabel("Number of Death")

#%%                Which states have the most kills
statewise_most_kills=df.groupby(['state'])['manner_of_death'].count().sort_values(ascending=False)[0:10]
statewise_most_kills.plot(kind='bar')
plt.xlabel("States")
plt.ylabel("Number of Death")
plt.legend("Citywise Death by Police")
#            Which City has most kills
citywise_most_kills=df.groupby(['city'])['manner_of_death'].count().sort_values(ascending=False)[0:10]
citywise_most_kills.plot(kind='bar')
plt.xlabel("City")
plt.ylabel("Number of Death")
plt.legend("Citywise Death by Police")

#%%         rate of killings relative to race and age
kill_race_age=df.groupby(['race','manner_of_death'])['age'].mean()
print(kill_race_age)