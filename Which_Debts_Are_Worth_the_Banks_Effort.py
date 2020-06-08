import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal,chi2_contingency
import statsmodels.api as sm
import numpy as np

url="https://raw.githubusercontent.com/SergeyShk/DataCamp-Projects/master/Python/Which%20Debts%20Are%20Worth%20the%20Bank's%20Effort/datasets/bank_data.csv"
df=pd.read_csv(url)

df.head()

plt.scatter(df.expected_recovery_amount,df.age,c='g',s=2)
plt.xlim(0,2000)
plt.ylim(0,60)
plt.xlabel("Expected Recovery Amount")
plt.ylabel("Age")
plt.legend(loc=2)
plt.show()

#Statistical test: age vs. expected recovery amount

recov_amt_900_1100=df.loc[(df['expected_recovery_amount']<1100) & 
                            (df['expected_recovery_amount']>=900)]

recov_strategy=recov_amt_900_1100.groupby(['recovery_strategy'])

recov_strategy['age'].describe().unstack()

# Perform Kruskal-Wallis test 

level_zero_age=recov_amt_900_1100.loc[df['recovery_strategy']=='Level 0 Recovery']['age']
level_one_age=recov_amt_900_1100.loc[df['recovery_strategy']=='Level 1 Recovery']['age']

kruskal(level_zero_age,level_one_age)

# Number of customers in each category

crosstab=pd.crosstab(df.loc[(df['expected_recovery_amount']<2000) & (df['expected_recovery_amount']>=0)]['recovery_strategy'],df['sex'])

print(crosstab)

# Chi-square test
chi2_stat, p_val, dof, ex=chi2_contingency(crosstab)

print(p_val)

# Scatter plot of Actual Recovery Amount vs. Expected Recovery Amount
plt.scatter(df['expected_recovery_amount'],df['actual_recovery_amount'],c='g',s=2)
plt.xlim(900,1100)
plt.ylim(0,2000)
plt.xlabel("Expected_Recovery_Amount")
plt.ylabel("Actual_Recovery_Amount")
plt.legend()
plt.show()

recov_strategy['actual_recovery_amount'].describe().unstack()

level_zero_actual_age=recov_amt_900_1100.loc[df['recovery_strategy']=='Level 0 Recovery']['actual_recovery_amount']
level_one_actual_age=recov_amt_900_1100.loc[df['recovery_strategy']=='Level 1 Recovery']['actual_recovery_amount']

print(kruskal(level_zero_actual_age,level_one_actual_age))

recov_amt_950_1050=df.loc[(df['expected_recovery_amount']<1050) & 
                            (df['expected_recovery_amount']>=950)]

level_zero_actual_age=recov_amt_950_1050.loc[df['recovery_strategy']=='Level 0 Recovery']['actual_recovery_amount']
level_one_actual_age=recov_amt_950_1050.loc[df['recovery_strategy']=='Level 1 Recovery']['actual_recovery_amount']

print(kruskal(level_zero_actual_age,level_one_actual_age))

#Regression modeling: no threshold

X=recov_amt_900_1100['expected_recovery_amount']
y=recov_amt_900_1100['actual_recovery_amount']

X=sm.add_constant(X)

model=sm.OLS(y,X).fit()
pred=model.predict(X)
model.summary()

#Regression modeling: adding true threshold
df['indicator_1000']=np.where(df['expected_recovery_amount']<1000,0,1)

recov_amt_900_1100=df.loc[(df['expected_recovery_amount']<1100) & (df['expected_recovery_amount']>=900)]

X=recov_amt_900_1100['expected_recovery_amount']
y=recov_amt_900_1100['actual_recovery_amount']
X=sm.add_constant(X)

model=sm.OLS(y,X).fit()
model.summary()

recov_amt_950_1050 = df.loc[(df['expected_recovery_amount']<1050) & 
                      (df['expected_recovery_amount']>=950)]

X=recov_amt_950_1050[['expected_recovery_amount','indicator_1000']]
y=recov_amt_950_1050['actual_recovery_amount']
X=sm.add_constant(X)

model=sm.OLS(y,X).fit()
model.summary()