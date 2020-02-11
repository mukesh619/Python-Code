import pandas as pd
from urllib.request import urlretrieve
import matplotlib.pyplot as plt
from scipy.stats import kruskal,chi2_contingency

url="https://raw.githubusercontent.com/SergeyShk/DataCamp-Projects/master/Python/Which%20Debts%20Are%20Worth%20the%20Bank's%20Effort/datasets/bank_data.csv"
urlretrieve(url,"bank_debt.csv")
df_train=pd.read_csv("bank_debt.csv")

df_train.head()

# checking age is not factor for extra charging of customer rocovery by graph

plt.scatter(x=df_train['expected_recovery_amount'],y=df_train['age'])
plt.xlim(0,2000)
plt.ylim(0,60)
plt.xlabel("expected_recovery_amount")
plt.ylabel("age")
plt.show()

 # checking by statitical test
era_900_1000=df_train.loc[(df_train['expected_recovery_amount']<=1100) & 
                     (df_train['expected_recovery_amount']>=900)]

recov_strategy=era_900_1000.groupby(['recovery_strategy'])
recov_strategy.age.describe().unstack()

level_0_strategy=era_900_1000.loc[df_train['recovery_strategy']=='Level 0 Recovery']['age']
level_1_strategy=era_900_1000.loc[df_train['recovery_strategy']=='Level 1 Recovery']['age']

kruskal(level_0_strategy,level_1_strategy)


# Number of customers in each category
crosstab = pd.crosstab(df_train.loc[(df_train['expected_recovery_amount']<2000) & 
                              (df_train['expected_recovery_amount']>=0)]['recovery_strategy'], 
                       df_train['sex'])
print(crosstab)

# Chi-square test
chi2_stat, p_val, dof, ex = chi2_contingency(crosstab)
print(p_val)

#seeing any backchodhi in actual vs expected recovery

plt.scatter(x=df_train['expected_recovery_amount'],y=df_train['actual_recovery_amount'])
plt.xlabel("Expeted reco amt")
plt.ylabel("Acyual amt")
plt.show()

plt.scatter(x=df_train['expected_recovery_amount'],y=df_train['actual_recovery_amount'])
plt.xlim(900,1000)
plt.ylim(0,2000)
plt.xlabel("Expeted reco amt")
plt.ylabel("Actual amt")
plt.show()

# Compute average actual recovery amount just below and above the threshold
recov_strategy['actual_recovery_amount'].describe().unstack()

# Perform Kruskal-Wallis test
Level_0_actual = era_900_1000.loc[df_train['recovery_strategy']=='Level 0 Recovery']['actual_recovery_amount']
Level_1_actual = era_900_1000.loc[df_train['recovery_strategy']=='Level 1 Recovery']['actual_recovery_amount']
print(kruskal(Level_0_actual, Level_1_actual))

# Repeat for a smaller range of $950 to $1050
era_950_1050 = df_train.loc[(df_train['expected_recovery_amount']<1050) & 
                      (df_train['expected_recovery_amount']>=950)]
Level_0_actual = era_950_1050.loc[df_train['recovery_strategy']=='Level 0 Recovery']['actual_recovery_amount']
Level_1_actual = era_950_1050.loc[df_train['recovery_strategy']=='Level 1 Recovery']['actual_recovery_amount']
kruskal(Level_0_actual, Level_1_actual)

