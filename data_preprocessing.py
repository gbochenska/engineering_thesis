import pandas as pd 
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure



#Load dataset
df = pd.read_csv('datasets/heart_2020_cleaned.csv')
# print(df.head())

# print(df.shape)

#Check the data info
# print(df.info())

#Check the null values
df.isnull()

#descriptive overview
df.describe()


#heatmap
# data1 = df[df['Sex'] == 'Female'][['Smoking','Age_Category','HeartDisease']]
# data1 = data1.pivot('Smoking','Age_Category','HeartDisease')
# data1.head(3)
# sns.set(rc={'figure.figsize':(10,8)})
# ax = sns.heatmap(df)




# objects -> float
# Check the data types of the columns
# print(df.dtypes)

def data_numeric(df):
    age = sorted(df["AgeCategory"].unique())
    # age = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older']
    race = sorted(df['Race'].unique())
    # race = ['American Indian/Alaskan Native', 'Asian', 'Black', 'Hispanic', 'Other', 'White']
    health = ['Poor', 'Fair', 'Good', 'Very good', 'Excellent']
    #change
    for o in df.columns:
        if df[o].dtype != 'float64':
            if df[o][0] in ['Yes', 'No']:
                df[o] = df[o].map({'Yes': 1.0, 'No': 0.0})
            elif df[o][0] in ['Male', 'Female']:
                df[o] = df[o].map({'Male': 1.0, 'Female': 0.0})
            elif df[o][0] in age:
                df[o] = df[o].map({'18-24':1.0, '25-29':2.0, '30-34':3.0, '35-39':4.0, '40-44':5.0, '45-49':6.0, '50-54':7.0, '55-59':8.0, '60-64':9.0, '65-69':10.0, '70-74':11.0, '75-79':12.0, '80 or older':13.0})
            elif df[o][0] in race:
                    # one hot coding is better?
                df[o] = df[o].map({'American Indian/Alaskan Native':1.0, 'Asian':2.0, 'Black':3.0, 'Hispanic':4.0, 'Other':5.0, 'White':6.0})
            elif df[o][0] in health:
                df[o] = df[o].map({'Poor':1.0, 'Fair':2.0, 'Good':3.0, 'Very good':4.0, 'Excellent':5.0})
            else:
                pass
        df.dropna(inplace=True)
    return df

df = data_numeric(df)
print(df.head())

print(df["HeartDisease"].value_counts())

#correlation matrix and heatmap
cor_matrix = df.corr().abs()
print(cor_matrix)

fig, ax = plt.subplots(figsize=(18,18))
dataplot = sns.heatmap(df.corr().abs(), cmap="YlGnBu", annot=True, annot_kws={'size': 10}, ax=ax)
plt.show()

print(df.head())
