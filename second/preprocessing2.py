import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.preprocessing import StandardScaler, MinMaxScaler

"""Data Preprocessing involves below steps:

Getting the dataset
Importing libraries
Importing datasets
Finding Missing Data
Encoding Categorical Data"""

#Load dataset
df = pd.read_csv('second/datasets/cleaned.csv')

# print(df.head())
# print(df.shape)

#Check the data info
# print(df.head())

#Check the null values
# print(df.isnull())
# df.dropna(inplace = True)
# # print(df.head())

# #descriptive overview
# print(df.describe())

# #remove duplicates
# # print(df.duplicated().sum())
# df = df.drop_duplicates()
# print(df.info())

# df.to_csv('C:/Users/gboch/Desktop/inzynierka/engineering_thesis/second/datasets/cleaned.csv', index=False)
# """Encoding Categorical Data - objects -> float
# Label Encoding Scheme
# One-Hot-Encoding"""
# # Check the data types of the columns
# # print(df.dtypes)
def data_numeric(df):
    age = sorted(df["AgeCategory"].unique())
    # age = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older']
    health = ['Poor', 'Fair', 'Good', 'Very good', 'Excellent']
    df = pd.get_dummies(df, columns=['Race', ], dtype=int)
    #change
    for o in df.columns:
        print(df[o][0])
        if df[o][0] in ['Yes', 'No']:
            df[o] = df[o].map({'Yes': 1.0, 'No': 0.0})
        elif df[o][0] in ['Male', 'Female']:
            df[o] = df[o].map({'Male': 1.0, 'Female': 0.0})
        elif df[o][0] in age:
            df[o] = df[o].map({'18-24':1.0, '25-29':2.0, '30-34':3.0, '35-39':4.0, '40-44':5.0, '45-49':6.0, '50-54':7.0, '55-59':8.0, '60-64':9.0, '65-69':10.0, '70-74':11.0, '75-79':12.0, '80 or older':13.0})
        elif df[o][0] in health:
            df[o] = df[o].map({'Poor':1.0, 'Fair':2.0, 'Good':3.0, 'Very good':4.0, 'Excellent':5.0})
        else:
            pass
        df.dropna(inplace=True)
    return df

# #check the dataset
df_numeric = data_numeric(df)
print(df_numeric.head())
print(df_numeric.info())
# df_numeric.to_csv('C:/Users/gboch/Desktop/inzynierka/engineering_thesis/second/datasets/after_preprocessing.csv', index=False)

# print(df_numeric["HeartDisease"].value_counts())

#correlation matrix and heatmap
def headmap(df):
    cor_matrix = df.corr().abs()
    fig, ax = plt.subplots(figsize=(18,18))
    dataplot = sns.heatmap(df.corr().abs(), cmap="YlGnBu", annot=True, annot_kws={'size': 10}, ax=ax)
    plt.show()

headmap(df_numeric)
# def standarization(df):
#     y = df['HeartDisease']
#     X = df.drop('HeartDisease',axis=1)
#     # StandardScaler object initialization
#     scaler = MinMaxScaler()
#     # standarization
#     standarized = scaler.fit_transform(X)
#     df_standarized = pd.DataFrame(standarized, columns=X.columns)
#     return df_standarized

# df_standarized = standarization(df_numeric)

