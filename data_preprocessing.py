import pandas as pd 
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split



#Load dataset
df = pd.read_csv('datasets/heart_2020_cleaned.csv')
print(df.head())

print(df.shape)

#Check the data info
print(df.info())

#Check the null values
df.isnull()

#descriptive overview
df.describe()

# objects -> float
# Check the data types of the columns
print(df.dtypes)
#change
for o in df.columns:
    df[o] = df[o].map({'Yes': 1.0, 'No': 0.0})
print(df.info())

print(df["HeartDisease"].value_counts())


#heatmap
sns.set(rc={'figure.figsize':(10,8)})
ax = sns.heatmap(df)


# X = df.drop(columns='HeartDisease')
# Y = df['HeartDisease']

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# cor_matrix = df.corr().abs()
# print(cor_matrix)