import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression

df_after_preprocessing = pd.read_csv('datasets/data_after_preprocessing.csv')
df = pd.read_csv('datasets/heart_2020_cleaned.csv')

# df_after_preprocessing = df_after_preprocessing.drop('Unnamed: 0',axis=1)

# descriptive statistics
mean = df_after_preprocessing.mean()
median = df_after_preprocessing.median()
std = df_after_preprocessing.std()
minimum = df_after_preprocessing.min()
maximum = df_after_preprocessing.max()

X = df_after_preprocessing.drop(columns='HeartDisease')
y = pd.DataFrame(df['HeartDisease'])

# #distribution of data in each class chart
def distribution(df):
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize = (15,9))
    ax = fig.gca()
    df.hist(ax = ax,bins=30)
    plt.tight_layout()
    plt.show()
distribution(df)

def scatter_plot(df):
    # pairwise bivariate distributions - problemm!!!!!!!!!!!!!1
    sns.pairplot(df, hue='HeartDisease')
    print(df)
    plt.show()
# scatter_plot(df_after_preprocessing)


df_numerical = df._get_numeric_data()
# scatter_plot(df_numerical)

def do_you_have_heart_disease(y):
    sns.countplot(x="HeartDisease", data=y)
    plt.title("Do you have a heart disease?")
    plt.show()
# do_you_have_heart_disease()
def histograms(df):
    df_numerical = df._get_numeric_data()
    for i,column in  enumerate(df_numerical.columns):
        plt.rcParams.update({'font.size': 12})
        plt.hist(df[df["HeartDisease"]=='No'][column], bins=30, label="No HeartDisease")
        plt.hist(df[df["HeartDisease"]=='Yes'][column], bins=30, label="HeartDisease")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.legend()
        plt.xticks([])
        plt.show()
histograms(df)
# plt.hist(df[df["HeartDisease"]=='No']["BMI"],bins=60,histtype='bar',label="No HeartDisease")
# plt.hist(df[df["HeartDisease"]=='Yes']["BMI"], bins=60,histtype='bar',label="HeartDisease")
# plt.tight_layout()
# plt.show()
#pie charts
def pie(df):
    for col in df.columns:
        if col == "BMI":
            continue
        df[col].value_counts().plot(kind='pie', autopct="%.1f")
        plt.title(col)
        plt.show()

        
# pie(df)

def outliers(df, column):
    # for col in df.columns:
    #     df_num=df[col]
    #     sns.boxplot(x=df_num)
    #     plt.show()
    #rows with outliers (dataFrame) 
    df.reset_index(inplace=True, drop=True)
    print(df)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    
    # Create arrays of Boolean values indicating the outlier rows
    upper_array = np.where(df[column]>=upper)[0]
    lower_array = np.where(df[column]<=lower)[0]
    print(upper_array)
    # Removing the outliers
    df.drop(index=upper_array, inplace=True)
    # df.drop(index=lower_array, inplace=True)
    return df

# df_numerical = df_numerical.drop(["HeartDisease"], axis=1)
# df = outliers(df_after_preprocessing, 'BMI')
# df = outliers(df_after_preprocessing, 'PhysicalHealth')
# df = outliers(df_after_preprocessing, 'MentalHealth')
# df = outliers(df_after_preprocessing, 'SleepTime')
