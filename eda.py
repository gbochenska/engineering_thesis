import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df_after_preprocessing = pd.read_csv('datasets/data_after_preprocessing.csv')
df = pd.read_csv('datasets/heart_2020_cleaned.csv')

df_after_preprocessing = df_after_preprocessing.drop('Unnamed: 0',axis=1)

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
    fig = plt.figure(figsize = (15,9))
    ax = fig.gca()
    df_after_preprocessing.hist(ax = ax,)
    plt.tight_layout()
    plt.show()
# distribution(df_after_preprocessing)

def scatter_plot(df):
    # pairwise bivariate distributions - problemm!!!!!!!!!!!!!1
    sns.pairplot(df, hue='HeartDisease')
    plt.show()

df_numerical = df._get_numeric_data()
df_numerical = df_numerical.join(df['HeartDisease'], lsuffix='_caller', rsuffix='_other')
# scatter_plot(df_numerical)

def do_you_have_heart_disease(y):
    sns.countplot(x="HeartDisease", data=y)
    plt.title("Do you have a heart disease?")
    plt.show()

def histograms(df):
    for i,column in  enumerate(df.columns):
        plt.hist(df[df["HeartDisease"]=='No'][column], bins=3, label="No HeartDisease")
        plt.hist(df[df["HeartDisease"]=='Yes'][column], bins=3, label="HeartDisease")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.legend()
        plt.xticks([])
        plt.show()
# histograms(df)

#pie charts
def pie(df):
    for col in df.columns:
        if col == "BMI":
            continue
        df[col].value_counts().plot(kind='pie', autopct="%.1f")
        plt.title(col)
        plt.show()
# pie(df)

def outliers(df):
    # for col in df.columns:
    #     df_num=df[col]
    #     sns.boxplot(x=df_num)
    #     plt.show()
    #rows with outliers (dataFrame) 
    Q1 = [24.03, 0, 0, 6]
    Q3 = [31.42, 2, 3, 8]
    columns = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
    IQR = []
    lower_limit = []
    upper_limit = []
    outliers = []
    for i in range(4):
        IQR.append(Q3[i] - Q1[i])
        lower_limit.append(Q1[i] - 1.5*IQR[i])
        upper_limit.append(Q3[i] + 1.5*IQR[i])
        outliers.append(df[(df[columns[i]]<lower_limit[i]) | (df[columns[i]]>upper_limit[i])])
    return outliers
# print(outliers(df_numerical))
