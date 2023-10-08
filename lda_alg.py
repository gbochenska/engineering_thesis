import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

from data_preprocessing import data_numeric


df = pd.read_csv('datasets/heart_2020_cleaned.csv')
df = data_numeric(df)

def standarization(df, target, features):
    # StandardScaler object initialization
    scaler = StandardScaler()

    # standarization
    df_standarized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_standarized

target = df['HeartDisease']
features = df.drop('HeartDisease',axis=1)
stand = standarization(df, target, features)

def lda_algorithm(df, components):

    X_train, X_test, y_train, y_test = train_test_split(features, target, stratify=target, test_size=0.2)
    lda = LDA(n_components=components)
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)

    sns.scatterplot(x=X_train[:,0], y=X_train[:,1], hue=y_train)
    plt.show()
    # eigenvalues that explain the variance in the data
    explained_variance = lda.explained_variance_

    # Eigenvectors (principal components)
    components = lda.components_

    # Transformed data
    transformed_data = lda.transform(df)

    return explained_variance, components, transformed_data


explained_variance, components, transformed_data = lda_algorithm(stand, 2)