import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

df = pd.read_csv('datasets/data_after_preprocessing.csv')
df = df.drop('Unnamed: 0',axis=1)

y = df['HeartDisease']
X = df.drop('HeartDisease',axis=1)
X = StandardScaler().fit_transform(X)


def lda_algorithm(df, components):

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
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


explained_variance, components, transformed_data = lda_algorithm(df, 2)