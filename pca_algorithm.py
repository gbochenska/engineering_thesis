import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from data_preprocessing import data_numeric

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

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

def pca(df, components):
    pca = PCA(n_components = 2)
    transformed_data = pca.fit_transform(df.drop('HeartDisease', axis=1))

    X_train, X_test, y_train, y_test = train_test_split(transformed_data, target, stratify=target, test_size=0.2)
    sns.scatterplot(x=X_train[:,0], y=X_train[:,1], hue=y_train)
    plt.show()
    # eigenvalues that explain the variance in the data
    explained_variance = pca.explained_variance_

    # Eigenvectors (principal components)
    components = pca.components_

    # Transformed data
    transformed_data = pca.transform(df)

    return explained_variance, components, transformed_data



explained_variance, components, transformed_data = pca(stand, 17)
plt.plot(np.cumsum(explained_variance) / np.sum(explained_variance))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()