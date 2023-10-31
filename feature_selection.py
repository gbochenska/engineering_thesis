import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel



def cor_selector(X, y,num_feats='all'):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature


def chi_2(X, y, num_feats='all'):
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()
    return chi_support, chi_feature


def RFE_selector(X, y, num_feats='all'):
    X_norm = MinMaxScaler().fit_transform(X)
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    return rfe_support, rfe_feature


def lasso_selector(X, Y, num_feats='all'):
    X_norm = MinMaxScaler().fit_transform(X)
    embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1"), max_features=num_feats)
    embeded_lr_selector.fit(X_norm, y)
    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
    return embeded_lr_support, embeded_lr_feature


df = pd.read_csv('datasets/data_after_preprocessing.csv')
df = df.drop('Unnamed: 0',axis=1)
y = df['HeartDisease']
X = df.drop('HeartDisease',axis=1)
num_feats = 10

def features_proportion(X, y):
    pearson = np.zeros(23)
    chi = np.zeros(23)
    rfe = np.zeros(23)
    embeded_lr = np.zeros(23)

    for i in range(1, 23):
        num_feats = i
        cor_support, cor_feature = cor_selector(X, y,num_feats)
        chi_support, chi_feature = chi_2(X, y,num_feats)
        rfe_support, rfe_feature = RFE_selector(X, y,num_feats)
        embeded_lr_support, embeded_lr_feature = RFE_selector(X, y,num_feats)

        pearson = [pearson[j]+1 if cor_support[j] is True else pearson[j] for j in range(22)]
        chi = [chi[j]+1 if chi_support[j] is True else chi[j] for j in range(22)]
        rfe = [rfe[j]+1 if rfe_support[j] is True else rfe[j] for j in range(22)]
        embeded_lr = [embeded_lr[j]+1 if embeded_lr_support[j] is True else embeded_lr[j] for j in range(22)]
        
    plt.pie(pearson, labels = X.columns, autopct='%.0f%%')
    plt.show()
    plt.pie(chi, labels = X.columns, autopct='%.0f%%')
    plt.show()
    plt.pie(rfe, labels = X.columns, autopct='%.0f%%')
    plt.show()
    plt.pie(embeded_lr, labels = X.columns, autopct='%.0f%%')
    plt.show()
    return pearson, chi, rfe, embeded_lr

pearson, chi, rfe, embeded_lr =  features_proportion(X, y)