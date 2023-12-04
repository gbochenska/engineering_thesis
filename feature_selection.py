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
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso



def cor_selector(X, y,num_feats='all'):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append((i, cor))
    # replace NaN with 0
    cor_list = [(i, 0) if np.isnan(c) else (i, c) for i, c in cor_list]

    # sort correlations in descending order
    sorted_cor_list = sorted(cor_list, key=lambda x: abs(x[1]), reverse=True)

    # feature selection? 0 for not select, 1 for select
    #selected_features = [feature[0] for feature in sorted_cor_list[:num_feats]]
    for feature, correlation in sorted_cor_list:
        print(f"{feature}: {correlation}")
    return sorted_cor_list


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


def lasso_selector(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    from sklearn.linear_model import LassoCV
    # Lasso with 5 fold cross-validation
    model = LassoCV(cv=5, random_state=0, max_iter=10000)
    model.fit(X_train, y_train)
    model = Lasso(model.alpha_)
    model.fit(X_train, y_train)
    coefficients = model.coef_
    important_feature_indices = [i for i, coef in enumerate(coefficients) if abs(coef) > 0]
    important_coefficients = coefficients[important_feature_indices]
    feature_names = X.columns
    important_feature_names = [feature_names[i] for i in important_feature_indices]
    sorted_indices = np.argsort(np.abs(important_coefficients))[::-1]
    sorted_feature_names = [important_feature_names[i] for i in sorted_indices]
    sorted_coefficients = important_coefficients[sorted_indices]
    for feature, coef in zip(sorted_feature_names, sorted_coefficients):
        print(f"{feature}: {coef}")
    return sorted_feature_names

def variance_selector(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Standaryzacja danych
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    variance_selector = VarianceThreshold(threshold=0.1)
    X_train_selected = variance_selector.fit_transform(X_train_scaled)
    linear_model = LogisticRegression()
    linear_model.fit(X_train_selected, y_train)
    coefficients = linear_model.coef_[0]
    important_feature_indices = [i for i, coef in enumerate(coefficients) if abs(coef) > 0]
    important_coefficients = coefficients[important_feature_indices]
    feature_names = X.columns
    important_feature_names = [feature_names[i] for i in important_feature_indices]
    sorted_indices = np.argsort(np.abs(important_coefficients))[::-1]
    sorted_feature_names = [important_feature_names[i] for i in sorted_indices]
    sorted_coefficients = important_coefficients[sorted_indices]
    for feature, coef in zip(sorted_feature_names, sorted_coefficients):
        print(f"{feature}: {coef}")
    return sorted_feature_names


df = pd.read_csv('datasets/data_after_preprocessing.csv')
df = df.drop('Unnamed: 0',axis=1)
y = df['HeartDisease']
X = df.drop('HeartDisease',axis=1)
num_feats = 10
sorted_feature_names1 = cor_selector(X, y)
sorted_feature_names2 = lasso_selector(X, y)
sorted_feature_names3 = variance_selector(X, y)
print(sorted_feature_names1, sorted_feature_names2, sorted_feature_names3)

# def features_proportion(X, y):
#     pearson = np.zeros(23)
#     chi = np.zeros(23)
#     rfe = np.zeros(23)
#     embeded_lr = np.zeros(23)

#     for i in range(1, 23):
#         num_feats = i
#         cor_support, cor_feature = cor_selector(X, y,num_feats)
#         chi_support, chi_feature = chi_2(X, y,num_feats)
#         rfe_support, rfe_feature = RFE_selector(X, y,num_feats)
#         embeded_lr_support, embeded_lr_feature = RFE_selector(X, y,num_feats)

#         pearson = [pearson[j]+1 if cor_support[j] is True else pearson[j] for j in range(22)]
#         chi = [chi[j]+1 if chi_support[j] is True else chi[j] for j in range(22)]
#         rfe = [rfe[j]+1 if rfe_support[j] is True else rfe[j] for j in range(22)]
#         embeded_lr = [embeded_lr[j]+1 if embeded_lr_support[j] is True else embeded_lr[j] for j in range(22)]
        
#     plt.pie(pearson, labels = X.columns, autopct='%.0f%%')
#     plt.show()
#     plt.pie(chi, labels = X.columns, autopct='%.0f%%')
#     plt.show()
#     plt.pie(rfe, labels = X.columns, autopct='%.0f%%')
#     plt.show()
#     plt.pie(embeded_lr, labels = X.columns, autopct='%.0f%%')
#     plt.show()
#     return pearson, chi, rfe, embeded_lr

# pearson, chi, rfe, embeded_lr =  features_proportion(X, y)