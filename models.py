import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Lasso
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import VarianceThreshold
# import xgboost as xgb
from sklearn.metrics import mean_squared_error

def evaluate_model(model, x_test, y_test):
    from sklearn import metrics

    # Predict Test Data 
    y_pred = model.predict(x_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    kappa = metrics.cohen_kappa_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(x_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa, 
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}


def cor_selector(X, y,num_feats='all'):
    cor_list = []
    feature_name = X.columns.tolist()
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature

df = pd.read_csv('second/datasets/after_preprocessing.csv')
# df = df.drop('Unnamed: 0',axis=1)
print(df.head())
y = df['HeartDisease']
X = df.drop('HeartDisease',axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

def pca_algorithm(X, n_components=None):
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

# pca, X_pca = pca_algorithm(X, .95)
# X_train = pca.transform(X_train)
# X_test = pca.transform(X_test)


# KNeighborsClassifier
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)
# param_grid = {
#     'metric': ['minkowski', 'euclidean']
# }

# model = KNeighborsClassifier(n_neighbors=8, metric='euclidean')
# model.fit(X_train, y_train)
# eval = evaluate_model(model, X_test, y_test)

# grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# najlepsze_parametry = grid_search.best_params_
# najlepsza_dokladnosc = grid_search.best_score_
# print(najlepsze_parametry, najlepsza_dokladnosc)

# k_values = [i for i in range (1,31)]
# scores = []

# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# for k in k_values:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     score = cross_val_score(knn, X, y, cv=5)
#     scores.append(np.mean(score))
#     print(scores)

# sns.lineplot(x = k_values, y = scores, marker = 'o')
# plt.xlabel("K Values")
# plt.ylabel("Accuracy Score")
# plt.show()




# # DecisionTree

# cor_support, columns = cor_selector(X, y, num_feats=4)
# print(columns)
# X = df[columns]
# y = df['HeartDisease']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30)


# param_grid = {
#     'criterion': ['gini', 'entropy', 'log_loss'],
#     'splitter': ['best', 'random'],
#     'max_depth': [None, 10,15]
# }

# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)
# eval = evaluate_model(model, X_test, y_test)

# grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)
# najlepsze_parametry = grid_search.best_params_
# najlepsza_dokladnosc = grid_search.best_score_ 
# print(najlepsze_parametry, najlepsza_dokladnosc)





# RandomForest
# randomOverSample to deal with imbalanced data
# ros = RandomOverSampler(random_state=42)
# x_ros, y_ros = ros.fit_resample(X, y)
# X_train, X_test, y_train, y_test = train_test_split(x_ros, y_ros, test_size=0.2, random_state=30)

# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)
# model = RandomForestClassifier(random_state=30)
# model.fit(X_train, y_train)
# eval = evaluate_model(model, X_test, y_test)

# LogisticRegression

# #cross validation

# # kf = KFold(n_splits = 5)
# # for train_index, test_index in kf.split(X.values):
# #     print("TRAIN:", train_index, "TEST:", test_index)
# #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
# #     y_train, y_test = y[train_index], y[test_index]

threshold = VarianceThreshold()

new_X = threshold.fit_transform(X)

print("Oryginalne dane:")
print(X)

print("\nDane po usunięciu cech o niskiej wariancji:")
print(new_X)

X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size = 0.2, random_state = 30)

model = LogisticRegression()
model.fit(X_train, y_train)
eval = evaluate_model(model, X_test, y_test)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)
# from sklearn.linear_model import LassoCV

# # Lasso with 5 fold cross-validation
# model = LassoCV(cv=5, random_state=0, max_iter=10000)
# model.fit(X_train, y_train)
# print(model.alpha_)

# model = Lasso(model.alpha_)
# model.fit(X_train, y_train)
# print("Train Set R-square Val: {:.3f}".format(model.score(X_train, y_train)))
# print("Test Set R-square Val: {:.3f}".format(model.score(X_test, y_test)))
# print(mean_squared_error(y_test, model.predict(X_test)))
# print(mean_squared_error(y_train, model.predict(X_train)))

# Print result
print(model)
print('Accuracy:', eval['acc'])
print('Precision:', eval['prec'])
print('Recall:', eval['rec'])
print('F1 Score:', eval['f1'])
print('Cohens Kappa Score:', eval['kappa'])
print('Area Under Curve:', eval['auc'])
print('Confusion Matrix:\n', eval['cm'])
sns.heatmap(pd.DataFrame(eval['cm']), annot=True, cmap="YlGnBu" ,fmt='g')
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')