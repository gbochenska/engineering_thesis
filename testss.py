import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression

# def evaluate_model(model, x_test, y_test):
#     from sklearn import metrics

#     # Predict Test Data 
#     y_pred = model.predict(x_test)

#     # Calculate accuracy, precision, recall, f1-score, and kappa score
#     acc = metrics.accuracy_score(y_test, y_pred)
#     prec = metrics.precision_score(y_test, y_pred)
#     rec = metrics.recall_score(y_test, y_pred)
#     f1 = metrics.f1_score(y_test, y_pred)
#     kappa = metrics.cohen_kappa_score(y_test, y_pred)

#     # Calculate area under curve (AUC)
#     y_pred_proba = model.predict_proba(x_test)[::,1]
#     fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
#     auc = metrics.roc_auc_score(y_test, y_pred_proba)

#     # Display confussion matrix
#     cm = metrics.confusion_matrix(y_test, y_pred)

#     return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa, 
#             'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}

# df=pd.read_csv("second\\datasets\\after_preprocessing.csv")
# print(df.head())

# y = df['HeartDisease']
# X = df.drop('HeartDisease',axis=1)

# def headmap(df):
#     cor_matrix = df.corr().abs()
#     fig, ax = plt.subplots(figsize=(18,18))
#     dataplot = sns.heatmap(df.corr().abs(), cmap="YlGnBu", annot=True, annot_kws={'size': 10}, ax=ax)
#     plt.show()

# # headmap(df)


# # X = df[['AgeCategory','DiffWalking','Stroke','Diabetic', 'KidneyDisease','PhysicalHealth','GenHealth','Smoking']]
# # y = df['HeartDisease']

# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30)

# # model = LogisticRegression()
# # model.fit(X_train, y_train)
# # eval = evaluate_model(model, X_test, y_test)


# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_selection import SelectFromModel
# def lasso_selector(X, Y, num_feats='all'):
#     X_norm = MinMaxScaler().fit_transform(X)
#     embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=num_feats)
#     embeded_lr_selector.fit(X_norm, y)
#     embeded_lr_support = embeded_lr_selector.get_support()
#     embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
#     return embeded_lr_support, embeded_lr_feature

# df = pd.read_csv('datasets/data_after_preprocessing.csv')
# df = df.drop('Unnamed: 0',axis=1)
# y = df['HeartDisease']
# X = df.drop('HeartDisease',axis=1)
# num_feats = 10

# print(lasso_selector(X, y, 21))

# col = ['BMI', 'Stroke', 'Sex', 'AgeCategory', 'GenHealth', 'SleepTime']
# y = df['HeartDisease']
# X = df[col]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30)
# from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

# model = LogisticRegression()
# model.fit(X_train, y_train)
# eval = evaluate_model(model, X_test, y_test)

# print(model)
# print('Accuracy:', eval['acc'])
# print('Precision:', eval['prec'])
# print('Recall:', eval['rec'])
# print('F1 Score:', eval['f1'])
# print('Cohens Kappa Score:', eval['kappa'])
# print('Area Under Curve:', eval['auc'])
# print('Confusion Matrix:\n', eval['cm'])
# sns.heatmap(pd.DataFrame(eval['cm']), annot=True, cmap="YlGnBu" ,fmt='g')
# plt.tight_layout()
# plt.title('Confusion matrix', y=1.1)
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')

# accuracy and precision depends on features
accur = [0.9089727585204439, 0.9089556577799819, 0.9082716281614994, 0.90844263556612, 0.9079809155736444, 
         0.90844263556612, 0.9085281392684303, 0.9089727585204439, 0.9085965422302785, 0.9094515792533817, 
         0.9094002770319954, 0.9090924637036784, 0.9091266651846025, 0.9092805718487611, 0.9089727585204439,
         0.9087846503753613, 0.9037912341604392, 0.903192708244267, 0.9100330044290917, 0.9100330044290917,
         0.9100330044290917, 0.9100330044290917]

prec = [0.4660831509846827, 0.4628975265017668, 0.42232277526395173, 0.4124293785310734, 0.3880597014925373, 
        0.38107416879795397, 0.3358208955223881, 0.34183673469387754, 0.28125, 0.2976190476190476, 
        0.2465753424657534, 0.1232876712328767, 0.1267605633802817, 0.1206896551724138, 0.14772727272727273,
        0.13131313131313133, 0.14563106796116504, 0.13636363636363635, 0.0, 0.0, 0.0, 0.0]


plt.plot(np.arange(len(accur)), accur)
plt.plot(np.arange(len(prec)), prec)
plt.xlabel("Liczba cech")
plt.ylabel("Wynik")
plt.legend(['Accuracy', 'Precision'])
plt.title('Miary oceny w zależności od liczby cech')
plt.show()