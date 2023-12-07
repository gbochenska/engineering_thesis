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

# accuracy and precision depends on features knearestneighbors
accur = [0.9089727585204439, 0.9089556577799819, 0.9081006207568788, 0.9082032251996511, 0.9082203259401133, 
         0.9078783111308719, 0.9079296133522582, 0.9080493185354925, 0.9079809155736444, 0.907604699283479, 
         0.9071600800314653, 0.908459736306582, 0.9085965422302785, 0.9086136429707407, 0.909810694803085, 
         0.9066128563366794, 0.909554183696154, 0.9029361971373361, 0.9077757066880996, 0.9067838637413,
         0.9074849941002445, 0.9100330044290917]
prec = [0.4660831509846827, 0.4654983570646221, 0.43542857142857144, 0.4438614900314795, 0.44432773109243695,
        0.4351851851851852, 0.43743641912512715, 0.44081632653061226, 0.43788819875776397, 0.4234913793103448, 
        0.41515151515151516, 0.44933920704845814, 0.45464362850971923, 0.45434543454345433, 0.49295774647887325, 
        0.4215070643642072, 0.48507462686567165, 0.36990595611285265, 0.4472843450479233, 0.39651416122004357,
        0.40581542351453853, 0.0]

#Decision tree
# accur = [0.8541819860799973, 0.8544897994083144, 0.8556184482788105, 0.8564221830805274, 0.8573285223250167,
#           0.8634334866699728, 0.8630059681584212, 0.86266395334918, 0.8632453785248901, 0.8994647468235375,
#             0.9000803734801717, 0.9019956564119226, 0.9030730030610326, 0.9049711852523215, 0.9109906458949673,
#               0.9107854370094225, 0.9113668621851326, 0.9116233732920636, 0.9116746755134497, 
#               0.9109564444140431, 0.9100330044290917, 0.9100330044290917]
# prec = [0.2248989218328841, 0.2282463186077644, 0.22868349249658937, 0.22866539726501645, 0.23190675017397355,
#          0.23883457926011117, 0.24027200604457877, 0.23639132089836315, 0.23478092283830942, 
#          0.34672619047619047, 0.34953464322647365, 0.36478711162255467, 0.36442371752165226, 
#          0.3897168405365127, 0.535264483627204, 0.5321637426900585, 0.5553977272727273, 
#          0.5792163543441227, 0.6176470588235294, 0.5798816568047337, 0.0, 0.0]


# RandomForest
accur = [0.8982163927698069, 0.897959881662876, 0.8954631735554149, 0.892983566188416, 0.8938044017305949, 
         0.8891701010653761, 0.8881611573781145, 0.8885373736682798, 0.8865023855532944, 0.9017904475263779, 
         0.9017733467859158, 0.9031756075038049, 0.903757032679515, 0.9054671067257212, 0.9104434222001813, 
         0.9108367392308087, 0.911212955520974, 0.9116062725516015, 0.9117088769943739, 0.9109735451545052, 
         0.9100330044290917, 0.9100330044290917]
prec = [0.31901519119958094, 0.31243358129649307, 0.30350553505535055, 0.2902818679007152, 0.3044911413267408, 
        0.2944743935309973, 0.2881748923484598, 0.29112662013958124, 0.28405524168236035, 0.36844978165938863, 
        0.36650082918739635, 0.3832265579499126, 0.38437303087586644, 0.40683879972086534, 0.5127118644067796, 
        0.5303225806451612, 0.5465587044534413, 0.5779661016949152, 0.6231155778894473, 0.5779036827195467, 
        0.0, 0.0]


# #Logistic Regression
# accur = [0.9119653881013048, 0.9119653881013048, 0.9118456829180703, 0.9117943806966842, 0.9118285821776083, 
#          0.9119995895822289, 0.912016690322691, 0.9119653881013048, 0.9120679925440771, 0.9119653881013048, 
#          0.9120679925440771, 0.9119653881013048, 0.9121192947654634, 0.9121876977273116, 0.9121705969868495, 
#          0.9115549703302153, 0.9113839629255946, 0.9115207688492912, 0.9117259777348359, 0.9104434222001813, 
#          0.909554183696154, 0.9100330044290917]
# prec = [0.5548011639185257, 0.5548011639185257, 0.5501893939393939, 0.549662487945998, 0.5506268081002893, 
#         0.5536881419234361, 0.5542056074766355, 0.5527544351073763, 0.5561850802644004, 0.5519779208831647, 
#         0.5549399815327793, 0.5533522190745986, 0.558206106870229, 0.5595463137996219, 0.5599232981783318, 
#         0.5497206703910614, 0.5394605394605395, 0.5516014234875445, 0.5632183908045977, 0.5148514851485149, 
#         0.46744186046511627, 0.0]

# plt.plot(np.arange(len(accur))[::-1], accur)
# plt.plot(np.arange(len(prec))[::-1], prec)
# plt.xlabel("Liczba cech")
# plt.ylabel("Wynik")
# plt.legend(['Accuracy', 'Precision'])
# plt.title('Miary oceny w zależności od liczby cech')
# plt.grid()
# plt.show()

# from sklearn.metrics import confusion_matrix

# cm =[[51939,1277],[4520,741]]
# sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
# plt.text(1.5, 1.3, 'TP', ha='center', va='center', color='green', fontsize=10)
# plt.text(0.5, 1.3, 'FN', ha='center', va='center', color='red', fontsize=10)
# plt.text(1.5, 0.3, 'FP', ha='center', va='center', color='red', fontsize=10)
# plt.text(0.5, 0.3, 'TN', ha='center', va='center', color='green', fontsize=10)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()



acc_basic = [0.90087, 0.85461, 0.89859, 0.91197]
prec_basic = [0.3672, 0.22604, 0.32441, 0.5548]
recall_basic = [0.14085, 0.25413, 0.11747, 0.10872]

acc_hiperparam = [0.90897, 0.85869, 0.8985, 0.91209]
prec_hiperparam = [0.46608, 0.22316, 0.31979, 0.5655]
recall_hiperparam = [0.08097, 0.22999, 0.11347, 0.09846]

acc_best = [0.90945, 0.91167, 0.91167, 0.91209]
prec_best = [0.48303, 0.61765, 0.61765, 0.5655]
recall_best = [0.09199, 0.0479, 0.0479, 0.10872]

experiments = ['KNeighbors', 'DecisionTree', 'RandomForest', 'LogisticRegression']
bar_width = 0.3
for i, label in enumerate(experiments):
    plt.bar(i - bar_width, acc_basic[i], color='skyblue', width=bar_width)
    plt.bar(i, acc_hiperparam[i], color='steelblue', width=bar_width)
    plt.bar(i + bar_width, acc_best[i], color='turquoise', width=bar_width)
plt.legend(['Klasyczny model', 'Dostrajanie hiperparametrów', 'Najlepszy model'], loc='lower right')
plt.xticks(range(len(experiments)), experiments)
plt.xlabel('Model')
plt.ylabel('Wartość oceny modelu')
plt.title('Ocena dokładności modeli')
plt.ylim(0, 1)
plt.show()

for i, label in enumerate(experiments):
    plt.bar(i - bar_width, prec_basic[i], color='skyblue', width=bar_width)
    plt.bar(i, prec_hiperparam[i], color='steelblue', width=bar_width)
    plt.bar(i + bar_width, prec_best[i], color='turquoise', width=bar_width)
plt.legend(['Klasyczny model', 'Dostrajanie hiperparametrów', 'Najlepszy model'], loc='lower right')
plt.xticks(range(len(experiments)), experiments)
plt.xlabel('Model')
plt.ylabel('Wartość oceny modelu')
plt.title('Ocena precyzji modeli')
plt.ylim(0, 1)
plt.show()

for i, label in enumerate(experiments):
    plt.bar(i - bar_width, recall_basic[i], color='skyblue', width=bar_width)
    plt.bar(i, recall_hiperparam[i], color='steelblue', width=bar_width)
    plt.bar(i + bar_width, recall_best[i], color='turquoise', width=bar_width)
plt.legend(['Klasyczny model', 'Dostrajanie hiperparametrów', 'Najlepszy model'])
plt.xticks(range(len(experiments)), experiments)
plt.xlabel('Model')
plt.ylabel('Wartość oceny modelu')
plt.title('Ocena czułości modeli')
plt.ylim(0, 1)
plt.show()