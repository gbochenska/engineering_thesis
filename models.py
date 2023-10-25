import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# import xgboost as xgb

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

df = pd.read_csv('datasets/data_after_preprocessing.csv')
df = df.drop('Unnamed: 0',axis=1)
y = df['HeartDisease']
X = df.drop('HeartDisease',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

def pca_algorithm(X, n_components=None):
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

pca, X_pca = pca_algorithm(X, .95)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


# # KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors = 5)
# model.fit(X_train, y_train)
# eval = evaluate_model(model, X_test, y_test)

# # DecisionTree
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)
# eval = evaluate_model(model, X_test, y_test)

# RandomForest
# model = RandomForestClassifier()
# model.fit(X_train, y_train)
# eval = evaluate_model(rf, X_test, y_test)

# LogisticRegression

#cross validation

# kf = KFold(n_splits = 5)
# for train_index, test_index in kf.split(X.values):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y[train_index], y[test_index]

model = LogisticRegression(random_state=16)
model.fit(X_train, y_train)
eval = evaluate_model(model, X_test, y_test)
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