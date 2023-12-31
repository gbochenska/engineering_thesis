import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
from sklearn import metrics


def evaluate_model(model, x_test, y_test):
    # Predict Test Data 
    start_time = time.time()
    y_pred = model.predict(x_test)
    end_time = time.time()

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
    prediction_time = end_time - start_time
    print(f"Czas predykcji: {prediction_time} sekundy")

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa, 
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}


def pca_algorithm(X, n_components=None):
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca


df = pd.read_csv('datasets/data_after_preprocessing.csv')
df = df.drop('Unnamed: 0',axis=1)
y = df['HeartDisease']
X = df.drop('HeartDisease',axis=1)

k_best = ['AgeCategory','Stroke','GenHealth','Sex','Diabetic','KidneyDisease','DiffWalking','Smoking']
X = X[k_best]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# model = KNeighborsClassifier()
# param_grid = {
#     'metric': ['minkowski', 'euclidean']
# }
# grid_search = GridSearchCV(model, param_grid, cv=5, scoring='precision')
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


# k_best = ['AgeCategory','Stroke','GenHealth','Sex','Diabetic','KidneyDisease','DiffWalking','Smoking',
#           'PhysicalHealth','SkinCancer','Asthma','Race_Black','AlcoholDrinking','BMI','Race_White',
#           'Race_Asian','Race_Hispanic','SleepTime','Race_American Indian/Alaskan Native','PhysicalActivity',
#           'MentalHealth','Race_Other']
# precisions = []
# accuracies = []
# for i in range(len(k_best)):
#     k_now = k_best[0:len(k_best)-i]
#     print(k_now)
#     X = df[k_now]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30)
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.fit_transform(X_test)
#     model = KNeighborsClassifier(n_neighbors=8, metric='euclidean')
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracies.append(metrics.accuracy_score(y_test, y_pred))
#     precisions.append(metrics.precision_score(y_test, y_pred))
#     print(accuracies, precisions)
# print(accuracies, precisions)
from imblearn.over_sampling import SMOTE

# smote = SMOTE()
# X_train, y_train = smote.fit_resample(X_train, y_train)
model = KNeighborsClassifier(n_neighbors=8, metric='euclidean', weights='distance', p=2, leaf_size=30)
print("aaaa")
start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
eval = evaluate_model(model, X_test, y_test)

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
training_time = end_time - start_time
print(f"Czas uczenia: {training_time} sekundy")