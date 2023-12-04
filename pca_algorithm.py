import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv('datasets/data_after_preprocessing.csv')
df = df.drop('Unnamed: 0',axis=1)

y = df['HeartDisease']
X = df.drop('HeartDisease',axis=1)
X = StandardScaler().fit_transform(X)

def pca_algorithm(X, n_components=None):
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

pca, X_pca = pca_algorithm(X)
# explained_variance plot
xi = np.arange(1, 23, step=1)
explained_variance = pca.explained_variance_ratio_
plt.plot(xi, np.cumsum(explained_variance), marker='o', linestyle='--', color='b')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.9, 'Próg odcięcia 95%', color = 'red', fontsize=10)
plt.xlabel('Liczba składowych')
plt.ylabel('Skumulowana wartość wariancji')
plt.grid(axis='x')
plt.title("PCA")
plt.show()

sns.set()
# Bar plot of explained_variance
plt.bar(
    range(1,len(pca.explained_variance_)+1),
    pca.explained_variance_
    )
plt.xlabel('Liczba składowych')
plt.ylabel('Wyjaśniona wariancja')
plt.title('Wariancja')
plt.show()

# visualize 2d plot
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['HeartDisease'])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA dla 2 składowych")
plt.show()


#     X_train, X_test, y_train, y_test = train_test_split(transformed_data, target, stratify=target, test_size=0.2)
#     sns.scatterplot(x=X_train[:,0], y=X_train[:,1], hue=y_train)
#     plt.show()
#     # eigenvalues that explain the variance in the data
#     explained_variance = pca.explained_variance_

#     # Eigenvectors (principal components)
#     components = pca.components_

#     # Transformed data
#     transformed_data = pca.transform(df)

#     return explained_variance, components, transformed_data



# explained_variance, components, transformed_data = pca_algorithm(stand, 2)
# plt.plot(np.cumsum(explained_variance) / np.sum(explained_variance))
# plt.xlabel('Number of components')
# plt.ylabel('Cumulative explained variance')
# plt.show()