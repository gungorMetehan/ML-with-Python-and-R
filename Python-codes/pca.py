# PCA w/ 2 components
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

## data set
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

## PCA  (2 components)
pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X)

## components
df_pca = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
df_pca['target'] = y
df_pca['species'] = [target_names[i] for i in y]

## data visualization
plt.figure(figsize = (8, 6))
sns.scatterplot(data = df_pca, x = 'PC1', y = 'PC2', hue = 'species', palette = 'Set1', s = 80)
plt.title('PCA of Iris Dataset', fontsize = 14)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
plt.legend(title = 'Species')
plt.grid(True)
plt.tight_layout()
plt.show()

# PCA w/ 3 components
from mpl_toolkits.mplot3d import Axes3D  # 3D visualization
import pandas as pd
import seaborn as sns

## data set
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

## PCA (3 components)
pca = PCA(n_components = 3)
X_pca = pca.fit_transform(X)

## components
df_pca = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2', 'PC3'])
df_pca['target'] = y
df_pca['species'] = [target_names[i] for i in y]

## data visualization (3D)
fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot(111, projection = '3d')

colors = ['r', 'g', 'b']
for i, target_name in enumerate(target_names):
    subset = df_pca[df_pca['target'] == i]
    ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'], 
               label = target_name, color = colors[i], s = 60, alpha = 0.7)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)')
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}% var.)')
ax.set_title('PCA of Iris Dataset')
ax.legend(title = 'Species')
plt.tight_layout()
plt.show()