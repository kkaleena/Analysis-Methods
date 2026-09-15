import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Principal Component Analysis(PCA)
# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA with 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Print explained variance
print(f"\nExplained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_)*100:.2f}%")

# Step 3: Create the scatter plot
plt.figure(figsize=(10, 8))

# Colors for each class
colors = ['red', 'green', 'blue']
labels = ['Setosa', 'Versicolor', 'Virginica']

# Plot each class
for i, label in enumerate(labels):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
               c=colors[i], label=label, s=50, alpha=0.7)

# Add labels and title
plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA on Iris Dataset (2 Components)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()