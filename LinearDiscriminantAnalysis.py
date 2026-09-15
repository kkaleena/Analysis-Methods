import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

# Linear Discriminant Analysis(LDA)
# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Standardize the data (optional but recommended)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply LDA with 2 components
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# Print explained variance ratio
print(f"\nExplained variance ratio: {lda.explained_variance_ratio_}")
print(f"Total variance explained: {sum(lda.explained_variance_ratio_)*100:.2f}%")

# Create the scatter plot
plt.figure(figsize=(10, 8))

# Colors for each class
colors = ['red', 'green', 'blue']
labels = ['Setosa', 'Versicolor', 'Virginica']

# Plot each class
for i, label in enumerate(labels):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1],
               c=colors[i], label=label, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

# Add labels and title
plt.xlabel(f'First Linear Discriminant ({lda.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'Second Linear Discriminant ({lda.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('Linear Discriminant Analysis on Iris Dataset (2 Components)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print additional information
print(f"\nNumber of features originally: {X.shape[1]}")
print(f"Number of LDA components: {X_lda.shape[1]}")
print(f"LDA classes: {lda.classes_}")
