import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

#  t-Distributed Stochastic Neighbor Embedding (t-SNE)
# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Standardize the data (optional but recommended for t-SNE)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE with 2 components
# Note: In newer sklearn versions, use 'max_iter' instead of 'n_iter'
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

print(f"Perplexity used: {tsne.perplexity}")
print(f"Number of iterations: {tsne.max_iter}")

# Create the scatter plot
plt.figure(figsize=(10, 8))

# Colors for each class
colors = ['red', 'green', 'blue']
labels = ['Setosa', 'Versicolor', 'Virginica']

# Plot each class
for i, label in enumerate(labels):
    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1],
               c=colors[i], label=label, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

# Add labels and title
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE on Iris Dataset (2 Components)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()