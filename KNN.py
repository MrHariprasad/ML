import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

# Step 1: Load the Iris dataset
iris = datasets.load_iris()
data = iris.data
labels = iris.target

# Display specific samples
for i in [0, 79, 99, 101]:
    print(f"Index: {i:3}, Features: {data[i]}, Label: {labels[i]}")

# Step 2: Shuffle and split the dataset
np.random.seed(42)
indices = np.random.permutation(len(data))
n_training_samples = 12  # Number of training samples

learn_data = data[indices[:-n_training_samples]]
learn_labels = labels[indices[:-n_training_samples]]
test_data = data[indices[-n_training_samples:]]
test_labels = labels[indices[-n_training_samples:]]

# Step 3: Display training set samples
print("\nFirst samples of our training set:")
print(f"{'Index':7s} {'Data':20s} {'Label':3s}")
for i in range(5):
    print(f"{i:4d} {learn_data[i]} {learn_labels[i]:3}")

# Step 4: Display test set samples
print("\nFirst samples of our test set:")
print(f"{'Index':7s} {'Data':20s} {'Label':3s}")
for i in range(5):
    print(f"{i:4d} {test_data[i]} {test_labels[i]:3}")

# Step 5: Visualizing the dataset in 3D
colours = ("r", "b", "y")
X = [[] for _ in range(3)]  # Create 3 empty lists for 3 classes

# Separate data by class
for i in range(len(learn_data)):
    X[learn_labels[i]].append(learn_data[i])

# Convert to NumPy arrays
X = [np.array(X[i]) for i in range(3)]

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot data points for each class
for iclass in range(3):
    ax.scatter(X[iclass][:, 0], X[iclass][:, 1], X[iclass][:, 2], c=colours[iclass])

plt.title("3D Scatter Plot of Training Data")
plt.show()

# Step 6: Define Euclidean Distance Function
def distance(instance1, instance2):
    """Calculates the Euclidean distance between two instances"""
    return np.linalg.norm(np.subtract(instance1, instance2))

# Step 7: Find Nearest Neighbors
def get_neighbors(training_set, labels, test_instance, k):
    """Finds k nearest neighbors using Euclidean distance"""
    distances = []

    for index in range(len(training_set)):
        dist = distance(test_instance, training_set[index])
        distances.append((training_set[index], dist, labels[index]))

    # Sort distances in ascending order
    distances.sort(key=lambda x: x[1])
    
    # Select k nearest neighbors
    neighbors = distances[:k]
    return neighbors

# Step 8: Test the KNN Algorithm
k = 3  # Number of neighbors
for i in range(5):  # Check first 5 test samples
    neighbors = get_neighbors(learn_data, learn_labels, test_data[i], k)
    
    print(f"\nTest Sample Index: {i}")
    print(f"Test Data: {test_data[i]}")
    print(f"Actual Label: {test_labels[i]}")
    print(f"Neighbors: {neighbors}")
