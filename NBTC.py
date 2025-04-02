print("NAIVE BAYES ENGLISH TEST CLASSIFICATION")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

# Disable SSL verification (workaround for SSL certificate error)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

sns.set()  # Use seaborn plotting style

# Load the dataset
data = fetch_20newsgroups()

# Get the text categories
text_categories = data.target_names

# Define the training set
train_data = fetch_20newsgroups(subset="train", categories=text_categories)

# Define the test set
test_data = fetch_20newsgroups(subset="test", categories=text_categories)

print("We have {} unique classes".format(len(text_categories)))
print("We have {} training samples".format(len(train_data.data)))
print("We have {} test samples".format(len(test_data.data)))

# Let’s have a look at some training data
# print(test_data.data[5])

# Build the model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model using the training data
model.fit(train_data.data, train_data.target)

# Predict the categories of the test data
predicted_categories = model.predict(test_data.data)

# Print predicted categories
print(np.array(test_data.target_names)[predicted_categories])

# Plot the confusion matrix
mat = confusion_matrix(test_data.target, predicted_categories)
sns.heatmap(mat.T, square=True, annot=True, fmt="d", xticklabels=train_data.target_names, yticklabels=train_data.target_names)
plt.xlabel("True labels")
plt.ylabel("Predicted labels")
plt.show()

# Print the accuracy score
print("The accuracy is {}".format(accuracy_score(test_data.target, predicted_categories)))


#output
NAIVE BAYES ENGLISH TEST CLASSIFICATION
We have 20 unique classes
We have 11314 training samples
We have 7532 test samples
['rec.autos' 'sci.crypt' 'alt.atheism' ... 'rec.sport.baseball'
 'comp.sys.ibm.pc.hardware' 'soc.religion.christian']
