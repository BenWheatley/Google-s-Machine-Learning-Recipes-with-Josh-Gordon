#!/usr/python

# Import/set up data
from sklearn import datasets

iris = datasets.load_iris()

x = iris.data
y = iris.target

# Split data into training and validation
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

# Compare various classifiers
# First, decision tree:
from sklearn import tree
tree_classifier = tree.DecisionTreeClassifier()

# Second, KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
neighbors_classifier = KNeighborsClassifier()

# Method to apply to both classifiers
from sklearn.metrics import accuracy_score
def trainPredictAndShow(classifier, comment):
	classifier.fit(x_train, y_train)
	predictions = classifier.predict(x_test) # Predict whole set at once
	# Show accuracies
	print comment, accuracy_score(y_test, predictions)

# Show results
trainPredictAndShow(tree_classifier, "Decision Tree accuracy: ")
trainPredictAndShow(neighbors_classifier, "K-Neighbors accuracy: ")
