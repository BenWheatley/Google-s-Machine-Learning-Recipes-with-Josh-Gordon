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

# Third, custom bare-bones re-implimentation of KNN
import random
class ScrappyKNN():
	def fit(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train
	
	def predict(self, x_test):
		predictions = []
		for row in x_test:
			label = random.choice(self.y_train)
			predictions.append(label)
		return predictions

scrappy_classifier = ScrappyKNN()

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
trainPredictAndShow(scrappy_classifier, "Hand-written KNN accuracy: ")
