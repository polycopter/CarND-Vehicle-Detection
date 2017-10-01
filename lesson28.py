
# code from lesson 28

import numpy as np
# Define a labels vector based on features lists
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

from sklearn.preprocessing import StandardScaler
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)  # NameError: name 'car_features' is not defined
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)	      

 #And now we're ready to shuffle and split the data into training and testing sets. 
# To do this we'll use the Scikit-Learn train_test_split() function,
# but it's worth noting that recently, this function moved from the sklearn.cross_validation package (in sklearn version <=0.17) 
# to the sklearn.model_selection package (in sklearn version >=0.18).

# In the quiz editor we're still running sklearn v0.17, so we'll import it like this:

# from sklearn.cross_validation import train_test_split
# But, if you are using scikit-learn >= 0.18 (yes, in carnd-term1) then use this:
from sklearn.model_selection import train_test_split

# train_test_split() performs both the shuffle and split of the data and you'll call it like this 
# (here choosing to initialize the shuffle with a different random state each time):

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)
    
from sklearn.svm import LinearSVC
# Use a linear SVC (support vector classifier)
svc = LinearSVC()
# Train the SVC
svc.fit(X_train, y_train)

# Then you can check the accuracy of your classifier on the test dataset like this:

print('Test Accuracy of SVC = ', svc.score(X_test, y_test))

# Or you can make predictions on a subset of the test data and compare directly with ground truth:

print('My SVC predicts: ', svc.predict(X_test[0:10].reshape(1, -1)))
print('For labels: ', y_test[0:10])

# Play with the parameter values spatial and histbin in the exercise below 
# to see how the classifier accuracy and training time vary with the feature vector input.

