import numpy as np
# You need to build your own model here instead of using well-built python packages such as sklearn

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# You can use the models form sklearn packages to check the performance of your own models

class BinaryClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPredictZero(BinaryClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

# IMPLEMENTED
class NaiveBayesClassifier(BinaryClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        self.prior = None
        self.likelihood = None
        # raise Exception("Must be implemented")
        

    def fit(self, X, Y):
        # Obtain number of sentences and features from X
        n_sentences, n_features = X.shape
        self.classes = np.unique(Y)
        n_classes = len(self.classes)

        # Initialize prior and likelihood probabilities
        self.prior = np.zeros(n_classes)
        self.likelihood = np.zeros((n_classes, n_features))

        # Compute prior and likelihood probabilities (with add-1 smoothing)
        for i, c in enumerate(self.classes):
            self.prior[i] = np.mean(Y == c)

        for i, c in enumerate(self.classes):
            X_c = X[Y == c]
            self.likelihood[i] = ((X_c.sum(axis=0)) + 1 / (np.sum(X_c.sum(axis=0)) + n_features))        
    
    def predict(self, X):
        # Initialize list of predicted class labels
        y_pred = []

        # Iterate over sentences in X
        for x in X:
            posteriors = []

            for i, c in enumerate(self.classes):
                # For class c ...
                # Compute the logs of the proir and likelihood probabilities
                prior = np.log(self.prior[i])
                likelihood = np.log(self.likelihood[i, :])

                # Compute posterior probability and append to list
                posterior = np.sum(likelihood * x) + prior
                posteriors.append(posterior)
            
            # Append the label with the highest log posterior probability
            y_pred.append(self.classes[np.argmax(posteriors)])
        
        # Return predicted class labels
        return np.array(y_pred)

# TODO: Implement this
class LogisticRegressionClassifier(BinaryClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self, learning_rate = 0.01, num_iters = 1000):
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.lamb = None
        self.theta = None
        # raise Exception("Must be implemented")

    def sigmoid(self, z):
        # Sigmoid function to ensure values are between 0 and 1
        return 1 / (1 + np.exp(-z))   

    def fit(self, X, Y):
        # Add your code here!
        raise Exception("Must be implemented")
        
    
    def predict(self, X):
        # Add your code here!
        raise Exception("Must be implemented")


# you can change the following line to whichever classifier you want to use for bonus
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()
