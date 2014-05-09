'''
Created on Dec 24, 2013

@author: Jennifer Reiber Kyle
'''

from sklearn import svm
from sklearn import preprocessing as pp
from sklearn import cross_validation as cv
from sklearn.ensemble import ExtraTreesClassifier
   
class Preprocessor():
    def __init__(self, train, trainLabels, test):
        self.train = train
        self.trainLabels = trainLabels
        self.test = test
    
    def getOriginalValues(self):
        origValues = self.train, self.trainLabels, self.test
        return origValues
        
    def getScaledValues(self, doScaleIndependently = False):      
        if doScaleIndependently:
            scaledValues = pp.scale(self.train), self.trainLabels, pp.scale(self.test)
        else:      
            scaler = pp.StandardScaler().fit(self.test)
            scaledValues = scaler.transform(self.train), self.trainLabels, scaler.transform(self.test)
        return scaledValues
    
    def getSelectedValues(self):
        (train, trainLabels, test) = self.getScaledValues()
        
        selector = ExtraTreesClassifier(compute_importances=True, random_state=0)
        train = selector.fit_transform(train, trainLabels)
        
        return (train, trainLabels, test)
        test = selector.transform(test)
            
class Classifier():
    def __init__(self, train, trainLabels, test):
        self.train = train
        self.trainLabels = trainLabels
        self.test = test
        
    def LinearSVM(self):
        clf = svm.LinearSVC()
        clf.fit(self.train, self.trainLabels)
        predictions = clf.predict(self.test)
        
        self.predictScore(clf)
        return predictions
    
    def FancySVM(self):
        
        
        print("Running SVC with default model parameters.")
        clf = svm.SVC()
        clf.fit(self.train, self.trainLabels)
        predictions = clf.predict(self.test)
        
        self.predictScore(clf)
        return predictions

    def ModelSVM(self):
        
        #clf = svm.SVC(C=10.0,gamma=.01,kernel='rbf',probability=True)
        C = 8
        gamma = 0.01
        
        print("Running SVC with C = {} and gamma = {}.".format(C, gamma))
        clf = svm.SVC(C=C,gamma=gamma)   
        clf.fit(self.train, self.trainLabels)
        predictions = clf.predict(self.test)
        
        self.predictScore(clf)
        return predictions

    def predictScore(self, classifier):
        scores = cv.cross_val_score(classifier, self.train, self.trainLabels, cv=30)
        print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))

    
 
