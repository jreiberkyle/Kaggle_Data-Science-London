'''
Created on Dec 24, 2013

@author: jenn6701
'''

import numpy as np
from sklearn import svm
from sklearn import preprocessing as pp
from sklearn import cross_validation as cv
from sklearn.ensemble import ExtraTreesClassifier

dataDir = r"D:\NWR\Kaggle\DataScienceLondon"

class Preprocessor():
    def __init__(self, dataDir):
        self.dataDir = dataDir
        self.readData()
    
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
    
    def readData(self):
        self.train = np.loadtxt(open(self.dataDir+r"\train.csv", "rb"),
                           delimiter=",", skiprows=0)
        self.trainLabels = np.loadtxt(open(self.dataDir+r"\trainLabels.csv", "rb"),
                           delimiter=",", skiprows=0)
        self.test = np.loadtxt(open(self.dataDir+r"\test.csv", "rb"),
                           delimiter=",", skiprows=0)
         
class Classifier():
    def __init__(self, train, trainLabels, test):
        self.train = train
        self.trainLabels = trainLabels
        self.test = test
               
    def run(self):
        
        s = Submitter()
        #predictions = self.LinearSVM()
        #s.saveSubmission(predictions, "linearSVMSubmission")
        
        predictions = self.FancySVM()
        s.saveSubmission(predictions, "fancySVMSubmission")
   
        predictions = self.ModelSVM()
        s.saveSubmission(predictions, "modelSVMSubmission")
        
    def LinearSVM(self):
        clf = svm.LinearSVC()
        clf.fit(self.train, self.trainLabels)
        predictions = clf.predict(self.test)
        
        self.predictScore(clf)
        return predictions
    
    def FancySVM(self):
        clf = svm.SVC()
            
        clf.fit(self.train, self.trainLabels)
        predictions = clf.predict(self.test)
        
        self.predictScore(clf)
        return predictions

    def ModelSVM(self):
        
        #clf = svm.SVC(C=10.0,gamma=.01,kernel='rbf',probability=True)
        
        clf = svm.SVC(C=8,gamma=.17)
            
        clf.fit(self.train, self.trainLabels)
        predictions = clf.predict(self.test)
        
        self.predictScore(clf)
        return predictions

    def predictScore(self, classifier):
        scores = cv.cross_val_score(classifier, self.train, self.trainLabels, cv=30)
        print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))
 
class Submitter():
    def __init__(self):
        self.submissionDir = r"D:\NWR\Kaggle\DataScienceLondon\tut_svm"         

    def saveSubmission(self, predictions, name):
        solution = (predictions.astype(int))
        sub = np.column_stack((np.array(range(1, len(solution)+1)), solution))
        sub = np.vstack((["Id", "Solution"],sub))
        
        filePath = r"{}\{}.csv".format(self.submissionDir, name)
        np.savetxt(filePath, sub, fmt='%s', delimiter=",")
        print('Saved as {}'.format(filePath))      

if __name__ == '__main__':
    p = Preprocessor(dataDir)
    train, trainLabels, test = p.getOriginalValues()
    s = Classifier(train,trainLabels,test)
    s.run()
    
    train, trainLabels, test = p.getSelectedValues()
    s2 = Classifier(train,trainLabels,test)
    s.run()