'''
Created on May 9, 2014

@author: Jennifer Reiber Kyle
'''

import numpy as np

class DataReader():
    """Kaggle contest data reader
    
    Kaggle contest data reader. Data files are train.csv, trainLabels.csv, 
    and test.csv.
    """
    
    def __init__(self, dataDir):
        self.dataDir = dataDir
        self._readData()   

    def _readData(self):
        self.train = np.loadtxt(open(self.dataDir+r"\train.csv", "rb"),
                           delimiter=",", skiprows=0)
        self.trainLabels = np.loadtxt(open(self.dataDir+r"\trainLabels.csv", "rb"),
                           delimiter=",", skiprows=0)
        self.test = np.loadtxt(open(self.dataDir+r"\test.csv", "rb"),
                           delimiter=",", skiprows=0)
    
    def getData(self):
        return (self.train, self.trainLabels, self.test)

class Submitter():
    def __init__(self, submissionDir):
        self.submissionDir = submissionDir

    def saveSubmission(self, predictions, name):
        solution = (predictions.astype(int))
        sub = np.column_stack((np.array(range(1, len(solution)+1)), solution))
        sub = np.vstack((["Id", "Solution"],sub))
        
        filePath = r"{}\{}.csv".format(self.submissionDir, name)
        np.savetxt(filePath, sub, fmt='%s', delimiter=",")
        print('Saved as {}'.format(filePath))  
        