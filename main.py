'''
Created on May 9, 2014

@author: Jennifer Reiber Kyle
'''

from kaggle import Submitter, DataReader
from prediction import Preprocessor, Classifier

def runPrediction(sourceDirectory,submitDirectory):
    reader = DataReader(dataDir)

    p = Preprocessor(*reader.getData())
    submitter = Submitter(submitDir)
    
    train, trainLabels, test = p.getOriginalValues()
    classifier = Classifier(train,trainLabels,test)
    
    ## Predict using Fancy SVM
    predictions = classifier.FancySVM()
    submitter.saveSubmission(predictions, "fancySVMSubmission")
    
    ## Predict using Model SVM
    predictions = classifier.ModelSVM()
    submitter.saveSubmission(predictions, "modelSVMSubmission")       
    
if __name__ == '__main__':
    dataDir = r"D:\NWR\Kaggle\DataScienceLondon"
    submitDir = r"D:\NWR\Kaggle\DataScienceLondon\tut_svm"
    
    runPrediction(dataDir, submitDir)
    