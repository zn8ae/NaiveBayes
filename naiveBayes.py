#!/usr/bin/python
import string
import sys
import os
import numpy as np
import nltk
from copy import deepcopy
from sklearn.naive_bayes import MultinomialNB
import math

###############################################################################
#Global data#
vocabulary = {"love":0, "wonderful":0, "great":0, "best":0, "superb":0, "still":0, "beautiful":0, 
    "bad":0, "worst":0, "stupid":0, "waste":0, "boring":0, "?":0, "!":0, "UNK":0}

lis = ["love", "wonderful", "great", "best", "superb", "still", "beautiful", 
                    "bad", "worst", "stupid", "waste", "boring", "?", "!", "UNK"]

def transfer(fileDj, vocabulary):
    BOWDj = deepcopy(vocabulary)
    f = open(fileDj, 'r')
    for line in f:
        line.replace("loved","love")
        line.replace("loves","love")
        line.replace("loving","love")
        tokens = line.split()
        for t in tokens:
            if t in BOWDj:
                BOWDj[t]+=1
            else:
                BOWDj['UNK']+=1 
    return BOWDj


def loadData(Path):

    Xtrain = np.empty(shape=[0,15])
    ytrain = []
    Xtest = np.empty(shape=[0,15])
    ytest = []

    #training data
    #read in pos documents 
    posPath = Path + "training_set/" +"pos"
    for filename in os.listdir(posPath):
        if filename.endswith(".txt"): 
            ytrain.append(1)
            filepath = os.path.join(posPath, filename)
            BOWDj = transfer(filepath, vocabulary)
            row = [] 
            for i in lis:
                row.append(BOWDj[i])
            Xtrain = np.vstack([Xtrain,row])

    #read in neg documents 
    negPath = Path + "training_set/" +"neg"
    for filename in os.listdir(negPath):
        if filename.endswith(".txt"): 
            ytrain.append(-1)
            filepath = os.path.join(negPath, filename)
            BOWDj = transfer(filepath, vocabulary)
            row = [] 
            for i in lis:
                row.append(BOWDj[i])
            Xtrain = np.vstack([Xtrain,row])

    #testing data
    #read in pos documents 
    posPath = Path + "test_set/" +"pos"
    for filename in os.listdir(posPath):
        if filename.endswith(".txt"): 
            ytest.append(1)
            filepath = os.path.join(posPath, filename)
            BOWDj = transfer(filepath, vocabulary)
            row = [] 
            for i in lis:
                row.append(BOWDj[i])
            Xtest = np.vstack([Xtest,row])
      

    #read in neg documents 
    negPath = Path + "test_set/" +"neg"
    for filename in os.listdir(negPath):
        if filename.endswith(".txt"): 
            ytest.append(-1)
            filepath = os.path.join(negPath, filename)
            BOWDj = transfer(filepath, vocabulary)
            row = [] 
            for i in lis:
                row.append(BOWDj[i])
            Xtest = np.vstack([Xtest,row])


    #Xtrain 1400 documents, 700 pos 700 neg
    #Xtest 600 entries 300 pos 300 neg
    return Xtrain, Xtest, ytrain, ytest


def naiveBayesMulFeature_train(Xtrain, ytrain):
    thetaPos = []
    thetaNeg = []

    sumPos = 0
    freqPos = [sum(x) for x in zip(*Xtrain[0:700])]
    for i in freqPos:
        sumPos+=i
    for i in range(0,15):
    	#with smoothing a = 1, numerator + 1, denominator + 15
        thetaPos.append((freqPos[i]+1)/float(sumPos+15))

    sumNeg = 0
    freqNeg = [sum(x) for x in zip(*Xtrain[700:1400])]
    for i in freqNeg:
        sumNeg+=i
    for i in range(0,15):
        thetaNeg.append((freqNeg[i]+1)/float(sumNeg+15))
    
    return thetaPos, thetaNeg


def naiveBayesMulFeature_test(Xtest, ytest,thetaPos, thetaNeg):
    yPredict = []
    #We only need to compare the product of P(wi|ci) to determine MLA
    length = len(Xtest)
    for i in range(0,length):
    	testArray = Xtest[i]
    	scorePos = 0
    	scoreNeg = 0
    	for i in range(0,15):
    		scorePos += (math.log(thetaPos[i]) * testArray[i])
    		scoreNeg += (math.log(thetaNeg[i]) * testArray[i])
    	if max(scorePos,scoreNeg) == scorePos:
    		yPredict.append(1)
    	else:
    		yPredict.append(-1)

    total = len(ytest) 
    correct = 0
    for i in range(0,total):
    	if yPredict[i]==ytest[i]:
    		correct+=1

    Accuracy = float(correct)/total
    return yPredict, Accuracy


def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
	clf = MultinomialNB()
	clf.fit(Xtrain, ytrain)
	yPredict = clf.predict(Xtest)

	total = len(ytest) 
	correct = 0
	for i in range(0,total):
		if(ytest[i]==yPredict[i]):
			correct+=1
	Accuracy = correct/float(total)
	return Accuracy


def naiveBayesMulFeature_testDirectOne(path,thetaPos, thetaNeg, vocabulary):
    #Directly calculate the sum of independent probabilities
    scorePos = 0
    scoreNeg = 0
    yPredict = 0
    f = open(path, 'r')
    for line in f:
        line.replace("loved","love")
        line.replace("loves","love")
        line.replace("loving","love")
        tokens = line.split()
        for t in tokens:
            if t in vocabulary:
                index = lis.index(t)
                scorePos += math.log(thetaPos[index])
                scoreNeg += math.log(thetaNeg[index])
            else:
                scorePos += math.log(thetaPos[14])
                scoreNeg += math.log(thetaNeg[14])
    if max(scorePos,scoreNeg)==scorePos:
    	yPredict = 1
    else:
    	yPredict = -1

    return yPredict


def naiveBayesMulFeature_testDirect(path,thetaPos, thetaNeg, vocabulary):
    yPredict = []
    correct = 0
    total = 0

    posPath = path + "pos"
    ylabel = 1
    for filename in os.listdir(posPath):
    	if filename.endswith(".txt"): 
        	total+=1 
        	filepath = os.path.join(posPath, filename)
        	pred = naiveBayesMulFeature_testDirectOne(filepath,thetaPos, thetaNeg, vocabulary)
        	yPredict.append(pred)
        	if pred == ylabel:
        		correct+=1
      

    negPath = path + "neg"
    ylabel = -1
    for filename in os.listdir(negPath):
        if filename.endswith(".txt"): 
            total+=1
            filepath = os.path.join(negPath, filename)
            pred = naiveBayesMulFeature_testDirectOne(filepath,thetaPos, thetaNeg, vocabulary)
            yPredict.append(pred)
            if pred == ylabel:
            	correct+=1

    Accuracy = float(correct)/total

    return yPredict, Accuracy


def naiveBayesBernFeature_train(Xtrain, ytrain):
	thetaPosTrue=[]
	posMat = Xtrain[0:700]
	index = 0
	for k in range(0,15):
		sum = 0
		for i in posMat:
			if i[index] > 0:
				sum+=1
	 	thetaPosTrue.append((float)(sum+1)/(700+2))
	 	index+=1

	thetaNegTrue=[]
	negMat = Xtrain[700:1400]
	index = 0
	for k in range(0,15):
		sum = 0
		for i in negMat:
			if i[index] > 0:
				sum+=1
	 	thetaNegTrue.append((float)(sum+1)/(700+2))
	 	index+=1

 	return thetaPosTrue, thetaNegTrue

    
def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
    yPredict = []

    length = len(Xtest)
    for i in range(0,length):
    	testArray = Xtest[i]
    	scorePos = 0
    	scoreNeg = 0
    	for i in range(0,15):
    		#P(W|C) = P(W1=T|C) * P(W2=F|C)...
    		if testArray[i] > 0:
	    		scorePos += math.log(thetaPosTrue[i])
	    		scoreNeg += math.log(thetaNegTrue[i])
    		else:
    			scorePos += math.log(1-thetaPosTrue[i])
	    		scoreNeg += math.log(1-thetaNegTrue[i])
    	if max(scorePos,scoreNeg) == scorePos:
    		yPredict.append(1)
    	else:
    		yPredict.append(-1)

    total = len(ytest) 
    correct = 0
    for i in range(0,total):
    	if yPredict[i]==ytest[i]:
    		correct+=1

    Accuracy = float(correct)/total
    return yPredict, Accuracy

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python naiveBayes.py dataSetPath testSetPath"
        sys.exit()

    print "--------------------"
    textDataSetsDirectoryFullPath = sys.argv[1]
    testFileDirectoryFullPath = sys.argv[2]


    Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)


    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    print "thetaPos =", thetaPos
    print "thetaNeg =", thetaNeg
    print "--------------------"

    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print "MNBC classification accuracy =", Accuracy

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print "Sklearn MultinomialNB accuracy =", Accuracy_sk

    yPredict, Accuracy = naiveBayesMulFeature_testDirect(testFileDirectoryFullPath, thetaPos, thetaNeg, vocabulary)
    print "Directly MNBC tesing accuracy =", Accuracy
    print "--------------------"

    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print "thetaPosTrue =", thetaPosTrue
    print "thetaNegTrue =", thetaNegTrue
    print "--------------------"

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print "BNBC classification accuracy =", Accuracy
    print "--------------------"

