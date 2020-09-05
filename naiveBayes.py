# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math
import numpy


TEST_SET_SIZE = 100
DATUM_WIDTH=28
DATUM_HEIGHT=28
SIZE = 10

class Model():
  def __init__(self, width, height, size):
    self.height = height
    self.width = width
    self.size = size
    # self.frequency = [0 for i in range(10)]
    # self.prob = [0 for i in range(10)]
    # self.test = [[0 for i in range(self.width)] for j in range(self.height)]
    self.count = [[[float(0) for i in range(self.height)] for j in range(self.width)] for k in range(size)]
    self.data = [[[[float(0) for i in range(self.height)] for j in range(self.width)] for k in range(size)]for b in range(2)]
  
  # def findEdge(self, data):
  #   self.findTop(data)
  #   self.findLeft(data)
  #   self.findBottom(data)
  #   self.findRight(data)

  # def findTop(self, data):
  #   for i in range(DIGIT_DATUM_WIDTH):
  #     for j in range(DIGIT_DATUM_HEIGHT):
  #       if data[(i,j)] == 1:
  #         self.top = i
  #         return
  # def findLeft(self, data):
  #   for i in range(DIGIT_DATUM_HEIGHT):
  #     for j in range(DIGIT_DATUM_WIDTH):
  #       if data[(j,i)] == 1:
  #         self.left = i
  #         return
  # def findBottom(self, data):
  #   for i in range(DIGIT_DATUM_WIDTH):
  #     for j in range(DIGIT_DATUM_HEIGHT):
  #       if data[(DIGIT_DATUM_WIDTH-i,DIGIT_DATUM_HEIGHT-j)] == 1:
  #         self.bottom = DIGIT_DATUM_WIDTH-i
  #         return
  # def findRight(self, data):
  #   for i in range(DIGIT_DATUM_HEIGHT):
  #     for j in range(DIGIT_DATUM_WIDTH):
  #       if data[(DIGIT_DATUM_WIDTH-j,DIGIT_DATUM_HEIGHT-i)] == 1:
  #         self.right = DIGIT_DATUM_HEIGHT-i
  #         return


  

  

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels, thisWidth, thisHeight, num):
    """
    Outside shell to call your method. Do not modify this method.
    """  
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));

    DATUM_WIDTH = thisWidth
    DATUM_HEIGHT = thisHeight
    SIZE = num

    self.trainData = Model(thisWidth, thisHeight, num)
    self.arr = [0 for i in range(num)]
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"

    arr = [0 for i in range(self.trainData.size)]
    it = 0
    for tData in trainingData:
      tlabel = int(trainingLabels[it])
      for feature, currentBool in tData.items():
        ft = numpy.asarray(feature)
        i = ft[0]
        j = ft[1]
        self.trainData.data[currentBool][tlabel][i][j] += 1
        self.trainData.count[tlabel][i][j] += 1
      it += 1
      arr[tlabel] += 1
    arr /= numpy.linalg.norm(arr) 
    self.arr = arr
    # self.trainData.count +=1
    # self.trainData.frequency = arr
    for num in range(self.trainData.size):
      for feature in self.features:
        ft = numpy.asarray(feature)
        i = ft[0]
        j = ft[1]
        self.trainData.count[num][i][j] += 2
        self.trainData.data[1][num][i][j] += 1

    # for llabel in self.legalLabels:
    #   for lData in self.features:
    #     for i in range(self.trainData.width):
    #       for j in range(self.trainData.height):
    #         currentBool = lData[(j,i)]


    for i in range(self.trainData.width):
      for j in range(self.trainData.height):
        for num in range(self.trainData.size):
          # self.trainData.count[num][i][j] +=2
          for currentBool in range(2):
            if self.trainData.data[currentBool][num][i][j] != 0:
            # self.trainData.data[binary][num][i][j] += 
              temp = self.trainData.data[currentBool][num][i][j]
              self.trainData.data[currentBool][num][i][j] = float(self.k / 2 + temp) / (self.k + self.trainData.count[num][i][j])
        
    # # for i in range(10):
    # #   self.trainData.prob[i] = arr[i] / len(trainingLabels)

    # util.raiseNotDefined()
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    
    "*** YOUR CODE HERE ***"
    # probability = [0 for i in range(10)]
    # for i in range(10):
    #   probability[i] = math.log10(self.trainData.frequency[i])
    for n in range(self.trainData.size):
      logJoint[n] = math.log10(self.arr[n])
      for feature, currentBool in datum.items():
        ft = numpy.asarray(feature)
        i = ft[0]
        j = ft[1]
    # for i in range(self.trainData.width):
    #   for j in range(self.trainData.height):
        if currentBool == 0:
          # for num in range(self.trainData.size):
          
          logJoint[n] += math.log10(1-self.trainData.data[1][n][i][j])
        else:
          # for num in range(self.trainData.size):
          logJoint[n] += math.log10(self.trainData.data[1][n][i][j])
            

    # util.raiseNotDefined()
    
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds
    

    
      
