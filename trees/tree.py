'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator
import matplotlib.pyplot as plt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

# 计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # 为所有可能的分类创建字典
    for featVec in dataSet: #the the number of unique elements and their occurance 
        currentLabel = featVec[-1]
        if currentLabel not in list(labelCounts.keys()): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt

# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
    
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in list(classCount.keys()): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(iter(classCount.items()), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            
    
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

def test1():
    dataSet, labels = createDataSet()
    res = calcShannonEnt(dataSet)
    print(res)

def test2():
    dataSet, labels = createDataSet()
    chooseBestFeatureToSplit(dataSet)
    print(dataSet)

def test3():
    dataSet, labels = createDataSet()
    print(createTree(dataSet, labels))

def test4():
    # 定义文本框和箭头方式
    decisionNode = dict(boxstyle='sawtooth', fc='0.8')
    leafNode = dict(boxstyle='round4', fc='0.8')
    arrow_args = dict(arrowstyle='<-')
    # 绘制待箭头的注解
    def plotNode(nodeText, centerPt, parentPt, nodeType):
        createPlot.axl.annotate(nodeText, xy=parentPt, xycoords='axes fraction',
        xytext=centerPt, textcoords='axes fraction', va='center', ha='center',
        bbox=nodeType, arrowprops=arrow_args)
    def createPlot():
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        createPlot.axl = plt.subplot(111, frameon=False)
        plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
        plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
        plt.show()
    createPlot()

def getNumLeafs(myTree):
    numLeafs = 0
    


if __name__ == "__main__":
    # test1()
    # test2()
    # test3()
    # test4()