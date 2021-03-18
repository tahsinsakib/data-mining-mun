import json
import numpy as np


def main(fname):

    testData = np.loadtxt("../data/test.txt", dtype=int)
    with open("../data/dataDesc.txt", "r") as dataDescFile:
        dataDesc = json.load(dataDescFile)
    with open('../data/' + fname, "r") as treeFile:
        tree = json.load(treeFile)

    descIndices = dict()

    for i in range(1, len(testData)):
        descIndices[dataDesc[i][0]] = i
    
    testObjects = np.transpose(testData)

    objDictList = []
    for obj in testObjects:
        objDict = dict()
        for attrName, index in descIndices.items():
            objDict[attrName] = obj[index]
        objDictList.append(objDict)

    labels = []
    for obj in objDictList:
        node = tree
        while (type(node) == list):
            value = obj[node[0]]
            node = node[1][str(value)]
        labels.append(node)
    
    noOfErrs = 0
    for i in range(0, len(testData[0])):
        if(testData[0][i]!=labels[i]):
            noOfErrs += 1

    errRate = noOfErrs / len(testData[0])
    print("Error rate: " + str(errRate) + "\n")