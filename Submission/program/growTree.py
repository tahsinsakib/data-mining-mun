import json
import os
import numpy as np
import math


# recursive function call
def computeNodes(node, data, desc):

    if type(node) == int:
        return node

    # calculate gains for every attribute
    # select attribute with highest gain
    index = int(highestGainIndex(data))

    v = data[0][0]
    allSame = True

    for value in data[0]:
        if value != v:
            allSame = False
            break

    if allSame:
        return int(v)

    # store that attribute name in node
    node[0] = desc[index][0]
    # run algorithm on all attribute values
    branches = {}

    for branch in desc[index][1]:
        newData = []
        for i, value in enumerate(data[index]):
            if value == branch:
                newData.append(data[:, i])
        newData = np.transpose(np.asarray(newData))

        if len(newData) != 0:
            newData = np.delete(newData, index, 0)
        else:
            newData = data

        newDesc = list(desc)
        newDesc.pop(index)

        newNode = None

        if len(data) == 2:
            counts = np.bincount(newData[0])
            m = np.argmax(counts)

            return int(m)

        else:
            newNode = computeNodes([None, None], newData, newDesc)

        branches[int(branch)] = newNode

    node[1] = branches

    return node


def highestGainIndex(data):

    gains = []
    
    for i in range(1, len(data)):
        gains.append(gain(data[i], data[0]))

    m = max(gains)

    mIndex = 0
    for i in range(len(gains)):
        if(gains[i]==m):
            mIndex = i

    return mIndex + 1


def gain(values, labels):
    
    return (entropy(labels) - entropy2(values, labels))


def entropy(labels):

    prob = np.bincount(labels)/len(labels)

    e = 0
    for x in prob:
        if x != 0:
            e = e - x*np.log2(x)

    return e


def entropy2(values, labels):   # gives us weighted average entropy or entropy(S|A)

    values = np.ndarray.flatten(values)
    labels = np.ndarray.flatten(labels)

    prob2 = np.bincount(values)/len(values)
    lst0 = [[], [], []]
    for i in range(len(values)):
        if(values[i]==1): lst0[0].append(labels[i])
        elif(values[i]==2): lst0[1].append(labels[i])
        else: lst0[2].append(labels[i])

    e2 = 0
    for j in range(len(prob2)):
        if prob2[j]!= 0:
            e2 = e2 + prob2[j]*entropy(lst0[j-1])

    return e2


def main():
    filename = "treeOriginal.txt" 

    tree = [None, None]  # list
    trainingData = np.loadtxt("../data/train.txt").astype(int)  # numpy 2d array

    desc = None  # list
    with open('../data/dataDesc.txt') as f:
        desc = json.load(f)

    computeNodes(tree, trainingData, desc)

    with open("../data/" + str(filename), 'w') as f:
        json.dump(tree, f)

    return filename