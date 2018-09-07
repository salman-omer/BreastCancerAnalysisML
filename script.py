import pandas

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
import numpy as np

# create dict of pre-processing columns with appropriate instructions
# pre-processing isntructions: one hot encoder, bin to avg, bin to low/high, label encoder
'''
    Instructions format:
    'column name' : Array of tuple [
                        Each tuple has pre-processing instruction and additional information
                    ]
                    
    Additional information depends on what the instruction is
    bin to avg => one map of classification for low value to use
                  second map of classification for high value of option
    one hot encoding => no additional information
    label encoder =>  no additional information
'''
instructions = {
    'age': [
        ('bin to avg',
         {
             '10-19': 15,
             '20-29': 25,
             '30-39': 35,
             '40-49': 45,
             '50-59': 55,
             '60-69': 65,
             '70-79': 75,
             '80-89': 85,
             '90-99': 95,
         }
         ),
        ('bin to low/high',
         {
             '10-19': 10,
             '20-29': 20,
             '30-39': 30,
             '40-49': 40,
             '50-59': 50,
             '60-69': 60,
             '70-79': 70,
             '80-89': 80,
             '90-99': 90,
         },
         {
             '10-19': 19,
             '20-29': 29,
             '30-39': 39,
             '40-49': 49,
             '50-59': 59,
             '60-69': 69,
             '70-79': 79,
             '80-89': 89,
             '90-99': 99,
         }
         )
    ],
    'menopause': [
        ('label encoder'),
        ('one hot encoding')
    ],
    'tumor-size': [
        ('bin to avg',
         {
             '0-4': 2,
             '5-9':7,
             '10-14': 12,
             '15-19': 17,
             '20-24': 22,
             '25-29': 27,
             '30-34': 32,
             '35-39': 37,
             '40-44': 42,
             '45-49': 47,
             '50-54': 52,
             '55-59': 57
         }
         ),
        ('bin to low/high',
         {
             '0-4': 0,
             '5-9': 5,
             '10-14': 10,
             '15-19': 15,
             '20-24': 20,
             '25-29': 25,
             '30-34': 30,
             '35-39': 35,
             '40-44': 40,
             '45-49': 45,
             '50-54': 50,
             '55-59': 55
         },
         {
             '0-4': 4,
             '5-9': 9,
             '10-14': 14,
             '15-19': 19,
             '20-24': 24,
             '25-29': 29,
             '30-34': 34,
             '35-39': 39,
             '40-44': 44,
             '45-49': 49,
             '50-54': 54,
             '55-59': 59
         }
         )
    ],
    'inv-nodes': [
        ('bin to avg',
         {
             '0-2': 1,
             '3-5': 4,
             '6-8': 7,
             '9-11': 10,
             '12-14': 13,
             '15-17': 16,
             '18-20': 19,
             '21-23': 22,
             '24-26': 25,
             '27-29': 28,
             '30-32': 31,
             '33-35': 34,
             '36-39': 37.5,
         }
         ),
        ('bin to low/high',
         {
             '0-2': 0,
             '3-5': 3,
             '6-8': 6,
             '9-11': 9,
             '12-14': 12,
             '15-17': 15,
             '18-20': 18,
             '21-23': 21,
             '24-26': 24,
             '27-29': 27,
             '30-32': 30,
             '33-35': 33,
             '36-39': 36,
         },
         {
             '0-2': 2,
             '3-5': 5,
             '6-8': 8,
             '9-11': 11,
             '12-14': 14,
             '15-17': 17,
             '18-20': 20,
             '21-23': 23,
             '24-26': 26,
             '27-29': 29,
             '30-32': 32,
             '33-35': 35,
             '36-39': 39,
         }
         )
    ],
    'node-caps': [
        ('label encoder')
    ],
    'deg-malig': [
        ('label encoder')
    ],
    'breast': [
        ('label encoder')
    ],
    'breast-quad': [
        ('label encoder'),
        ('one hot encoding')
    ],
    'irradiat': [
        ('label encoder')
    ]
}

# instruction is a tuple where the first element in the tuple is always the instruction itself

def execute_instruction(columnName, instruction, dataset):
    if instruction[0] == 'label encoder' or instruction == 'label encoder':
        label_encoder = LabelEncoder()
        X = label_encoder.fit_transform(dataset[columnName])
        dfEncoded = pandas.DataFrame(X, columns=[columnName + "_Encoder"])
        dataset = pandas.concat([dataset, dfEncoded], axis = 1)
    if instruction[0] == 'one hot encoding' or instruction == 'one hot encoding':
        label_binarizer = LabelBinarizer()
        X = label_binarizer.fit_transform(dataset[columnName].values)
        dfBinarized = pandas.DataFrame(X, columns=[columnName + "_" + str(int(i)) for i in range(X.shape[1])])
        dataset = pandas.concat([dataset, dfBinarized], axis=1)
    if instruction[0] == 'bin to avg' or instruction == 'bin to avg':
        X = np.zeros((dataset[columnName].__len__(), 1))
        for i in range(dataset[columnName].__len__()):
            X[i] = instruction[1].get(dataset[columnName][i])
        dfAvg = pandas.DataFrame(X, columns=[columnName + "_" + str(int(i)) for i in range(X.shape[1])])
        dataset = pandas.concat([dataset, dfAvg], axis=1)
    if instruction[0] == 'bin to low/high' or instruction == 'bin to low/high':
        X = np.zeros((dataset[columnName].__len__(), 2))
        for i in range(dataset[columnName].__len__()):
            X[i][0] = instruction[1].get(dataset[columnName][i])
            X[i][1] = instruction[2].get(dataset[columnName][i])
        dfLowHigh = pandas.DataFrame(X, columns=[columnName + "_" + str(int(i)) for i in range(X.shape[1])])
        dataset = pandas.concat([dataset, dfLowHigh], axis=1)
    if instruction[0] == 'numerical' or instruction == 'numerical':
        X = dataset[columnName].copy
        dfNumeric = pandas.DataFrame(X, columns=[columnName + "_" + str(int(i)) for i in range(X.shape[1])])
        dataset = pandas.concat([dataset, dfNumeric], axis=1)
    return dataset

# binary tree node contains data field,
# left and right pointer
class InstructionNode:
    # constructor to create tree node
    def __init__(self, columnName, instruction):
        self.columnName = columnName
        self.instruction = instruction
        self.children = []

def printPath(path):
    pathString = 'PATH: '
    for node in path:
        if node.instruction[0].__len__() > 1:
            pathString += node.columnName + ': ' + str(node.instruction[0]) + ' '
        else:
            pathString += node.columnName + ': ' + str(node.instruction) + ' '
    return pathString


# function to print all path from root
# to leaf in binary tree
def printPaths(root, path, paths):
    path.append(root)
    if root.children.__len__() == 0:
        paths.append(path)
        # printPath(path)
        return
    for child in root.children:
        printPaths(child, path.copy(), paths)

def addInstructionNodes(root, columnName, instructions):
    if root.children.__len__() > 0:
        for child in root.children:
            addInstructionNodes(child, columnName, instructions)
    else:
        for instruction in instructions:
            root.children.append(InstructionNode(columnName, instruction))


# generate multiple instruction trees based on the number of instructions for the first type
def generateInstructionTrees(paths):
    instructionTreeRoots = []
    for columnName in instructions:
        if instructionTreeRoots.__len__() > 0:
            for root in instructionTreeRoots:
                addInstructionNodes(root, columnName, instructions.get(columnName))
        else:
            for instruction in instructions.get(columnName):
                instructionTreeRoots.append((InstructionNode(columnName, instruction)))
    for root in instructionTreeRoots:
        printPaths(root, path=[], paths=paths)

def create_preprocessed_dataset(originalDataSet, path):
    copiedDataSet = originalDataSet.copy()
    for instructionData in path:
        copiedDataSet = execute_instruction(instructionData.columnName, instructionData.instruction, copiedDataSet)
    return copiedDataSet

class AlgorithmEvaluation:
    # constructor to create an algorithm data store
    def __init__(self, modelName, mean, standardDeviation, pathArray, X_train, X_validation, Y_train, Y_validation):
        self.modelName = model_selection
        self.mean = mean
        self.standardDeviation = standardDeviation
        self.pathArray = pathArray
        self.X_train = X_train
        self.X_validation = X_validation
        self.Y_train = Y_train
        self.Y_validation = Y_validation
        self.accuracyScore = 0

    def to_string(self):
        return "%s: %f (%f) %s" % (self.modelName, self.mean, self.standardDeviation, printPath(self.pathArray))

    def set_accuracy_score(self, accuracyScore):
        self.accuracyScore = accuracyScore


def validation_dataset_evaluation(algorithmInfo, fullOutput=False, numTopResults = 150):
    algorithmInfo.sort(key=lambda algorithm: algorithm.mean, reverse=True)

    for algorithm in algorithmInfo:
        print(algorithm.to_string())

    if algorithmInfo.__len__() < numTopResults:
        numTopResults = algorithmInfo.__len__()

    models = {}
    models['LR'] = LogisticRegression()
    models['KNN'] = KNeighborsClassifier()
    models['CART'] = DecisionTreeClassifier()
    models['RFC'] = RandomForestClassifier()
    models['NB'] = GaussianNB()
    models['SVM'] = SVC()

    for i in range(numTopResults):
        currentAlgorithm = algorithmInfo[i]
        algo = models.get(currentAlgorithm.modelName)
        algo.fit(currentAlgorithm.X_train, currentAlgorithm.Y_train)
        predictions = algo.predict(currentAlgorithm.X_validation)
        accuracyScore = accuracy_score(currentAlgorithm.Y_validation, predictions)
        currentAlgorithm.set_accuracy_score(accuracyScore)
        if fullOutput:
            print('\n\n VALIDATION DATASET FOR' + currentAlgorithm.to_string())
            print('ACCURACY SCORE: \n\t' + str(accuracyScore))
            print('CONFUSION MATRIX: \n' + str(confusion_matrix(currentAlgorithm.Y_validation, predictions)))
            print('CLASSIFICATION_REPORT: \n\t' + str(classification_report(currentAlgorithm.Y_validation, predictions)))

        algorithmInfo.sort(key= lambda algorithm: algorithm.accuracyScore, reverse = True)
        for i in range(numTopResults):
            currentAlgorithm = algorithmInfo[i]
            print('ACCURACY SCORE FOR: ' + currentAlgorithm.to_string())
            print('\t ' + str(currentAlgorithm.accuracyScore))


def dataset_evaluation(dataset, path, algorithmInfo):
    array = dataset.values
    X = array[:, 10:]
    Y = array[:,0]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

    # test options and evaluation metric
    seed = 7  # ensures the same kind of random number generation so we can directly compare the algorithms
    scoring = 'accuracy'

    # spot check algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('RFC', RandomForestClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))

    # evaluate each model in turn
    results = []
    names = []
    for (name, model) in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv= kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        algorithmInfo.append(AlgorithmEvaluation(name, cv_results.mean(), cv_results.std(), path, X_train, X_validation,
                                                 Y_train, Y_validation))

    return algorithmInfo


# run on validation dataset!

names = ['class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad',
         'irradiat']
breastCancerOriginalDataset = pandas.read_csv("breast-cancer.data", names=names, delimiter=',')

paths = []

generateInstructionTrees(paths)

algorithmInfo = []

for path in paths:
    print(printPath(path))
    algorithmInfo = dataset_evaluation(create_preprocessed_dataset(breastCancerOriginalDataset, path), path,
                                       algorithmInfo)

validation_dataset_evaluation(algorithmInfo)