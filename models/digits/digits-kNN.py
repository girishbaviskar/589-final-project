

import numpy as np
import pandas as pd
from sklearn import datasets
from models.helpers.helper import read_data_file, splitArgumentsAndLabel, calculate_recall, calculate_precision, calculate_f1_score
from backprop_digits import calculate_precision_recall

#Shuffle the dataset
#sklearn train_test_split function by default shuffles the data so this is optional

def shuffleData(dataFrame):
    from sklearn.utils import shuffle
    return shuffle(dataFrame)
    #print(df_shuffled.head().to_string()) 
    #return df_shuffled



def splitTrainAndTest(X,y,test_size):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y,test_size = test_size)

def normalize(data):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

    
def calculateEuclidianDistance(X1, X2):
    sub = X1-X2
    subSq = sub ** 2
    sqSum = np.sum(subSq)
    dist = np.sqrt(sqSum) 
    return dist


def calculateAccuracy(listOfPredictedLabels, listOfActualLabels): 
    if(len(listOfPredictedLabels) == len(listOfActualLabels)):
        correctCount = 0;
        for index, label in enumerate(listOfPredictedLabels):
            #print(label)
            if label == listOfActualLabels[index]:
                correctCount += 1 
        return ((correctCount/len(listOfPredictedLabels)))


def calculate_confusion_matrix(y_test, y_pred):
    tp = tn = fp = fn = 0
    for i in range(len(y_test)):
        if y_test[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_test[i] == 0 and y_pred[i] == 0:
            tn += 1
        elif y_test[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_test[i] == 1 and y_pred[i] == 0:
            fn += 1
    return tp, tn, fp, fn

def plotGraph(xValues, yValues, title, xLable, yLabel, std_dev):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.xlabel(xLable)
    plt.ylabel(yLabel)
    #plt.plot(xValues, yValues)
    ax.errorbar(xValues, yValues,
            yerr=std_dev,
            fmt='-o', ecolor='black')
    plt.title(title)
    plt.show()



def calculate_std_deviation(X, mean):
    variance = sum([((x - mean) ** 2) for x in X]) / len(X)
    res = variance ** 0.5
    return res

def knn_model(trainData, testPoint, k, y_train):
    #calculate distances with all points in trainData
    distanceDict = {}
    classCount = [0,0,0,0,0,0,0,0,0,0]
    for index, x1 in enumerate(trainData):
        dist = 0
        dist = calculateEuclidianDistance(x1,testPoint)
        distanceDict.update({index:dist})
    #get top k neighbours
    sortedDict = sorted(distanceDict.items(), key=lambda x:x[1])
    topKNeighbours = sortedDict[:k]
    #print(topKNeighbours)
    for neighbour in topKNeighbours:
        if y_train[neighbour[0]] == 0:
            classCount[0] += 1
        elif y_train[neighbour[0]] == 1:
            classCount[1] += 1
        elif y_train[neighbour[0]] == 2:
            classCount[2] += 1
        elif y_train[neighbour[0]] == 3:
            classCount[3] += 1
        elif y_train[neighbour[0]] == 4:
            classCount[4] += 1
        elif y_train[neighbour[0]] == 5:
            classCount[5] += 1
        elif y_train[neighbour[0]] == 6:
            classCount[6] += 1
        elif y_train[neighbour[0]] == 7:
            classCount[7] += 1
        elif y_train[neighbour[0]] == 8:
            classCount[8] += 1
        elif y_train[neighbour[0]] == 9:
            classCount[9] += 1
        # else:
        #     classCount[2] += 1
    #find the class with majority
    maxValue = max(classCount)
    indexOfMax = classCount.index(maxValue)
    predictedLabel = ''
    if indexOfMax == 0:
        predictedLabel = 0
    elif indexOfMax == 1:
        predictedLabel = 1
    elif indexOfMax == 2:
        predictedLabel = 2
    elif indexOfMax == 3:
        predictedLabel = 3
    elif indexOfMax == 4:
        predictedLabel = 4
    elif indexOfMax == 5:
        predictedLabel = 5
    elif indexOfMax == 6:
        predictedLabel = 6
    elif indexOfMax == 7:
        predictedLabel = 7
    elif indexOfMax == 8:
        predictedLabel = 8
    elif indexOfMax == 9:
        predictedLabel = 9

    return predictedLabel


# def runKNNforOneValueOfK(k):
#     listOfPredictedLabels = []
#     for x in X_train:
#         predictedLabel = knn_model(X_train, x, k , y_train.to_numpy(), y_test)
#         listOfPredictedLabels.append(predictedLabel)
#         #print(predictedLabel)
#     acc = calculateAccuracy(listOfPredictedLabels, list(y_train[4]))
#     print(acc)

def runKNNforAllValuesOfK(dataSet, run_norm):
    digits = datasets.load_digits(return_X_y=True)
    digits_dataset_X = digits[0]
    dfret_x = pd.DataFrame(digits_dataset_X)
    digits_dataset_y = digits[1]
    dfret_y = pd.DataFrame(digits_dataset_y)
    dfret = pd.concat([dfret_x, dfret_y], axis=1)
    #run for all odd values of k from 0 to 50
    listOfAvgAccuracies = []
    listOfValuesOfK = []
    for k in range(50):
        if(k%2 != 0):
            accuracySum = 0
            f1Sum = 0
            listOfValuesOfK.append(k)
            avgAccu = 0
            for n in range(20):
                #shuffle and split here  
                df = shuffleData(dfret)
                X, y = splitArgumentsAndLabel(df)
                X_train, X_test, y_train, y_test = splitTrainAndTest(X, y, 0.2)
                if run_norm == True:
                    X_train = normalize(X_train)
                    X_test = normalize(X_test)
                else:
                    X_train = X_train.to_numpy()
                    X_test = X_test.to_numpy()
                if(dataSet == 'train'):
                    dataToCompute = X_train
                    dataToCompare = y_train[4]
                elif(dataSet == 'test'):
                    dataToCompute = X_test
                    dataToCompare = y_test
                listOfPredictedLabels = []
                for x in dataToCompute:
                    predictedLabel = knn_model(X_train, x, k, y_train.to_numpy())
                    listOfPredictedLabels.append(predictedLabel)
                precision, recall = calculate_precision_recall(list(dataToCompare.to_numpy().flatten()), listOfPredictedLabels)
                f1_score = calculate_f1_score(precision, recall)
                acc = calculateAccuracy(listOfPredictedLabels, list(dataToCompare.to_numpy().flatten()))
                accuracySum += acc
                f1Sum += f1_score
            avgAccu = accuracySum/20
            avgf1 = f1Sum/20
            print('Avg Acc for k= '+ str(k)+ ' - ' + str(avgAccu))
            print('Avg f1 for k= ' + str(k) + ' - ' + str(avgf1))
            listOfAvgAccuracies.append(avgAccu)
    mean_acc = sum(listOfAvgAccuracies)/len(listOfAvgAccuracies)
    std_dev = calculate_std_deviation(listOfAvgAccuracies, mean_acc)
    print(std_dev)
    plotGraph(listOfValuesOfK, listOfAvgAccuracies, 'Avg Accuracy vs value of k', 'K', 'Accuracy', std_dev)
    



runKNNforAllValuesOfK('test', True)










