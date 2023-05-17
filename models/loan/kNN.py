

import numpy as np
import pandas as pd
from helper import read_data_file, splitArgumentsAndLabel, calculate_recall, calculate_precision, calculate_f1_score




#import dataset
# def readDataFile():
#     df = pd.read_csv('../../datasets/loan.csv')
#     #print(df.head().to_string())
#
#     return df




#Shuffle the dataset
#sklearn train_test_split function by default shuffles the data so this is optional

def shuffleData(dataFrame):
    from sklearn.utils import shuffle
    return shuffle(dataFrame)
    #print(df_shuffled.head().to_string()) 
    #return df_shuffled


#
# def splitArgumentsAndLabel(df_shuffled):
#     X = df_shuffled.iloc[:, :-1]
#     y = df_shuffled.iloc[:, -1:]
#     return X, y




def splitTrainAndTest(X,y,test_size):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y,test_size = test_size)
    



#normalize
def normalize(data):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)
    



#normalize all parameters
#minTrainColumnValues = X_train.min(axis=0)
#maxTrainColumnValues = X_train.max(axis=0)
#print(minTrainColumnValues)
#print(maxTrainColumnValues)
#def normalize(data, minVal, maxVal):
#    normData=np.array;
#    for x in data: 
#        x = (x-minVal)/(maxVal-minVal) 
#        normData.append(x)
#    return normData
#ret = normalize(X_train[:,:1], minColumnValues[0], maxTrainColumnValues[0])
#print(type(ret))


#todo convert y to numerical class values?
# k-NN model
    #calculate euclidian distance
    #sort
    #take first k nearest one 
    #find out majority label from k 
  
    
def calculateEuclidianDistance(X1, X2):
    sub = X1-X2
    subSq = sub ** 2
    sqSum = np.sum(subSq)
    dist = np.sqrt(sqSum) 
    return dist


#print(calculateEuclidianDistance(X_train[0], X_train[1]))     
#print(X_train.shape)



def calculateAccuracy(listOfPredictedLabels, listOfActualLabels): 
    if(len(listOfPredictedLabels) == len(listOfActualLabels)):
        correctCount = 0;
        for index, label in enumerate(listOfPredictedLabels):
            #print(label)
            if label == listOfActualLabels[index]:
                correctCount += 1 
        return ((correctCount/len(listOfPredictedLabels)) * 100)



def calculate_confusion_matrix(y_test, y_pred):
    tp = tn = fp = fn = 0
    for i in range(len(y_test)):
        if y_test[i] == 'Y' and y_pred[i] == 'Y':
            tp += 1
        elif y_test[i] == 'N' and y_pred[i] == 'N':
            tn += 1
        elif y_test[i] == 'N' and y_pred[i] == 'Y':
            fp += 1
        elif y_test[i] == 'Y' and y_pred[i] == 'N':
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



#model definition
#classCount index vs classLabel
#0- Iris-setosa
#1- Iris-versicolor
#2- Iris-virginica
def knn_model(trainData, testPoint, k, y_train):
    #calculate distances with all points in trainData
    distanceDict = {}
    classCount = [0,0]
    for index, x1 in enumerate(trainData):
        dist = 0
        dist = calculateEuclidianDistance(x1,testPoint)
        distanceDict.update({index:dist})
    #get top k neighbours
    sortedDict = sorted(distanceDict.items(), key=lambda x:x[1])
    topKNeighbours = sortedDict[:k]
    #print(topKNeighbours)
    for neighbour in topKNeighbours:
        if y_train[neighbour[0]] == 'N':
            classCount[0] += 1
        elif y_train[neighbour[0]] == 'Y':
            classCount[1] += 1
        # else:
        #     classCount[2] += 1
    #find the class with majority
    maxValue = max(classCount)
    indexOfMax = classCount.index(maxValue)
    predictedLabel = ''
    if indexOfMax == 0:
        predictedLabel = 'N'
    elif indexOfMax == 1:
        predictedLabel = 'Y'
    # else:
    #     predictedLabel = 'Iris-virginica'
    
    return predictedLabel


def runKNNforOneValueOfK(k):
    listOfPredictedLabels = []
    for x in X_train:
        predictedLabel = knn_model(X_train, x, k , y_train.to_numpy(), y_test)
        listOfPredictedLabels.append(predictedLabel)
        #print(predictedLabel)
    acc = calculateAccuracy(listOfPredictedLabels, list(y_train[4]))  
    print(acc)

def runKNNforAllValuesOfK(dataSet, run_norm): 
    dfret = read_data_file('../../datasets/loan.csv')
    dfret = dfret.drop('Loan_ID', axis=1)
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
                one_hot_encoded_data_X = pd.get_dummies(X, columns=['Gender', 'Married', 'Education', 'Self_Employed','Property_Area', 'Dependents'], dtype=int)
                X_train, X_test, y_train, y_test = splitTrainAndTest(one_hot_encoded_data_X, y, 0.2)
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
                #print(listOfPredictedLabels)
                tp, tn, fp, fn = calculate_confusion_matrix(list(dataToCompare.to_numpy().flatten()), listOfPredictedLabels)
                precision = calculate_precision(tp, fp)
                recall = calculate_recall(tp, fn)
                f1_score = calculate_f1_score(precision, recall)
                acc = calculateAccuracy(listOfPredictedLabels, list(dataToCompare.to_numpy().flatten()))
                accuracySum += acc
                f1Sum += f1_score
                #print('Acc for n= '+ str(n)+ ' - ' + str(acc))  
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










