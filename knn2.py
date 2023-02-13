import argparse
import numpy as np
import pandas as pd
from math import sqrt
import statistics as st
import matplotlib as plt

def euclidean_distance(row1, row2):
    distances = np.sum((row1 - row2)**2, axis=1)
    distances = np.sqrt(distances)
    return distances

class Knn(object):
    k = 0    # number of neighbors to use
    x_train = None
    y = None

    

    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """

        self.k = k

    def train(self, xFeat, y):
        """
        Train the k-nn model.

        
        Returns
        -------
        self : object
        """
        self.x_train = pd.DataFrame(xFeat) # nd-array with shape n x d Training data 
        self.y_train = y # 1d array with shape n Array of labels associated with training data
        
        return self


    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted class label per sample
        """
        yHat = [] # variable to store the estimated class label

        for i in range(len(xFeat)):
            # calculate euclidean distance between one row at a time in new data and each row in training data
            distances = euclidean_distance(self.x_train, xFeat.iloc[i]) 
            
            ind = np.argsort(distances, axis=0) # sort distances but preserve index value for distances sorted
    
            ind = ind[:self.k] 
        
            df = pd.DataFrame(ind)

            x = 0
            for r in self.y_train[df[0]]: # plugs in index value of nearest neighbour into y
               
                x += r
            
            if x >=2:
                yHat.append(1)
            else:
                yHat.append(0)
          
            

        return yHat

        
        
def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape n
        Predicted class label for n samples
    yTrue : 1d-array with shape n
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    # TODO calculate the accuracy
    acc = 0

    for i in range(len(yHat)):
        if yHat[i] == yTrue[i]:
            acc += 1
    acc = acc / float(len(yTrue))
    return acc
   
def plot(xtrain, ytrain, xtest, ytest):
    df = pd.DataFrame()
    train_accs = []
    test_accs = []
    k_values = []

    for k in range(1, 25): # change to 10 or N
        # create Knn object
        k_values.append(k)
        knn = Knn(k)

        # train model
        knn.train(xtrain, ytrain.iloc[:,0])

        # predict the training accuracy
        yHatTrain = knn.predict(xtrain)
        trainAcc = accuracy(yHatTrain, ytrain.iloc[:,0])
        train_accs.append(trainAcc)

        # predict the test accuracy
        yHatTest = knn.predict(xtest)
        testAcc = accuracy(yHatTest, ytest.iloc[:,0])
        test_accs.append(testAcc)

    # add accuracies to dataframe
    df['k'] = k_values
    df['train_acc'] = train_accs
    df['test_acc'] = test_accs

    print(df)

    # Plot Train Accuracy
    plt.plot(df['k'],df['train_acc'], color='red')
    plt.xticks(np.arange(min(df['k']), max(df['k']) + 1, 1.0))
    plt.title("Training Accuracy vs K Values")
    plt.xlabel("K Values")
    plt.ylabel("Accuracy (%)")
    plt.show()

    # Plot Test Accuracy
    plt.plot(df['k'],df['test_acc'], color='blue')
    plt.xticks(np.arange(min(df['k']), max(df['k']) + 1, 1.0))
    plt.title("Test Accuracy vs K Values")
    plt.xlabel("K Values")
    plt.ylabel("Accuracy (%)")
    plt.show()

    # Plot both
    plt.plot(df['k'], df['train_acc'], marker='.', color='r', label='train')
    plt.plot(df['k'], df['test_acc'], marker='+', color='b', label='test')
    plt.legend();
    # plt.xticks(np.arange(min(df['k']), max(df['k']) + 1, 1.0))
    plt.title("Accuracy vs K Values")
    plt.xlabel("K Values")
    plt.ylabel("Accuracy (%)")
    plt.show()

    return df

def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
