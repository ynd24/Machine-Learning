import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import knn2


def standard_scale(xTrain, xTest):
    """
    Preprocess the training data to have zero mean and unit variance.
    The same transformation should be used on the test data. For example,
    if the mean and std deviation of feature 1 is 2 and 1.5, then each
    value of feature 1 in the test set is standardized using (x-2)/1.5.
    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data
    xTest : nd-array with shape m x d
        Test data
    Returns
    -------
    xTrain : nd-array with shape n x d
        Transformed training data with mean 0 and unit variance
    xTest : nd-array with shape m x d
        Transformed test data using same process as training.
    """
    # Create and fit Standard Scaler object
    scaler = StandardScaler()
    stdScale = scaler.fit(xTrain)

    # Apply transform to train and test input data
    xTrain = stdScale.transform(xTrain)
    xTest = stdScale.transform(xTest)

    # Convert to DataFrame
    xTrain = pd.DataFrame(xTrain)
    xTest = pd.DataFrame(xTest)

    return xTrain, xTest


def minmax_range(xTrain, xTest):
    """
    Preprocess the data to have minimum value of 0 and maximum
    value of 1.The same transformation should be used on the test data.
    For example, if the minimum and maximum of feature 1 is 0.5 and 2, then
    then feature 1 of test data is calculated as:
    (1 / (2 - 0.5)) * x - 0.5 * (1 / (2 - 0.5))
    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data
    xTest : nd-array with shape m x d
        Test data
    Returns
    -------
    xTrain : nd-array with shape n x d
        Transformed training data with min 0 and max 1.
    xTest : nd-array with shape m x d
        Transformed test data using same process as training.
    """
    # Create and fit MinMaxScaler object
    scaler = MinMaxScaler()
    mmScale = scaler.fit(xTrain)

    # Apply transformation to train and test input data
    xTrain = mmScale.transform(xTrain)
    xTest = mmScale.transform(xTest)

    # Convert to DataFrame
    xTrain = pd.DataFrame(xTrain)
    xTest = pd.DataFrame(xTest)

    return xTrain, xTest


def add_irr_feature(xTrain, xTest):
    """
    Add 2 features using Gaussian distribution with 0 mean,
    standard deviation of 1.
    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data
    xTest : nd-array with shape m x d
        Test data
    Returns
    -------
    xTrain : nd-array with shape n x (d+2)
        Training data with 2 new noisy Gaussian features
    xTest : nd-array with shape m x (d+2)
        Test data with 2 new noisy Gaussian features
    """
    # Irrelevant Features for xTrain
    for i in range(0,2):
        name = "irr_feature_train_" + str(i)
        arr = np.random.normal(0,1,len(xTrain))
        xTrain[name] = arr

    # Irrelevant Features for xTest
    for i in range(0,2):
        name = "irr_feature_test_" + str(i)
        arr = np.random.normal(0,1,len(xTest))
        xTest[name] = arr

    return xTrain, xTest

def knn_train_test(k, xTrain, yTrain, xTest, yTest):
    """
    Given a specified k, train the knn model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.
    Parameters
    ----------
    k : int
        The number of neighbors
    xTrain : nd-array with shape n x d
        Training data
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.
    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    model = knn2.Knn(k)
    model.train(xTrain, yTrain['label'])

    # predict the test dataset
    yHatTest = model.predict(xTest)
    return knn2.accuracy(yHatTest, yTest['label'])

def plot(xTrain, yTrain, xTest, yTest):
    # initialize result Data Frame and arrays
    df = pd.DataFrame()
    k_values = []
    no_pp_acc = []
    ss_acc = []
    mm_acc = []
    irr_feature_acc = []

    for k in range (1,26):
        # k values
        k_values.append(k)

        # no preprocessing
        acc1 = knn_train_test(k, xTrain, yTrain, xTest, yTest)
        no_pp_acc.append(acc1)

        # preprocess the data using standardization scaling
        xTrainStd, xTestStd = standard_scale(xTrain, xTest)
        acc2 = knn_train_test(k, xTrainStd, yTrain, xTestStd, yTest)
        ss_acc.append(acc2)

        # preprocess the data using min max scaling
        xTrainMM, xTestMM = minmax_range(xTrain, xTest)
        acc3 = knn_train_test(k, xTrainMM, yTrain, xTestMM, yTest)
        mm_acc.append(acc3)

        # add irrelevant features
        xTrainIrr, yTrainIrr = add_irr_feature(xTrain, xTest)
        acc4 = knn_train_test(k, xTrainIrr, yTrain, yTrainIrr, yTest)
        irr_feature_acc.append(acc4)

    # assign arrays to df
    df['k'] = k_values
    df['No Preprocessing'] = no_pp_acc
    df['Standard Scaling'] = ss_acc
    df['Min-Max Scaling'] = mm_acc
    df['Irrelevant Features'] = irr_feature_acc

    # check df
    # print(df)

    plt.plot(df['k'], df['No Preprocessing'], marker='.', color='r', label='No Preprocessing')
    plt.plot(df['k'], df['Standard Scaling'], marker='+', color='b', label='Standard Scaling')
    plt.plot(df['k'], df['Min-Max Scaling'], marker='+', color='g', label='Min-Max Scaling')
    plt.plot(df['k'], df['Irrelevant Features'], marker='+', color='y', label='Irrelevant Features')
    plt.legend()
    plt.xticks(np.arange(min(df['k']), max(df['k']) + 1, 1.0))
    plt.title("Accuracy vs K Values")
    plt.xlabel("K Values")
    plt.ylabel("Accuracy (%)")
    plt.show()

    return df

def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)

    # no preprocessing
    acc1 = knn_train_test(args.k, xTrain, yTrain, xTest, yTest)
    print("Test Acc (no-preprocessing):", acc1)
    # preprocess the data using standardization scaling
    xTrainStd, xTestStd = standard_scale(xTrain, xTest)
    acc2 = knn_train_test(args.k, xTrainStd, yTrain, xTestStd, yTest)
    print("Test Acc (standard scale):", acc2)
    # preprocess the data using min max scaling
    xTrainMM, xTestMM = minmax_range(xTrain, xTest)
    acc3 = knn_train_test(args.k, xTrainMM, yTrain, xTestMM, yTest)
    print("Test Acc (min max scale):", acc3)
    # add irrelevant features
    xTrainIrr, yTrainIrr = add_irr_feature(xTrain, xTest)
    acc4 = knn_train_test(args.k, xTrainIrr, yTrain, yTrainIrr, yTest)
    print("Test Acc (with irrelevant feature):", acc4)

"""
def main():
    # For Plots Only
    parser = argparse.ArgumentParser()
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
    plot(xTrain, yTrain, xTest, yTest)
"""""

if __name__ == "__main__":
    main()