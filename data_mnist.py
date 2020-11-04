import pandas as pd
import numpy as np

np.random.seed(2)

from sklearn.model_selection import train_test_split
import itertools

# Load the data
def data_preprocessing():
    train = pd.read_csv("datasets/mnist/train.csv")
    test = pd.read_csv("datasets/mnist/test.csv")

    Y_train = train["label"]

    # Drop 'label' column
    X_train = train.drop(labels = ["label"],axis = 1) 

    # free some space
    del train

    # Normalize the data
    X_train = X_train / 255.0
    test = test / 255.0

    # Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
    X_train = X_train.values.reshape(-1,28,28,1)
    test = test.values.reshape(-1,28,28,1)  

    # Set the random seed
    random_seed = 2

    # Split the train and the validation set for the fitting
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

    return X_train, X_val, Y_train, Y_val, test
