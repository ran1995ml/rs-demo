import numpy as np
import tensorflow as tf
import pandas as pd
from data_loader import vectorize_dic
import argparse





if __name__ == "__main__":
    cols = ['user', 'item', 'rating', 'timestamp']
    train = pd.read_csv('../data/ml-100k/u1.base', delimiter='\t', names=cols)
    test = pd.read_csv('../data/ml-100k/u1.test', delimiter='\t', names=cols)

    train_dic = {'users': train['user'].values, 'items': train['item'].values}
    test_dic = {'users': test['user'].values, 'items': test['item'].values}

    X_train, ix = vectorize_dic(train_dic)
    X_test, ix = vectorize_dic(test_dic, ix, X_train.shape[1])

    y_train = train['rating'].values
    y_test = test['rating'].values

    X_train = X_train.todense()
    X_test = X_test.todense()

    n, p = X_train.shape
