import numpy as np

def to_categorical(value, amount):
    categorical = np.zeros(amount)
    categorical[value] = 1
    return categorical

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def load_mnist_60k(train_percentage, normalize=False, shuffled=False):
    X = []
    Y = []
    f = open('./mnist_train.csv', 'r')
    header = f.readline()
    counter = 0
    while True:
        line = f.readline()
        counter += 1
        if line == '':
            break

        x = list(map(int, line.split(',')))
        y = to_categorical(x.pop(0), 10)

        X.append(np.array(x))
        Y.append(y)
    f.close()

    X, y = np.array(X), np.array(Y)

    if shuffled:
        X, y = unison_shuffled_copies(X,y)

    if normalize:
        X = X / X.max(axis=0)

    idx = int(len(X) * train_percentage)
    return (X[:idx], y[:idx]), (X[idx:], y[idx:])

def load_pima(train_percentage, normalize=False, shuffled=False):
    X = []
    Y = []
    f = open('./pima.tsv', 'r')
    header = f.readline()
    while True:
        line = f.readline()
        if line == '':
            break

        x = list(map(float, line.split('\t')))
        y = to_categorical(int(x.pop()), 2)

        X.append(np.array(x))
        Y.append(y)
    f.close()

    X, y = np.array(X), np.array(Y)

    if shuffled:
        X, y = unison_shuffled_copies(X,y)

    if normalize:
        X = X / X.max(axis=0)

    idx = int(len(X) * train_percentage)
    return (X[:idx], y[:idx]), (X[idx:], y[idx:])

def load_wine(train_percentage, normalize=False, shuffled=False):
    X = []
    Y = []
    f = open('./wine.data', 'r')

    while True:
        line = f.readline()
        if line == '':
            break

        x = list(map(float, line.split(',')))
        y = to_categorical(int(x.pop(0))-1, 3)

        X.append(np.array(x))
        Y.append(y)
    f.close()

    X, y = np.array(X), np.array(Y)

    if shuffled:
        X, y = unison_shuffled_copies(X,y)

    if normalize:
        X = X / X.max(axis=0)

    idx = int(len(X) * train_percentage)
    return (X[:idx], y[:idx]), (X[idx:], y[idx:])